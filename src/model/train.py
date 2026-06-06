from __future__ import annotations

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from .mlflow_tracking import (
    log_and_register_mlflow_models,
    log_mlflow_metrics_and_artifacts,
    log_mlflow_params,
)
from .preprocess import build_sequence_dataset, normalize_opensky_history, read_history
from .training_artifacts import save_artifacts_and_reports
from .training_config import (
    configure_mlflow_tracking,
    load_train_params,
    project_path,
    validate_compression_params,
)
from .training_metrics import (
    best_threshold_for_f1,
    class_weight_for_binary,
    classification_metrics,
    distance_metrics,
    last_position_baseline,
    position_predictions_from_mode,
    position_skill_score,
    position_targets_for_mode,
    rmse,
)
from .training_models import (
    build_on_ground_lstm,
    build_trajectory_lstm,
    load_mlflow,
    load_tensorflow,
    load_tf2onnx,
    scale_sequences,
    set_reproducibility,
    temporal_split_arrays,
    write_json,
)
from .training_report import build_metrics, print_training_summary


def train_opensky_models(params_path: str = "params.yaml") -> int:
    try:
        params = _prepare_params(params_path)
        tf, mlflow, tf2onnx = _load_runtime(params)
        data = _prepare_training_data(params)

        mlflow.set_tracking_uri(params["mlflow_tracking_uri"])
        mlflow.set_experiment(params["mlflow_experiment_name"])

        with mlflow.start_run(run_name="train_opensky_lstm"):
            log_mlflow_params(mlflow, params, data["feature_columns"], len(data["X_train"]), len(data["X_test"]))
            models = _fit_models(data, params, tf)
            evaluation = _evaluate_models(models, data, params)
            metrics = build_metrics(params, data, models, evaluation)
            artifact_paths = save_artifacts_and_reports(models, data, evaluation, metrics, params, tf, tf2onnx)
            metrics["mlflow"]["model_registry"] = log_and_register_mlflow_models(
                mlflow,
                models["trajectory_model"],
                models["ground_model"],
                params,
            )
            metrics_path = project_path(params["metrics_path"])
            write_json(metrics_path, metrics)
            log_mlflow_metrics_and_artifacts(mlflow, metrics, metrics_path, artifact_paths)

        print_training_summary(metrics, params)
        return 0
    except Exception as exc:
        print(f"Model training failed: {exc}")
        return 1


def _prepare_params(params_path: str) -> dict:
    params = load_train_params(params_path)
    validate_compression_params(params)
    return configure_mlflow_tracking(params)


def _load_runtime(params: dict):
    tf = load_tensorflow()
    mlflow = load_mlflow()
    tf2onnx = load_tf2onnx()
    set_reproducibility(params["random_state"], tf)
    return tf, mlflow, tf2onnx


def _prepare_training_data(params: dict) -> dict:
    history = normalize_opensky_history(read_history(project_path(params["history_file"])))
    feature_columns = [column for column in params["feature_columns"] if column in history.columns]
    if not feature_columns:
        raise ValueError("No configured training feature columns are present in the history dataset.")

    X, y_position, y_ground, _ = build_sequence_dataset(
        history,
        feature_columns=feature_columns,
        window_size=params["window_size"],
        max_sequences=params["max_sequences"],
        max_sequence_gap_seconds=params["max_sequence_gap_seconds"],
    )
    if len(X) < 100:
        raise ValueError(f"Not enough sequences for LSTM training: {len(X)}")

    X_train, X_test, y_position_train, y_position_test, y_ground_train, y_ground_test = temporal_split_arrays(
        X,
        y_position,
        y_ground,
        params["test_size"],
    )
    X_train_raw = X_train.copy()
    X_test_raw = X_test.copy()
    X_train, X_test, feature_imputer, feature_scaler = scale_sequences(X_train, X_test)
    target_scaler = StandardScaler()
    y_position_train_target = position_targets_for_mode(
        y_position_train,
        X_train_raw,
        feature_columns,
        params["trajectory_target_mode"],
    )

    return {
        "feature_columns": feature_columns,
        "X_train": X_train,
        "X_test": X_test,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "y_position_train_scaled": target_scaler.fit_transform(y_position_train_target),
        "y_position_test": y_position_test,
        "y_ground_train": y_ground_train,
        "y_ground_test": y_ground_test,
        "feature_imputer": feature_imputer,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
    }


def _fit_models(data: dict, params: dict, tf) -> dict:
    input_shape = (data["X_train"].shape[1], data["X_train"].shape[2])
    trajectory_model = build_trajectory_lstm(input_shape, params, tf)
    ground_model = build_on_ground_lstm(input_shape, params, tf)
    ground_class_weight = class_weight_for_binary(data["y_ground_train"])

    trajectory_history = trajectory_model.fit(
        data["X_train"],
        data["y_position_train_scaled"],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        validation_split=params["validation_split"],
        callbacks=[_early_stopping(tf, params)],
        verbose=1,
    )
    ground_history = ground_model.fit(
        data["X_train"],
        data["y_ground_train"],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        validation_split=params["validation_split"],
        callbacks=[_early_stopping(tf, params)],
        class_weight=ground_class_weight,
        verbose=1,
    )
    return {
        "trajectory_model": trajectory_model,
        "ground_model": ground_model,
        "trajectory_history": trajectory_history,
        "ground_history": ground_history,
        "ground_class_weight": ground_class_weight,
        "input_shape": input_shape,
    }


def _early_stopping(tf, params: dict):
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=params["patience"],
        restore_best_weights=True,
    )


def _evaluate_models(models: dict, data: dict, params: dict) -> dict:
    position_pred_scaled = models["trajectory_model"].predict(data["X_test"], verbose=0)
    position_pred_raw = data["target_scaler"].inverse_transform(position_pred_scaled)
    position_pred = position_predictions_from_mode(
        position_pred_raw,
        data["X_test_raw"],
        data["feature_columns"],
        params["trajectory_target_mode"],
    )
    ground_probability = models["ground_model"].predict(data["X_test"], verbose=0).reshape(-1)
    ground_threshold, best_test_f1 = best_threshold_for_f1(data["y_ground_test"], ground_probability)
    ground_pred = (ground_probability >= ground_threshold).astype(int)

    position_mae = mean_absolute_error(data["y_position_test"], position_pred, multioutput="raw_values")
    position_rmse = [
        rmse(data["y_position_test"][:, index], position_pred[:, index])
        for index in range(position_pred.shape[1])
    ]
    position_distance_metrics = distance_metrics(data["y_position_test"], position_pred)
    baseline_position_pred = last_position_baseline(data["X_test_raw"], data["feature_columns"])
    baseline_distance_metrics = (
        distance_metrics(data["y_position_test"], baseline_position_pred)
        if baseline_position_pred is not None
        else None
    )

    return {
        "position_mae": position_mae,
        "position_rmse": position_rmse,
        "position_distance_metrics": position_distance_metrics,
        "baseline_distance_metrics": baseline_distance_metrics,
        "position_skill_score": position_skill_score(
            position_distance_metrics["mae_distance_m"],
            baseline_distance_metrics["mae_distance_m"] if baseline_distance_metrics else None,
        ),
        "ground_threshold": ground_threshold,
        "best_test_f1": best_test_f1,
        "ground_metrics": classification_metrics(data["y_ground_test"], ground_pred, ground_probability),
    }


if __name__ == "__main__":
    raise SystemExit(train_opensky_models())
