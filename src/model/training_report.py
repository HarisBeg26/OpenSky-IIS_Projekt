from __future__ import annotations

import numpy as np

from .training_config import project_path


def build_metrics(params: dict, data: dict, models: dict, evaluation: dict) -> dict:
    trajectory_history = models["trajectory_history"].history
    ground_history = models["ground_history"].history
    return {
        "training_sequences": int(len(data["X_train"])),
        "test_sequences": int(len(data["X_test"])),
        "window_size": params["window_size"],
        "feature_columns": data["feature_columns"],
        "mlflow": {
            "tracking_uri": params["mlflow_tracking_uri"],
            "experiment_name": params["mlflow_experiment_name"],
            "tracking_mode": params.get("mlflow_tracking_mode", "configured"),
            "fallback_reason": params.get("mlflow_fallback_reason"),
            "model_registry": {
                "enabled": params["register_models"],
                "trajectory_registered_model_name": params["trajectory_registered_model_name"],
                "on_ground_registered_model_name": params["on_ground_registered_model_name"],
            },
        },
        "models": {
            "trajectory_lstm": _trajectory_metrics(
                params,
                trajectory_history,
                evaluation,
            ),
            "on_ground_lstm": _ground_metrics(models, ground_history, evaluation),
        },
        "params": _serializable_params(params),
    }


def print_training_summary(metrics: dict, params: dict) -> None:
    trajectory = metrics["models"]["trajectory_lstm"]
    ground = metrics["models"]["on_ground_lstm"]
    print(f"Training sequences: {metrics['training_sequences']}, test sequences: {metrics['test_sequences']}")
    print(f"Window size: {params['window_size']}, features: {metrics['feature_columns']}")
    print(f"Trajectory MAE latitude: {trajectory['mae_latitude']:.6f}")
    print(f"Trajectory MAE longitude: {trajectory['mae_longitude']:.6f}")
    print(f"Trajectory mean distance error: {trajectory['mae_distance_m'] / 1000:.2f} km")
    if trajectory.get("baseline", {}).get("mae_distance_m") is not None:
        print(f"Baseline mean distance error: {trajectory['baseline']['mae_distance_m'] / 1000:.2f} km")
    if trajectory.get("skill_score_vs_baseline") is not None:
        print(f"Skill score vs baseline: {trajectory['skill_score_vs_baseline']:.3f}")
    print(f"On-ground accuracy: {ground['accuracy']:.6f}")
    print(f"On-ground F1: {ground['f1']:.6f}")
    _print_artifact_summary(metrics, params)


def _trajectory_metrics(params: dict, history: dict, evaluation: dict) -> dict:
    position_mae = evaluation["position_mae"]
    position_rmse = evaluation["position_rmse"]
    return {
        "task": "Predict next latitude and longitude from a sequence of aircraft states.",
        "type": "TensorFlow Keras LSTM",
        "target_columns": ["target_next_latitude", "target_next_longitude"],
        "target_mode": params["trajectory_target_mode"],
        "epochs_trained": len(history.get("loss", [])),
        "mae_latitude": float(position_mae[0]),
        "mae_longitude": float(position_mae[1]),
        "rmse_latitude": float(position_rmse[0]),
        "rmse_longitude": float(position_rmse[1]),
        **evaluation["position_distance_metrics"],
        "baseline": {
            "name": "last_observed_position",
            **(evaluation["baseline_distance_metrics"] or {}),
        },
        "skill_score_vs_baseline": evaluation["position_skill_score"],
        "final_validation_loss": float(history.get("val_loss", [np.nan])[-1]),
    }


def _ground_metrics(models: dict, history: dict, evaluation: dict) -> dict:
    return {
        "task": "Predict whether the aircraft will be on the ground in the next state.",
        "type": "TensorFlow Keras LSTM",
        "target_column": "target_next_on_ground",
        "epochs_trained": len(history.get("loss", [])),
        "decision_threshold": evaluation["ground_threshold"],
        "best_test_f1": evaluation["best_test_f1"],
        "class_weight": models["ground_class_weight"],
        **evaluation["ground_metrics"],
        "final_validation_loss": float(history.get("val_loss", [np.nan])[-1]),
    }


def _serializable_params(params: dict) -> dict:
    keys = [
        "test_size",
        "max_sequences",
        "window_size",
        "max_sequence_gap_seconds",
        "trajectory_target_mode",
        "random_state",
        "lstm_units",
        "dense_units",
        "dropout",
        "epochs",
        "batch_size",
        "validation_split",
        "patience",
        "onnx_opset",
        "local_mlflow_fallback",
        "require_remote_mlflow",
        "compression_enabled",
        "compression_method",
        "quantization_dtype",
        "explainability_enabled",
        "explainability_sample_size",
        "explainability_repeats",
        "explainability_top_n",
    ]
    return {key: params[key] for key in keys}


def _print_artifact_summary(metrics: dict, params: dict) -> None:
    best_reduction = metrics.get("compression", {}).get("summary", {}).get("best_storage_reduction_vs_keras")
    if best_reduction is not None:
        print(f"Best float16 quantized weight archive reduction vs Keras: {best_reduction:.2%}")
    if params["register_models"]:
        print(
            "Registered MLflow models: "
            f"{params['trajectory_registered_model_name']}, {params['on_ground_registered_model_name']}"
        )
    else:
        print(f"MLflow tracking mode: {params.get('mlflow_tracking_mode', 'configured')}")
    print(f"Explainability report: {project_path(params['explainability_report_path'])}")
    print(f"Saved ONNX models with opset: {params['onnx_opset']}")
    print(f"Saved models to: {project_path(params['models_dir'])}")
    print(f"Saved metrics to: {project_path(params['metrics_path'])}")
