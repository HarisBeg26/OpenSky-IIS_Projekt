from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def haversine_distance_m(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    lat_true = np.radians(y_true[:, 0].astype(float))
    lon_true = np.radians(y_true[:, 1].astype(float))
    lat_pred = np.radians(y_pred[:, 0].astype(float))
    lon_pred = np.radians(y_pred[:, 1].astype(float))

    delta_lat = lat_pred - lat_true
    delta_lon = lon_pred - lon_true
    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat_true) * np.cos(lat_pred) * np.sin(delta_lon / 2) ** 2
    )
    return 6_371_000 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def distance_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    distances = haversine_distance_m(y_true, y_pred)
    return {
        "mae_distance_m": float(np.mean(distances)),
        "median_distance_m": float(np.median(distances)),
        "p95_distance_m": float(np.percentile(distances, 95)),
        "rmse_distance_m": float(np.sqrt(np.mean(distances**2))),
    }


def last_position_baseline(X: np.ndarray, feature_columns: list[str]) -> np.ndarray | None:
    if "latitude" not in feature_columns or "longitude" not in feature_columns:
        return None
    latitude_index = feature_columns.index("latitude")
    longitude_index = feature_columns.index("longitude")
    return X[:, -1, [latitude_index, longitude_index]].astype(float)


def position_skill_score(model_error_m: float, baseline_error_m: float | None) -> float | None:
    if baseline_error_m is None or baseline_error_m <= 0:
        return None
    return float(1 - (model_error_m / baseline_error_m))


def position_targets_for_mode(
    y_position: np.ndarray,
    X_raw: np.ndarray,
    feature_columns: list[str],
    target_mode: str,
) -> np.ndarray:
    if target_mode == "absolute":
        return y_position
    if target_mode != "delta":
        raise ValueError("trajectory_target_mode must be either 'absolute' or 'delta'.")

    last_positions = last_position_baseline(X_raw, feature_columns)
    if last_positions is None:
        raise ValueError("Delta trajectory target requires latitude and longitude feature columns.")
    return y_position - last_positions


def position_predictions_from_mode(
    raw_prediction: np.ndarray,
    X_raw: np.ndarray,
    feature_columns: list[str],
    target_mode: str,
) -> np.ndarray:
    if target_mode == "absolute":
        return raw_prediction

    last_positions = last_position_baseline(X_raw, feature_columns)
    if last_positions is None:
        raise ValueError("Delta trajectory prediction requires latitude and longitude feature columns.")
    return last_positions + raw_prediction


def best_threshold_for_f1(y_true: np.ndarray, y_probability: np.ndarray) -> tuple[float, float]:
    y_true_int = y_true.astype(int)
    if len(np.unique(y_true_int)) < 2:
        return 0.5, 0.0

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        y_pred = (y_probability >= threshold).astype(int)
        score = f1_score(y_true_int, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, float(best_f1)


def class_weight_for_binary(y: np.ndarray) -> dict[int, float] | None:
    y_int = y.astype(int)
    positives = int(np.sum(y_int == 1))
    negatives = int(np.sum(y_int == 0))
    if positives == 0 or negatives == 0:
        return None
    total = positives + negatives
    return {
        0: total / (2 * negatives),
        1: total / (2 * positives),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, object]:
    y_true_int = y_true.astype(int)
    y_pred_int = y_pred.astype(int)
    matrix = confusion_matrix(y_true_int, y_pred_int, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "accuracy": float(accuracy_score(y_true_int, y_pred_int)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_int, y_pred_int)),
        "precision": float(precision_score(y_true_int, y_pred_int, zero_division=0)),
        "recall": float(recall_score(y_true_int, y_pred_int, zero_division=0)),
        "f1": float(f1_score(y_true_int, y_pred_int, zero_division=0)),
        "positive_rate_true": float(np.mean(y_true_int)),
        "positive_rate_predicted": float(np.mean(y_pred_int)),
        "mean_probability": float(np.mean(y_probability)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }
