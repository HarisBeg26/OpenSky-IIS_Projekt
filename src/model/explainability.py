from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from .training_metrics import distance_metrics, position_predictions_from_mode


def build_explainability_report(
    trajectory_model,
    ground_model,
    X_test: np.ndarray,
    X_test_raw: np.ndarray,
    y_position_test: np.ndarray,
    y_ground_test: np.ndarray,
    target_scaler: StandardScaler,
    feature_columns: list[str],
    params: dict,
    ground_threshold: float,
) -> dict:
    if not params["explainability_enabled"]:
        return {
            "status": "disabled",
            "method": "permutation_feature_importance",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    sample_index = _sample_for_explainability(
        len(X_test),
        params["explainability_sample_size"],
        params["random_state"],
    )
    X_sample = X_test[sample_index].copy()
    X_raw_sample = X_test_raw[sample_index].copy()
    y_position_sample = y_position_test[sample_index]
    y_ground_sample = y_ground_test[sample_index].astype(int)
    repeats = max(1, params["explainability_repeats"])
    rng = np.random.default_rng(params["random_state"] + 17)

    base_position = _prediction_positions(
        trajectory_model,
        X_sample,
        X_raw_sample,
        target_scaler,
        feature_columns,
        params["trajectory_target_mode"],
    )
    base_distance_m = distance_metrics(y_position_sample, base_position)["mae_distance_m"]
    base_ground_probability = ground_model.predict(X_sample, verbose=0).reshape(-1)
    base_ground_pred = (base_ground_probability >= ground_threshold).astype(int)
    base_ground_f1 = float(f1_score(y_ground_sample, base_ground_pred, zero_division=0))

    rows = []
    for feature_index, feature_name in enumerate(feature_columns):
        trajectory_increases = []
        ground_f1_drops = []
        for _ in range(repeats):
            order = rng.permutation(len(X_sample))
            X_permuted = X_sample.copy()
            X_raw_permuted = X_raw_sample.copy()
            X_permuted[:, :, feature_index] = X_permuted[order, :, feature_index]
            X_raw_permuted[:, :, feature_index] = X_raw_permuted[order, :, feature_index]

            permuted_position = _prediction_positions(
                trajectory_model,
                X_permuted,
                X_raw_permuted,
                target_scaler,
                feature_columns,
                params["trajectory_target_mode"],
            )
            permuted_distance_m = distance_metrics(y_position_sample, permuted_position)["mae_distance_m"]
            trajectory_increases.append(float(permuted_distance_m - base_distance_m))

            permuted_ground_probability = ground_model.predict(X_permuted, verbose=0).reshape(-1)
            permuted_ground_pred = (permuted_ground_probability >= ground_threshold).astype(int)
            permuted_ground_f1 = float(f1_score(y_ground_sample, permuted_ground_pred, zero_division=0))
            ground_f1_drops.append(float(base_ground_f1 - permuted_ground_f1))

        trajectory_increase = float(np.mean(trajectory_increases))
        trajectory_std = float(np.std(trajectory_increases))
        ground_f1_drop = float(np.mean(ground_f1_drops))
        ground_f1_std = float(np.std(ground_f1_drops))
        rows.append(
            {
                "feature": feature_name,
                "trajectory_distance_increase_m": trajectory_increase,
                "trajectory_distance_increase_std_m": trajectory_std,
                "trajectory_relative_error_increase": float(trajectory_increase / max(base_distance_m, 1)),
                "on_ground_f1_drop": ground_f1_drop,
                "on_ground_f1_drop_std": ground_f1_std,
                "on_ground_relative_f1_drop": float(ground_f1_drop / max(base_ground_f1, 1e-9)),
                "combined_importance": float(max(0, trajectory_increase / max(base_distance_m, 1)) + max(0, ground_f1_drop)),
            }
        )

    by_trajectory = sorted(rows, key=lambda item: item["trajectory_distance_increase_m"], reverse=True)
    by_ground = sorted(rows, key=lambda item: item["on_ground_f1_drop"], reverse=True)
    top_n = params.get("explainability_top_n", 10)
    return {
        "status": "available",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "permutation_feature_importance",
        "method_note": (
            "Global permutation feature importance, not SHAP: each feature is shuffled across "
            "test examples and the resulting degradation in model performance is measured."
        ),
        "interpretation": _build_interpretation(
            rows,
            base_distance_m,
            base_ground_f1,
            int(len(X_sample)),
            repeats,
        ),
        "sample_size": int(len(X_sample)),
        "repeats": repeats,
        "baseline": {
            "trajectory_mae_distance_m": base_distance_m,
            "on_ground_f1": base_ground_f1,
            "on_ground_threshold": ground_threshold,
        },
        "feature_importance": rows,
        "top_trajectory_features": by_trajectory[:top_n],
        "top_on_ground_features": by_ground[:top_n],
    }


def _build_interpretation(
    rows: list[dict],
    base_distance_m: float,
    base_ground_f1: float,
    sample_size: int,
    repeats: int,
) -> dict:
    warnings = []
    coordinate_rows = [row for row in rows if row["feature"] in {"latitude", "longitude"}]
    coordinate_reliance = max(
        (row["trajectory_relative_error_increase"] for row in coordinate_rows),
        default=0,
    )
    if coordinate_reliance >= 5:
        warnings.append(
            "Trajectory error rises by more than 5x the baseline when a coordinate is shuffled. "
            "This indicates strong geographic dependence and should be validated on unseen aircraft or regions."
        )
    if base_ground_f1 < 0.5:
        warnings.append(
            "The baseline on-ground F1 is below 0.50, so its feature ranking is fragile and must not be read as a causal explanation."
        )
    if repeats < 3:
        warnings.append(
            "Fewer than three permutation repeats were used, so importance values may be unstable."
        )
    if sample_size < 200:
        warnings.append(
            "The explanation sample contains fewer than 200 sequences."
        )

    return {
        "quality": "caution" if warnings else "stable",
        "headline": (
            "Interpret with caution: the ranking exposes model reliance, not whether a feature is good or bad."
            if warnings
            else "The ranking is suitable as a global model-reliance summary."
        ),
        "scope": "global test-set explanation",
        "higher_means": "Shuffling the feature damaged test performance more, so the model relied on it more.",
        "near_zero_means": "The feature had little measurable influence on this test sample.",
        "negative_means": "Shuffling improved performance, which can indicate noise, redundancy, or sampling variation.",
        "not_causality": "Importance does not show causal influence, prediction direction, or feature quality.",
        "bar_scale": "Bars are normalized within each task and show relative ranking, not an accuracy score.",
        "baseline_trajectory_mae_distance_m": float(base_distance_m),
        "baseline_on_ground_f1": float(base_ground_f1),
        "coordinate_reliance_multiple": float(coordinate_reliance),
        "warnings": warnings,
    }


def _prediction_positions(
    trajectory_model,
    X_scaled: np.ndarray,
    X_raw: np.ndarray,
    target_scaler: StandardScaler,
    feature_columns: list[str],
    target_mode: str,
) -> np.ndarray:
    position_scaled = trajectory_model.predict(X_scaled, verbose=0)
    position_raw = target_scaler.inverse_transform(position_scaled)
    return position_predictions_from_mode(position_raw, X_raw, feature_columns, target_mode)


def _sample_for_explainability(length: int, sample_size: int, random_state: int) -> np.ndarray:
    if sample_size <= 0 or length <= sample_size:
        return np.arange(length)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(length, size=sample_size, replace=False))
