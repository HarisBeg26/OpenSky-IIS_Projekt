from __future__ import annotations

import gzip
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .training_config import PROJECT_ROOT


def build_compression_report(
    artifact_groups: dict[str, dict[str, Path]],
    params: dict,
) -> tuple[dict, list[Path]]:
    compressed_artifacts: list[Path] = []
    model_rows = []

    for model_key, artifacts in artifact_groups.items():
        keras_path = artifacts["keras"]
        onnx_path = artifacts["onnx"]
        quantized_weights_path = artifacts["quantized_weights"]
        compressed_path = onnx_path.with_suffix(onnx_path.suffix + ".gz")
        if params["compression_enabled"]:
            _gzip_file(onnx_path, compressed_path)
            compressed_artifacts.append(compressed_path)

        keras_size = _file_size(keras_path)
        onnx_size = _file_size(onnx_path)
        compressed_size = _file_size(compressed_path)
        quantized_archive_size = _file_size(quantized_weights_path)
        quantized_archive_ratio = _ratio(quantized_archive_size, keras_size)
        model_rows.append(
            {
                "model_key": model_key,
                "strategy": "Post-training float16 weight quantization",
                "compression_method": params["compression_method"],
                "quantization_dtype": params["quantization_dtype"],
                "keras_path": _artifact_relpath(keras_path),
                "onnx_path": _artifact_relpath(onnx_path),
                "quantized_weights_path": _artifact_relpath(quantized_weights_path),
                "compressed_onnx_path": _artifact_relpath(compressed_path) if compressed_path.exists() else None,
                "keras_size_bytes": keras_size,
                "onnx_size_bytes": onnx_size,
                "quantized_weight_archive_size_bytes": quantized_archive_size,
                "compressed_onnx_size_bytes": compressed_size,
                "onnx_vs_keras_ratio": _ratio(onnx_size, keras_size),
                "quantized_archive_vs_keras_ratio": quantized_archive_ratio,
                "compressed_vs_onnx_ratio": _ratio(compressed_size, onnx_size),
                "estimated_storage_reduction_vs_keras": (
                    None if quantized_archive_ratio is None else float(1 - quantized_archive_ratio)
                ),
                "secondary_storage_compression": "gzip_on_onnx",
            }
        )

    best_reduction = [
        row["estimated_storage_reduction_vs_keras"]
        for row in model_rows
        if row["estimated_storage_reduction_vs_keras"] is not None
    ]
    return {
        "status": "available" if model_rows else "missing",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "compression_enabled": params["compression_enabled"],
        "primary_method": params["compression_method"],
        "quantization_dtype": params["quantization_dtype"],
        "purpose": (
            "Keras models remain available for local FastAPI inference. Float16 quantized weight archives "
            "demonstrate neural-network model compression, while ONNX remains the portable deployment format."
        ),
        "models": model_rows,
        "summary": {
            "model_count": len(model_rows),
            "best_storage_reduction_vs_keras": max(best_reduction) if best_reduction else None,
            "runtime_candidate": "Keras/ONNX",
            "storage_candidate": "float16 quantized weights (.npz)",
            "secondary_storage_candidate": "compressed ONNX (.onnx.gz)",
        },
    }, compressed_artifacts


def write_float16_quantized_weights(model, output_path: Path) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    manifest = []
    original_weight_size = 0
    quantized_weight_size = 0

    for index, weight in enumerate(model.get_weights()):
        original = np.asarray(weight)
        quantized = original.astype(np.float16)
        key = f"weight_{index}"
        payload[key] = quantized
        original_weight_size += int(original.nbytes)
        quantized_weight_size += int(quantized.nbytes)
        manifest.append(
            {
                "name": key,
                "shape": list(original.shape),
                "original_dtype": str(original.dtype),
                "quantized_dtype": "float16",
                "original_size_bytes": int(original.nbytes),
                "quantized_size_bytes": int(quantized.nbytes),
            }
        )

    np.savez_compressed(output_path, **payload)
    archive_size = _file_size(output_path)
    memory_ratio = _ratio(quantized_weight_size, original_weight_size)
    return {
        "path": _artifact_relpath(output_path),
        "method": "post_training_float16_weight_quantization",
        "weight_count": len(manifest),
        "original_weight_size_bytes": original_weight_size,
        "quantized_weight_size_bytes": quantized_weight_size,
        "archive_size_bytes": archive_size,
        "quantized_weight_memory_ratio": memory_ratio,
        "estimated_weight_memory_reduction": None if memory_ratio is None else float(1 - memory_ratio),
        "manifest": manifest,
    }


def _gzip_file(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("rb") as source, gzip.open(output_path, "wb", compresslevel=9) as target:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            target.write(chunk)


def _file_size(path: Path) -> int | None:
    return path.stat().st_size if path.exists() else None


def _ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    return float(numerator / denominator)


def _artifact_relpath(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))
