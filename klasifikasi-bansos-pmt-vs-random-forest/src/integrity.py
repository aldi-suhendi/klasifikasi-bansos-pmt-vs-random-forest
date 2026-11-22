from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(features: pd.DataFrame, target: pd.Series) -> str:
    frame = features.copy()
    frame["__TARGET__"] = pd.Series(target, index=features.index).astype("Int64")

    normalized = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            normalized[column] = pd.to_numeric(series, errors="coerce")
        else:
            normalized[column] = (
                series.astype("string")
                .fillna("<NA>")
                .str.strip()
                .str.upper()
            )

    payload = normalized.to_csv(
        index=False,
        na_rep="<NA>",
        float_format="%.17g",
        lineterminator="\n",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_artifact_hashes(
    model_dir: str | Path,
    metadata: dict,
) -> None:
    directory = Path(model_dir)
    expected = metadata.get("artifacts", {}).get("sha256", {})
    if not expected:
        raise ValueError("Metadata model belum memuat hash SHA-256 artefak.")

    for filename, expected_hash in expected.items():
        path = directory / filename
        if not path.exists():
            raise FileNotFoundError(f"Artefak model tidak ditemukan: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Hash artefak tidak sesuai untuk {filename}. "
                "Model dan metadata kemungkinan tercampur atau berubah."
            )
