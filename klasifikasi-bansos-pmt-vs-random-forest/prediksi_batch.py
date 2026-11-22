from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    NEGATIVE_LABEL,
    NEW_PREDICTION_COLUMNS,
    POSITIVE_LABEL,
)
from src.data import build_features, normalize_column_name
from src.environment import validate_environment
from src.integrity import validate_artifact_hashes
from src.model_io import load_joblib_model

BASE_DIR = Path(__file__).resolve().parent


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prediksi data baru menggunakan model final PMT dan Random Forest."
    )
    parser.add_argument(
        "input",
        help="File Excel atau CSV dengan 31 kolom sesuai template prediksi data baru.",
    )
    parser.add_argument(
        "--output",
        default="output/prediksi_data_baru.xlsx",
        help="Lokasi hasil prediksi.",
    )
    return parser.parse_args()


def read_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    if path.suffix.lower() == ".csv":
        dataframe = pd.read_csv(path)
    else:
        dataframe = pd.read_excel(path)

    normalized_columns = [normalize_column_name(column) for column in dataframe.columns]
    if len(set(normalized_columns)) != len(normalized_columns):
        raise ValueError("Terdapat nama kolom ganda setelah normalisasi.")
    dataframe.columns = normalized_columns
    dataframe = dataframe.dropna(how="all").reset_index(drop=True)
    if dataframe.empty:
        raise ValueError("File input tidak memiliki baris data.")

    missing = [column for column in NEW_PREDICTION_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError("Kolom data baru belum lengkap: " + ", ".join(missing))
    unexpected = [column for column in dataframe.columns if column not in NEW_PREDICTION_COLUMNS]
    if unexpected:
        raise ValueError(
            "File memiliki kolom tambahan yang tidak diperlukan: "
            + ", ".join(unexpected)
        )
    return dataframe[NEW_PREDICTION_COLUMNS].copy()


def main() -> None:
    validate_environment(include_gui=False)
    arguments = parse_arguments()

    input_path = Path(arguments.input)
    if not input_path.is_absolute():
        input_path = (BASE_DIR / input_path).resolve()
    output_path = Path(arguments.output)
    if not output_path.is_absolute():
        output_path = (BASE_DIR / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_dir = BASE_DIR / "output" / "model"
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_artifact_hashes(model_dir, metadata)

    source = read_input(input_path)
    X = build_features(source)

    pmt = load_joblib_model(model_dir / "model_pmt.joblib")
    rf = load_joblib_model(model_dir / "model_random_forest.joblib")

    pmt_probability = pmt.predict_proba(X)[:, 1]
    rf_probability = rf.predict_proba(X)[:, 1]
    pmt_prediction = (pmt_probability >= metadata["pmt"]["threshold"]).astype(int)
    rf_prediction = (
        rf_probability >= metadata["random_forest"]["threshold"]
    ).astype(int)

    result = source.copy()
    result["PROBABILITAS_PMT"] = pmt_probability
    result["CUT_OFF_PMT"] = metadata["pmt"]["threshold"]
    result["PREDIKSI_PMT"] = np.where(
        pmt_prediction == 1,
        POSITIVE_LABEL,
        NEGATIVE_LABEL,
    )
    result["PROBABILITAS_RANDOM_FOREST"] = rf_probability
    result["CUT_OFF_RANDOM_FOREST"] = metadata["random_forest"]["threshold"]
    result["PREDIKSI_RANDOM_FOREST"] = np.where(
        rf_prediction == 1,
        POSITIVE_LABEL,
        NEGATIVE_LABEL,
    )
    same_prediction = pmt_prediction == rf_prediction
    result["KETERANGAN_HASIL"] = np.where(
        ~same_prediction,
        "PERLU VERIFIKASI",
        np.where(
            rf_prediction == 1,
            "REKOMENDASI LAYAK",
            "REKOMENDASI TIDAK LAYAK",
        ),
    )
    result["KETERANGAN"] = np.where(
        ~same_prediction,
        "Hasil prediksi PMT dan Random Forest berbeda; perlu verifikasi lebih lanjut",
        np.where(
            rf_prediction == 1,
            "Kedua model memberikan rekomendasi Layak",
            "Kedua model memberikan rekomendasi Tidak Layak",
        ),
    )

    if output_path.suffix.lower() == ".csv":
        result.to_csv(output_path, index=False)
    else:
        if output_path.suffix.lower() != ".xlsx":
            output_path = output_path.with_suffix(".xlsx")
        result.to_excel(output_path, index=False)
    print(f"Prediksi selesai: {output_path}")


if __name__ == "__main__":
    main()
