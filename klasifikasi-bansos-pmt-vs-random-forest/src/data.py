from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import (
    EXPECTED_SOURCE_COLUMN_COUNT,
    MODEL_FEATURES,
    NEGATIVE_LABEL,
    NUMERIC_FEATURES,
    POSITIVE_LABEL,
    RAW_FEATURES,
    REFERENCE_DATE,
    SOURCE_COLUMNS,
    TARGET_COLUMN,
)


def normalize_column_name(value: object) -> str:
    return str(value).replace("\n", " ").strip().upper()


def _contains_header(values: Iterable[object]) -> bool:
    normalized = {normalize_column_name(value) for value in values if pd.notna(value)}
    required_markers = {"TANGGAL_LAHIR", "PENDIDIKAN", TARGET_COLUMN}
    return required_markers.issubset(normalized)


def _detect_header_row(path: Path) -> int:
    preview = pd.read_excel(path, header=None, nrows=20)
    for row_index in range(len(preview)):
        if _contains_header(preview.iloc[row_index].tolist()):
            return row_index
    raise ValueError("Baris header dataset tidak dapat ditemukan.")


def _validate_unique_columns(columns: list[str]) -> None:
    duplicates = sorted({column for column in columns if columns.count(column) > 1})
    if duplicates:
        raise ValueError(
            "Terdapat nama kolom ganda setelah normalisasi: "
            + ", ".join(duplicates)
        )


def load_dataset(
    path: str | Path,
    *,
    require_full_schema: bool = True,
) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {dataset_path}")

    header_row = _detect_header_row(dataset_path)
    dataframe = pd.read_excel(dataset_path, header=header_row)
    normalized_columns = [normalize_column_name(column) for column in dataframe.columns]
    _validate_unique_columns(normalized_columns)
    dataframe.columns = normalized_columns
    dataframe = dataframe.dropna(how="all").copy()

    unnamed_columns = [column for column in dataframe.columns if column.startswith("UNNAMED:")]
    if unnamed_columns:
        dataframe = dataframe.drop(columns=unnamed_columns)

    required_columns = SOURCE_COLUMNS if require_full_schema else [*RAW_FEATURES, TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError("Kolom wajib belum lengkap: " + ", ".join(missing_columns))

    if require_full_schema:
        unexpected_columns = [column for column in dataframe.columns if column not in SOURCE_COLUMNS]
        if unexpected_columns:
            raise ValueError(
                "Dataset memiliki kolom tambahan yang tidak termasuk skema final: "
                + ", ".join(unexpected_columns)
            )
        dataframe = dataframe[SOURCE_COLUMNS].copy()
        if dataframe.shape[1] != EXPECTED_SOURCE_COLUMN_COUNT:
            raise ValueError(
                "Jumlah kolom sumber tidak sesuai. "
                f"Diharapkan {EXPECTED_SOURCE_COLUMN_COUNT}, diperoleh {dataframe.shape[1]}."
            )

    normalized_target = dataframe[TARGET_COLUMN].astype("string").str.strip().str.upper()
    valid_target = normalized_target.isin({POSITIVE_LABEL, NEGATIVE_LABEL})
    if not valid_target.all():
        invalid_values = sorted(
            normalized_target.loc[~valid_target].dropna().astype(str).unique().tolist()
        )
        invalid_text = ", ".join(invalid_values) or "nilai kosong"
        raise ValueError(f"Label target tidak valid: {invalid_text}")

    dataframe[TARGET_COLUMN] = normalized_target
    return dataframe.reset_index(drop=True)


def calculate_age(series: pd.Series, reference_date: str = REFERENCE_DATE) -> pd.Series:
    birth_dates = pd.to_datetime(series, errors="coerce")
    reference = pd.Timestamp(reference_date)

    age = reference.year - birth_dates.dt.year
    birthday_not_reached = (reference.month < birth_dates.dt.month) | (
        (reference.month == birth_dates.dt.month)
        & (reference.day < birth_dates.dt.day)
    )
    return pd.to_numeric(age - birthday_not_reached.astype("Int64"), errors="coerce")


def _normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "NULL": pd.NA})
    )


def build_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in RAW_FEATURES if column not in dataframe.columns]
    if missing_columns:
        raise ValueError("Kolom prediktor belum lengkap: " + ", ".join(missing_columns))

    features = pd.DataFrame(index=dataframe.index)
    features["USIA"] = calculate_age(dataframe["TANGGAL_LAHIR"])

    for column in RAW_FEATURES:
        if column == "TANGGAL_LAHIR":
            continue
        if column in NUMERIC_FEATURES:
            features[column] = pd.to_numeric(dataframe[column], errors="coerce")
        else:
            features[column] = _normalize_text(dataframe[column])

    return features[MODEL_FEATURES]


def encode_target(series: pd.Series) -> pd.Series:
    mapping = {NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}
    encoded = series.astype("string").str.strip().str.upper().map(mapping)
    if encoded.isna().any():
        raise ValueError("Ditemukan label target di luar LAYAK dan TIDAK LAYAK.")
    return encoded.astype(int)
