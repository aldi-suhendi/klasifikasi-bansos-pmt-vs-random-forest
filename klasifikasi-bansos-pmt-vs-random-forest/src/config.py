from __future__ import annotations

from typing import Final

TARGET_COLUMN: Final = "KETERANGAN"
POSITIVE_LABEL: Final = "LAYAK"
NEGATIVE_LABEL: Final = "TIDAK LAYAK"
REFERENCE_DATE: Final = "2025-12-31"
RANDOM_STATE: Final = 42
TEST_SIZE: Final = 0.20
CV_FOLDS: Final = 5
EXPECTED_SOURCE_COLUMN_COUNT: Final = 35

IDENTITY_ADMIN_COLUMNS: Final[list[str]] = [
    "NAMA_KEPALA_KELUARGA",
    "KECAMATAN",
    "DESA/KELURAHAN",
    "ALAMAT",
]

CONSTANT_EXCLUDED_COLUMNS: Final[list[str]] = [
    "SUMBER_PENERANGAN_UTAMA",
    "PERAHU",
    "KAPAL_MOTOR",
]

RAW_FEATURES: Final[list[str]] = [
    "TANGGAL_LAHIR",
    "PENDIDIKAN",
    "PEKERJAAN",
    "JUMLAH-ART",
    "KEPEMILIKAN_RUMAH",
    "LANSIA",
    "DISABILITAS",
    "SAKIT_KRONIS",
    "LANTAI_TERLUAS",
    "DINDING_TERLUAS",
    "ATAP_TERLUAS",
    "DAYA_LISTRIK_TERPASANG",
    "BAHAN_BAKAR_MEMASAK",
    "FASILITAS_BAB",
    "JENIS_KLOSET",
    "PEMBUANGAN_AKHIR_TINJA",
    "SUMBER_AIR_MINUM",
    "LEMARI_ES",
    "AC",
    "TV_LAYAR_DATAR",
    "KOMPUTER/LAPTOP",
    "SEPEDA_MOTOR",
    "SEPEDA",
    "MOBIL",
    "SMARTPHONE",
    "TERNAK_BESAR",
    "TERNAK_KECIL",
]

SOURCE_COLUMNS: Final[list[str]] = [
    "NAMA_KEPALA_KELUARGA",
    "TANGGAL_LAHIR",
    "KECAMATAN",
    "DESA/KELURAHAN",
    "ALAMAT",
    "PENDIDIKAN",
    "PEKERJAAN",
    "JUMLAH-ART",
    "KEPEMILIKAN_RUMAH",
    "LANSIA",
    "DISABILITAS",
    "SAKIT_KRONIS",
    "LANTAI_TERLUAS",
    "DINDING_TERLUAS",
    "ATAP_TERLUAS",
    "SUMBER_PENERANGAN_UTAMA",
    "DAYA_LISTRIK_TERPASANG",
    "BAHAN_BAKAR_MEMASAK",
    "FASILITAS_BAB",
    "JENIS_KLOSET",
    "PEMBUANGAN_AKHIR_TINJA",
    "SUMBER_AIR_MINUM",
    "LEMARI_ES",
    "AC",
    "TV_LAYAR_DATAR",
    "KOMPUTER/LAPTOP",
    "SEPEDA_MOTOR",
    "SEPEDA",
    "MOBIL",
    "PERAHU",
    "KAPAL_MOTOR",
    "SMARTPHONE",
    "TERNAK_BESAR",
    "TERNAK_KECIL",
    TARGET_COLUMN,
]

MODEL_FEATURES: Final[list[str]] = [
    "USIA",
    *[column for column in RAW_FEATURES if column != "TANGGAL_LAHIR"],
]

NEW_PREDICTION_COLUMNS: Final[list[str]] = [
    column
    for column in SOURCE_COLUMNS
    if column not in CONSTANT_EXCLUDED_COLUMNS + [TARGET_COLUMN]
]

CONSTANT_DEFAULTS: Final[dict[str, object]] = {
    "SUMBER_PENERANGAN_UTAMA": "LISTRIK PLN",
    "PERAHU": 0,
    "KAPAL_MOTOR": 0,
}

NUMERIC_FEATURES: Final[list[str]] = [
    "USIA",
    "JUMLAH-ART",
    "LANSIA",
    "DAYA_LISTRIK_TERPASANG",
    "LEMARI_ES",
    "AC",
    "TV_LAYAR_DATAR",
    "KOMPUTER/LAPTOP",
    "SEPEDA_MOTOR",
    "SEPEDA",
    "MOBIL",
    "SMARTPHONE",
    "TERNAK_BESAR",
    "TERNAK_KECIL",
]

CATEGORICAL_FEATURES: Final[list[str]] = [
    column for column in MODEL_FEATURES if column not in NUMERIC_FEATURES
]

PMT_PARAM_GRID: Final[dict[str, list[object]]] = {
    "model__C": [0.01, 0.1, 1, 10, 100],
}

RF_PARAM_GRID: Final[dict[str, list[object]]] = {
    "model__n_estimators": [200],
    "model__max_depth": [None, 20],
    "model__min_samples_split": [2],
    "model__min_samples_leaf": [1, 2],
    "model__max_features": ["sqrt"],
}
