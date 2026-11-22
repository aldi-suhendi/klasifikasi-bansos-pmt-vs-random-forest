from __future__ import annotations

from pathlib import Path

from src.config import TARGET_COLUMN
from src.data import build_features, encode_target, load_dataset

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Data-Dummy.xlsx"


def main() -> None:
    print(f"Folder proyek : {ROOT}")
    print(f"Lokasi data   : {DATA}")
    print(f"File tersedia : {DATA.exists()}")
    if not DATA.exists():
        raise SystemExit(
            "Dataset tidak ditemukan. Ekstrak ulang ZIP tanpa memindahkan folder data."
        )

    raw = load_dataset(DATA, require_full_schema=True)
    features = build_features(raw)
    target = encode_target(raw[TARGET_COLUMN])

    print("Dataset berhasil divalidasi.")
    print(f"Observasi      : {len(raw):,}".replace(",", "."))
    print(f"Kolom awal     : {raw.shape[1]}")
    print(f"Prediktor      : {features.shape[1]}")
    print(f"Layak          : {int((target == 1).sum()):,}".replace(",", "."))
    print(f"Tidak layak    : {int((target == 0).sum()):,}".replace(",", "."))
    print(f"Nilai kosong   : {int(features.isna().sum().sum())}")
    print(f"Duplikat baris : {int(raw.duplicated().sum())}")


if __name__ == "__main__":
    main()
