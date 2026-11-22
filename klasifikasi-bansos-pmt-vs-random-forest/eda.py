from __future__ import annotations

import argparse
from pathlib import Path

from src.config import TARGET_COLUMN
from src.data import build_features, encode_target, load_dataset
from src.visualization import (
    plot_eda_categorical_assets,
    plot_eda_target_numeric,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Menghasilkan Gambar 4.1 dan Gambar 4.2 EDA."
    )
    parser.add_argument(
        "--input",
        default="data/Data-Dummy.xlsx",
        help="Lokasi dataset penelitian berlabel.",
    )
    parser.add_argument(
        "--output",
        default="output/gambar_bab4",
        help="Folder keluaran gambar EDA.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    project_dir = Path(__file__).resolve().parent

    input_path = Path(arguments.input)
    output_path = Path(arguments.output)
    if not input_path.is_absolute():
        input_path = project_dir / input_path
    if not output_path.is_absolute():
        output_path = project_dir / output_path

    input_path = input_path.resolve()
    output_path = output_path.resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    raw_data = load_dataset(input_path, require_full_schema=True)
    features = build_features(raw_data)
    target = encode_target(raw_data[TARGET_COLUMN])

    first_figure = output_path / "gambar_4_1_eda_target_numerik.png"
    second_figure = output_path / "gambar_4_2_eda_kategorikal_aset.png"

    plot_eda_target_numeric(features, target, first_figure)
    plot_eda_categorical_assets(features, target, second_figure)

    print("EDA selesai dibuat.")
    print(f"Gambar 4.1: {first_figure}")
    print(f"Gambar 4.2: {second_figure}")
    print(
        "Catatan: EDA bersifat deskriptif dan tidak mengubah label, memilih "
        "hyperparameter, menentukan cut-off, atau mengevaluasi model."
    )


if __name__ == "__main__":
    main()
