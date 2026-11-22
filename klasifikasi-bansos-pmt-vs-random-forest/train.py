from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import (
    CV_FOLDS,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    RANDOM_STATE,
    REFERENCE_DATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data import build_features, encode_target, load_dataset
from src.environment import collect_environment, validate_environment
from src.integrity import dataset_fingerprint, sha256_file
from src.evaluation import EvaluationResult, evaluate_predictions, exact_mcnemar
from src.modeling import (
    TrainedModel,
    create_pmt_search,
    create_random_forest_search,
    predict_with_threshold,
    train_optimized_model,
)
from src.visualization import (
    plot_confusion_matrix,
    plot_data_split,
    plot_eda_categorical_assets,
    plot_eda_target_numeric,
    plot_error_comparison,
    plot_mcnemar,
    plot_metric_comparison,
    plot_pmt_coefficients,
    plot_precision_recall_curves,
    plot_rf_importance,
    plot_roc_curves,
    plot_threshold_curve,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pelatihan final PMT Regresi Logistik dan Random Forest."
    )
    parser.add_argument(
        "--input",
        default="data/Data-Dummy.xlsx",
        help="Lokasi dataset Excel.",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Folder penyimpanan model, tabel, laporan, dan gambar BAB IV.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Jumlah proses paralel. Nilai 1 paling stabil di Windows.",
    )
    return parser.parse_args()


def to_serializable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def evaluation_table(pmt: EvaluationResult, rf: EvaluationResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, result in [("PMT Regresi Logistik", pmt), ("Random Forest", rf)]:
        rows.append(
            {
                "Metode": name,
                "Accuracy (%)": result.accuracy * 100,
                "Balanced Accuracy (%)": result.balanced_accuracy * 100,
                "Precision (%)": result.precision * 100,
                "Recall (%)": result.recall * 100,
                "F1-Score (%)": result.f1 * 100,
                "Inclusion Error (%)": result.inclusion_error * 100,
                "Exclusion Error (%)": result.exclusion_error * 100,
                "ROC-AUC (%)": result.roc_auc * 100,
                "Average Precision (%)": result.average_precision * 100,
                "TN": result.tn,
                "FP": result.fp,
                "FN": result.fn,
                "TP": result.tp,
            }
        )
    return pd.DataFrame(rows).round(4)


def save_tabular_outputs(
    output_path: Path,
    raw_data: pd.DataFrame,
    y: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    pmt_model: TrainedModel,
    rf_model: TrainedModel,
    pmt_result: EvaluationResult,
    rf_result: EvaluationResult,
    pmt_prediction: np.ndarray,
    rf_prediction: np.ndarray,
    pmt_probability: np.ndarray,
    rf_probability: np.ndarray,
    mcnemar_result: dict[str, int | float],
) -> None:
    table_path = output_path / "tabel"
    report_path = output_path / "laporan"
    table_path.mkdir(parents=True, exist_ok=True)
    report_path.mkdir(parents=True, exist_ok=True)

    metrics = evaluation_table(pmt_result, rf_result)
    metrics.to_csv(table_path / "01_metrik_evaluasi.csv", index=False)

    dataset_summary = pd.DataFrame(
        [
            {
                "Jumlah Observasi": len(y),
                "Jumlah Kolom Awal": raw_data.shape[1],
                "Jumlah Prediktor Akhir": 27,
                "Kelas Layak": int((y == 1).sum()),
                "Kelas Tidak Layak": int((y == 0).sum()),
                "Data Latih": len(y_train),
                "Data Uji": len(y_test),
            }
        ]
    )
    dataset_summary.to_csv(table_path / "00_ringkasan_dataset.csv", index=False)

    pmt_cm = pd.DataFrame(
        [[pmt_result.tn, pmt_result.fp], [pmt_result.fn, pmt_result.tp]],
        index=["Aktual Tidak Layak", "Aktual Layak"],
        columns=["Prediksi Tidak Layak", "Prediksi Layak"],
    )
    rf_cm = pd.DataFrame(
        [[rf_result.tn, rf_result.fp], [rf_result.fn, rf_result.tp]],
        index=["Aktual Tidak Layak", "Aktual Layak"],
        columns=["Prediksi Tidak Layak", "Prediksi Layak"],
    )
    pmt_cm.to_csv(table_path / "02_confusion_matrix_pmt.csv")
    rf_cm.to_csv(table_path / "03_confusion_matrix_random_forest.csv")

    pmt_model.threshold_curve.to_csv(table_path / "04_kurva_cutoff_pmt.csv", index=False)
    rf_model.threshold_curve.to_csv(table_path / "05_kurva_cutoff_random_forest.csv", index=False)

    pmt_preprocessor = pmt_model.pipeline.named_steps["preprocess"]
    pmt_feature_names = pmt_preprocessor.get_feature_names_out()
    pmt_coefficients = pmt_model.pipeline.named_steps["model"].coef_[0]
    coefficient_table = (
        pd.DataFrame(
            {
                "Fitur": pmt_feature_names,
                "Koefisien": pmt_coefficients,
                "Nilai Absolut": np.abs(pmt_coefficients),
            }
        )
        .sort_values("Nilai Absolut", ascending=False)
        .reset_index(drop=True)
    )
    coefficient_table.insert(0, "Peringkat", np.arange(1, len(coefficient_table) + 1))
    coefficient_table.to_csv(table_path / "06_koefisien_pmt.csv", index=False)

    rf_preprocessor = rf_model.pipeline.named_steps["preprocess"]
    rf_feature_names = rf_preprocessor.get_feature_names_out()
    rf_importance = rf_model.pipeline.named_steps["model"].feature_importances_
    importance_table = (
        pd.DataFrame({"Fitur": rf_feature_names, "Importance": rf_importance})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_table.insert(0, "Peringkat", np.arange(1, len(importance_table) + 1))
    importance_table.to_csv(
        table_path / "07_feature_importance_random_forest.csv", index=False
    )

    test_predictions = pd.DataFrame(
        {
            "No Data Uji": np.arange(1, len(y_test) + 1),
            "Indeks Sumber": y_test.index.to_numpy() + 1,
            "Status Aktual": np.where(np.asarray(y_test) == 1, POSITIVE_LABEL, NEGATIVE_LABEL),
            "Probabilitas PMT": pmt_probability,
            "Cut-Off PMT": pmt_model.threshold,
            "Prediksi PMT": np.where(pmt_prediction == 1, POSITIVE_LABEL, NEGATIVE_LABEL),
            "Probabilitas Random Forest": rf_probability,
            "Cut-Off Random Forest": rf_model.threshold,
            "Prediksi Random Forest": np.where(
                rf_prediction == 1, POSITIVE_LABEL, NEGATIVE_LABEL
            ),
        }
    )
    test_predictions["Kesesuaian PMT"] = np.where(
        test_predictions["Prediksi PMT"] == test_predictions["Status Aktual"],
        "BENAR",
        "SALAH",
    )
    test_predictions["Kesesuaian Random Forest"] = np.where(
        test_predictions["Prediksi Random Forest"] == test_predictions["Status Aktual"],
        "BENAR",
        "SALAH",
    )
    test_predictions["Kesesuaian Kedua Model"] = np.where(
        test_predictions["Prediksi PMT"] == test_predictions["Prediksi Random Forest"],
        "SAMA",
        "BERBEDA",
    )
    test_predictions.to_csv(table_path / "08_prediksi_data_uji.csv", index=False)

    mcnemar_table = pd.DataFrame([mcnemar_result])
    mcnemar_table.to_csv(table_path / "09_exact_mcnemar.csv", index=False)

    hyperparameters = pd.DataFrame(
        [
            {
                "Metode": "PMT Regresi Logistik",
                "Parameter Terbaik": json.dumps(pmt_model.best_params, ensure_ascii=False),
                "Rata-rata F1 Validasi": pmt_model.cv_f1,
                "Cut-Off": pmt_model.threshold,
            },
            {
                "Metode": "Random Forest",
                "Parameter Terbaik": json.dumps(rf_model.best_params, ensure_ascii=False),
                "Rata-rata F1 Validasi": rf_model.cv_f1,
                "Cut-Off": rf_model.threshold,
            },
        ]
    )
    hyperparameters.to_csv(table_path / "10_hasil_optimasi.csv", index=False)

    workbook_path = report_path / "hasil_pemodelan_lengkap.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        dataset_summary.to_excel(writer, sheet_name="Ringkasan Dataset", index=False)
        hyperparameters.to_excel(writer, sheet_name="Hasil Optimasi", index=False)
        metrics.to_excel(writer, sheet_name="Metrik Evaluasi", index=False)
        pmt_cm.to_excel(writer, sheet_name="CM PMT")
        rf_cm.to_excel(writer, sheet_name="CM Random Forest")
        mcnemar_table.to_excel(writer, sheet_name="Exact McNemar", index=False)
        pmt_model.threshold_curve.to_excel(writer, sheet_name="Cut-Off PMT", index=False)
        rf_model.threshold_curve.to_excel(writer, sheet_name="Cut-Off RF", index=False)
        coefficient_table.to_excel(writer, sheet_name="Koefisien PMT", index=False)
        importance_table.to_excel(writer, sheet_name="Feature Importance", index=False)
        test_predictions.to_excel(writer, sheet_name="Prediksi Data Uji", index=False)


def main() -> None:
    validate_environment(include_gui=False)
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
    figure_path = output_path / "gambar_bab4"
    model_path = output_path / "model"
    figure_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    print("[1/8] Membaca dan memvalidasi dataset final...")
    raw_data = load_dataset(input_path, require_full_schema=True)
    X = build_features(raw_data)
    y = encode_target(raw_data[TARGET_COLUMN])
    research_fingerprint = dataset_fingerprint(X, y)

    print("[2/8] Membuat EDA sebelum pembagian data dan preprocessing pipeline...")
    plot_eda_target_numeric(
        X,
        y,
        figure_path / "gambar_4_1_eda_target_numerik.png",
    )
    plot_eda_categorical_assets(
        X,
        y,
        figure_path / "gambar_4_2_eda_kategorikal_aset.png",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print("[3/8] Melatih PMT berbasis regresi logistik...")
    pmt_search = create_pmt_search(cross_validation, arguments.n_jobs)
    pmt_model = train_optimized_model(
        pmt_search,
        X_train,
        y_train,
        cross_validation,
        arguments.n_jobs,
    )

    print("[4/8] Melatih Random Forest...")
    rf_search = create_random_forest_search(
        cross_validation,
        arguments.n_jobs,
    )
    rf_model = train_optimized_model(
        rf_search,
        X_train,
        y_train,
        cross_validation,
        arguments.n_jobs,
    )

    print("[5/8] Mengevaluasi kedua model pada data uji...")
    pmt_prediction, pmt_probability = predict_with_threshold(
        pmt_model.pipeline,
        X_test,
        pmt_model.threshold,
    )
    rf_prediction, rf_probability = predict_with_threshold(
        rf_model.pipeline,
        X_test,
        rf_model.threshold,
    )

    pmt_result = evaluate_predictions(y_test, pmt_prediction, pmt_probability)
    rf_result = evaluate_predictions(y_test, rf_prediction, rf_probability)
    mcnemar_result = exact_mcnemar(y_test, pmt_prediction, rf_prediction)

    print("[6/8] Membuat gambar hasil pemodelan BAB IV...")
    plot_data_split(
        y_train,
        y_test,
        figure_path / "gambar_4_3_pembagian_data_latih_uji.png",
    )
    plot_threshold_curve(
        pmt_model.threshold_curve,
        pmt_model.threshold,
        "PMT Berbasis Regresi Logistik",
        figure_path / "gambar_4_5_pemilihan_cutoff_pmt.png",
    )
    plot_confusion_matrix(
        pmt_result,
        "PMT Berbasis Regresi Logistik",
        figure_path / "gambar_4_6_confusion_matrix_pmt.png",
    )
    plot_threshold_curve(
        rf_model.threshold_curve,
        rf_model.threshold,
        "Random Forest",
        figure_path / "gambar_4_8_pemilihan_cutoff_random_forest.png",
    )
    plot_confusion_matrix(
        rf_result,
        "Random Forest",
        figure_path / "gambar_4_9_confusion_matrix_random_forest.png",
    )
    plot_metric_comparison(
        pmt_result,
        rf_result,
        figure_path / "gambar_4_10_perbandingan_metrik_evaluasi.png",
    )
    plot_error_comparison(
        pmt_result,
        rf_result,
        figure_path / "gambar_4_11_perbandingan_kesalahan_sasaran.png",
    )
    plot_roc_curves(
        y_test,
        pmt_probability,
        rf_probability,
        pmt_result.roc_auc,
        rf_result.roc_auc,
        figure_path / "gambar_4_12_kurva_roc.png",
    )
    plot_precision_recall_curves(
        y_test,
        pmt_probability,
        rf_probability,
        pmt_result.average_precision,
        rf_result.average_precision,
        figure_path / "gambar_4_13_kurva_precision_recall.png",
    )
    plot_mcnemar(
        mcnemar_result,
        figure_path / "gambar_4_14_exact_mcnemar.png",
    )

    pmt_preprocessor = pmt_model.pipeline.named_steps["preprocess"]
    pmt_feature_names = pmt_preprocessor.get_feature_names_out()
    pmt_coefficients = pmt_model.pipeline.named_steps["model"].coef_[0]
    plot_pmt_coefficients(
        pmt_feature_names,
        pmt_coefficients,
        figure_path / "gambar_4_15_koefisien_pmt.png",
    )

    rf_preprocessor = rf_model.pipeline.named_steps["preprocess"]
    rf_feature_names = rf_preprocessor.get_feature_names_out()
    rf_importance = rf_model.pipeline.named_steps["model"].feature_importances_
    plot_rf_importance(
        rf_feature_names,
        rf_importance,
        figure_path / "gambar_4_16_feature_importance_random_forest.png",
    )

    print("[7/8] Menyimpan tabel dan laporan Excel...")
    save_tabular_outputs(
        output_path=output_path,
        raw_data=raw_data,
        y=y,
        y_train=y_train,
        y_test=y_test,
        pmt_model=pmt_model,
        rf_model=rf_model,
        pmt_result=pmt_result,
        rf_result=rf_result,
        pmt_prediction=pmt_prediction,
        rf_prediction=rf_prediction,
        pmt_probability=pmt_probability,
        rf_probability=rf_probability,
        mcnemar_result=mcnemar_result,
    )

    print("[8/8] Menyimpan model dan metadata...")
    pmt_model_file = model_path / "model_pmt.joblib"
    rf_model_file = model_path / "model_random_forest.joblib"
    joblib.dump(pmt_model.pipeline, pmt_model_file)
    joblib.dump(rf_model.pipeline, rf_model_file)

    artifact_hashes = {
        pmt_model_file.name: sha256_file(pmt_model_file),
        rf_model_file.name: sha256_file(rf_model_file),
    }

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "file": input_path.name,
            "rows": len(y),
            "columns": raw_data.shape[1],
            "positive": int((y == 1).sum()),
            "negative": int((y == 0).sum()),
            "train_rows": len(y_train),
            "test_rows": len(y_test),
            "predictors": X.shape[1],
            "transformed_features_pmt": len(pmt_feature_names),
            "transformed_features_random_forest": len(rf_feature_names),
            "feature_target_fingerprint_sha256": research_fingerprint,
        },
        "eda": {
            "stage": (
                "Setelah validasi, pembersihan, seleksi variabel, dan transformasi "
                "TANGGAL_LAHIR menjadi USIA; sebelum one-hot encoding, "
                "standardisasi, dan pembagian data."
            ),
            "figures": [
                "gambar_4_1_eda_target_numerik.png",
                "gambar_4_2_eda_kategorikal_aset.png",
            ],
            "variables": [
                "KETERANGAN",
                "USIA",
                "JUMLAH-ART",
                "LANSIA",
                "PENDIDIKAN",
                "PEKERJAAN",
                "KEPEMILIKAN_ASET",
                "DAYA_LISTRIK_TERPASANG",
            ],
            "purpose": "Analisis deskriptif; tidak digunakan untuk mengubah label atau memilih parameter model.",
        },
        "research": {
            "target_column": TARGET_COLUMN,
            "positive_label": POSITIVE_LABEL,
            "negative_label": NEGATIVE_LABEL,
            "reference_date": REFERENCE_DATE,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "evaluation": {
                "primary_metrics": [
                    "confusion_matrix",
                    "accuracy",
                    "balanced_accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "inclusion_error",
                    "exclusion_error",
                ],
                "supporting_metrics": ["roc_auc", "average_precision"],
                "statistical_test": "exact_mcnemar",
            },
        },
        "pmt": {
            "best_params": pmt_model.best_params,
            "cv_f1": pmt_model.cv_f1,
            "threshold": pmt_model.threshold,
            "test_metrics": pmt_result.to_dict(),
        },
        "random_forest": {
            "best_params": rf_model.best_params,
            "cv_f1": rf_model.cv_f1,
            "threshold": rf_model.threshold,
            "test_metrics": rf_result.to_dict(),
        },
        "mcnemar": mcnemar_result,
        "artifacts": {
            "sha256": artifact_hashes,
        },
        "environment": collect_environment(include_gui=False),
    }
    serializable_metadata = to_serializable(metadata)
    for file_name in ("metadata.json", "model_metadata.json"):
        with (model_path / file_name).open("w", encoding="utf-8") as file:
            json.dump(serializable_metadata, file, ensure_ascii=False, indent=2)

    print("\nPelatihan selesai.")
    print(f"Gambar BAB IV : {figure_path.resolve()}")
    print(f"Tabel          : {(output_path / 'tabel').resolve()}")
    print(
        "Laporan Excel  : "
        f"{(output_path / 'laporan' / 'hasil_pemodelan_lengkap.xlsx').resolve()}"
    )
    print(f"Model          : {model_path.resolve()}")
    print("\nHasil utama:")
    print(
        f"PMT  - Accuracy {pmt_result.accuracy * 100:.2f}% | "
        f"F1 {pmt_result.f1 * 100:.2f}% | Cut-off {pmt_model.threshold:.2f}"
    )
    print(
        f"RF   - Accuracy {rf_result.accuracy * 100:.2f}% | "
        f"F1 {rf_result.f1 * 100:.2f}% | Cut-off {rf_model.threshold:.2f}"
    )
    print(f"Exact McNemar p-value: {float(mcnemar_result['p_value']):.6f}")


if __name__ == "__main__":
    main()
