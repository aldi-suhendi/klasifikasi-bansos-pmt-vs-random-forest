from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, PercentFormatter
from sklearn.metrics import precision_recall_curve, roc_curve

from .config import CATEGORICAL_FEATURES, NEGATIVE_LABEL, POSITIVE_LABEL
from .evaluation import EvaluationResult


DPI = 300

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.weight": "normal",
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "legend.frameon": True,
    }
)


def _format_count_id(value: int | float) -> str:
    return f"{int(round(float(value))):,}".replace(",", ".")


def _format_decimal_id(value: float, decimals: int = 2) -> str:
    return f"{float(value):.{decimals}f}".replace(".", ",")


def _format_percent_id(
    value: float,
    decimals: int = 2,
    value_is_ratio: bool = True,
) -> str:
    percentage = float(value) * 100 if value_is_ratio else float(value)
    return f"{percentage:.{decimals}f}%".replace(".", ",")


def _display_class_label(label: str) -> str:
    return str(label).replace("_", " ").title()


def _format_feature_name(name: str) -> str:
    text = str(name)

    for prefix in ("numeric__", "categorical__", "num__", "cat__"):
        if text.startswith(prefix):
            text = text[len(prefix):]

    for column in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
        prefix = f"{column}_"
        if text.startswith(prefix):
            category = text[len(prefix):].replace("_", " ")
            label = column.replace("_", " ").replace("-", " ")
            return f"{label.title()}: {category.title()}"

    return text.replace("_", " ").replace("-", " ").title()


def _decimal_tick_id(value: float, _position: int | None = None) -> str:
    return _format_decimal_id(value, 1)


def _apply_normal_font(ax: plt.Axes) -> None:
    ax.title.set_fontweight("normal")
    ax.xaxis.label.set_fontweight("normal")
    ax.yaxis.label.set_fontweight("normal")

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("normal")

    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontweight("normal")


def _save(fig: plt.Figure, path: Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def _format_category_label(value: object) -> str:
    return str(value).replace("_", " ").title()


def _save_composite(fig: plt.Figure, path: Path, *, bottom: float = 0.07) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.02, bottom, 0.98, 0.95))
    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_eda_target_numeric(
    features: pd.DataFrame,
    y: pd.Series,
    path: Path,
) -> None:
    required = {"USIA", "JUMLAH-ART", "LANSIA"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(
            "Fitur EDA numerik belum lengkap: " + ", ".join(sorted(missing))
        )

    target = pd.Series(y, index=features.index).astype(int)
    labels = [
        _display_class_label(NEGATIVE_LABEL),
        _display_class_label(POSITIVE_LABEL),
    ]
    class_colors = ["#1f77b4", "#ff7f0e"]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.4))
    fig.suptitle(
        "Exploratory Data Analysis Variabel Target dan Numerik",
        fontsize=15,
        fontweight="normal",
        y=0.985,
    )

    ax = axes[0, 0]
    counts = [int((target == 0).sum()), int((target == 1).sum())]
    bars = ax.bar(labels, counts, color=class_colors)
    total = int(target.shape[0])
    ax.set_title("(a) Distribusi Kelas Target", fontsize=11)
    ax.set_xlabel("Status Kelayakan")
    ax.set_ylabel("Jumlah Data")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(counts) * 1.17)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.015,
            f"{_format_count_id(count)}\n({_format_percent_id(count / total)})",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )

    ax = axes[0, 1]
    age_values = [
        pd.to_numeric(features.loc[target == class_value, "USIA"], errors="coerce")
        .dropna()
        .to_numpy()
        for class_value in (0, 1)
    ]
    ax.boxplot(
        age_values,
        tick_labels=labels,
        showmeans=True,
        meanline=True,
        medianprops={"color": "#ff7f0e", "linewidth": 1.3},
        meanprops={"color": "#2ca02c", "linestyle": "--", "linewidth": 1.2},
        boxprops={"linewidth": 1.1},
        whiskerprops={"linewidth": 1.1},
        capprops={"linewidth": 1.1},
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 4.5,
        },
    )
    ax.set_title("(b) Distribusi Usia", fontsize=11)
    ax.set_xlabel("Status Hasil Verifikasi dan Validasi")
    ax.set_ylabel("Usia (tahun)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    ax = axes[1, 0]
    household_values = [
        pd.to_numeric(
            features.loc[target == class_value, "JUMLAH-ART"], errors="coerce"
        )
        .dropna()
        .to_numpy()
        for class_value in (0, 1)
    ]
    ax.boxplot(
        household_values,
        tick_labels=labels,
        showmeans=True,
        meanline=True,
        medianprops={"color": "#ff7f0e", "linewidth": 1.3},
        meanprops={"color": "#2ca02c", "linestyle": "--", "linewidth": 1.2},
        boxprops={"linewidth": 1.1},
        whiskerprops={"linewidth": 1.1},
        capprops={"linewidth": 1.1},
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 4.5,
        },
    )
    ax.set_title("(c) Distribusi Jumlah Anggota Rumah Tangga", fontsize=11)
    ax.set_xlabel("Status Hasil Verifikasi dan Validasi")
    ax.set_ylabel("Jumlah Anggota Rumah Tangga")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    ax = axes[1, 1]
    elderly = pd.to_numeric(features["LANSIA"], errors="coerce")
    categories = sorted(
        int(value) for value in elderly.dropna().unique().tolist()
    )
    table = pd.crosstab(elderly, target, normalize="columns").reindex(
        index=categories,
        columns=[0, 1],
        fill_value=0.0,
    )
    x = np.arange(len(categories))
    width = 0.36
    ax.bar(
        x - width / 2,
        table[0].to_numpy(dtype=float),
        width,
        label=labels[0],
        color=class_colors[0],
    )
    ax.bar(
        x + width / 2,
        table[1].to_numpy(dtype=float),
        width,
        label=labels[1],
        color=class_colors[1],
    )
    ax.set_title("(d) Distribusi Jumlah Lansia", fontsize=11)
    ax.set_xlabel("Jumlah Lansia dalam Rumah Tangga")
    ax.set_ylabel("Proporsi di dalam Kelas")
    ax.set_xticks(x, [str(value) for value in categories])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, max(1.0, float(table.to_numpy().max()) * 1.06))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")

    for axis in axes.flat:
        _apply_normal_font(axis)

    _save_composite(fig, path, bottom=0.025)


def _stacked_status_table(
    values: pd.Series,
    target: pd.Series,
    order: list[object],
) -> pd.DataFrame:
    table = pd.crosstab(values, target).reindex(columns=[0, 1], fill_value=0)
    existing = [value for value in order if value in table.index]
    remaining = [value for value in table.index if value not in existing]
    return table.reindex([*existing, *remaining], fill_value=0)


def _add_positive_percentage_labels(
    ax: plt.Axes,
    table: pd.DataFrame,
    *,
    decimals: int = 1,
) -> None:
    totals = table.sum(axis=1).to_numpy(dtype=float)
    positive = table[1].to_numpy(dtype=float)
    maximum = max(float(totals.max()) if len(totals) else 0.0, 1.0)
    for position, (total, pos) in enumerate(zip(totals, positive)):
        percentage = 0.0 if total == 0 else pos / total * 100
        text = f"Layak {percentage:.{decimals}f}%".replace(".", ",")
        ax.text(
            total + maximum * 0.018,
            position,
            text,
            va="center",
            ha="left",
            fontsize=8.4,
        )
    ax.set_xlim(0, maximum * 1.22)


def plot_eda_categorical_assets(
    features: pd.DataFrame,
    y: pd.Series,
    path: Path,
) -> None:
    required = {
        "PENDIDIKAN",
        "PEKERJAAN",
        "DAYA_LISTRIK_TERPASANG",
        "SMARTPHONE",
        "SEPEDA_MOTOR",
        "TV_LAYAR_DATAR",
        "LEMARI_ES",
        "KOMPUTER/LAPTOP",
        "SEPEDA",
        "AC",
        "MOBIL",
        "TERNAK_KECIL",
        "TERNAK_BESAR",
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(
            "Fitur EDA kategorikal/aset belum lengkap: "
            + ", ".join(sorted(missing))
        )

    target = pd.Series(y, index=features.index).astype(int)
    class_labels = [
        _display_class_label(NEGATIVE_LABEL),
        _display_class_label(POSITIVE_LABEL),
    ]
    class_colors = ["#1f77b4", "#ff7f0e"]

    fig, axes = plt.subplots(2, 2, figsize=(14.3, 9.6))
    fig.suptitle(
        "Exploratory Data Analysis Variabel Kategorikal dan Kepemilikan Aset",
        fontsize=15,
        fontweight="normal",
        y=0.985,
    )

    ax = axes[0, 0]
    education_order = [
        "DIPLOMA/SARJANA",
        "SMA/SEDERAJAT",
        "SMP/SEDERAJAT",
        "SD/SEDERAJAT",
        "TIDAK TAMAT SD/SEDERAJAT",
    ]
    education = _stacked_status_table(
        features["PENDIDIKAN"].astype("string"),
        target,
        education_order,
    )
    y_position = np.arange(len(education))
    ax.barh(
        y_position,
        education[0],
        color=class_colors[0],
        label=class_labels[0],
    )
    ax.barh(
        y_position,
        education[1],
        left=education[0],
        color=class_colors[1],
        label=class_labels[1],
    )
    education_labels = {
        "DIPLOMA/SARJANA": "Diploma/Sarjana",
        "SMA/SEDERAJAT": "SMA/Sederajat",
        "SMP/SEDERAJAT": "SMP/Sederajat",
        "SD/SEDERAJAT": "SD/Sederajat",
        "TIDAK TAMAT SD/SEDERAJAT": "Tidak Tamat SD/Sederajat",
    }
    ax.set_yticks(
        y_position,
        [education_labels.get(str(value), _format_category_label(value)) for value in education.index],
    )
    ax.invert_yaxis()
    ax.set_title("(a) Distribusi Pendidikan Berdasarkan Status", fontsize=11)
    ax.set_xlabel("Jumlah Data")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    _add_positive_percentage_labels(ax, education)

    ax = axes[0, 1]
    occupation_all = pd.crosstab(
        features["PEKERJAAN"].astype("string"), target
    ).reindex(columns=[0, 1], fill_value=0)
    occupation_order = (
        occupation_all.sum(axis=1)
        .sort_values(ascending=False)
        .head(6)
        .index.tolist()
    )
    occupation = occupation_all.reindex(occupation_order, fill_value=0)
    y_position = np.arange(len(occupation))
    ax.barh(
        y_position,
        occupation[0],
        color=class_colors[0],
        label=class_labels[0],
    )
    ax.barh(
        y_position,
        occupation[1],
        left=occupation[0],
        color=class_colors[1],
        label=class_labels[1],
    )
    ax.set_yticks(
        y_position,
        [_format_category_label(value) for value in occupation.index],
    )
    ax.invert_yaxis()
    ax.set_title("(b) 6 Pekerjaan Terbanyak Berdasarkan Status", fontsize=11)
    ax.set_xlabel("Jumlah Data")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    _add_positive_percentage_labels(ax, occupation)

    ax = axes[1, 0]
    asset_columns = [
        "SMARTPHONE",
        "SEPEDA_MOTOR",
        "TV_LAYAR_DATAR",
        "LEMARI_ES",
        "KOMPUTER/LAPTOP",
        "SEPEDA",
        "AC",
        "MOBIL",
        "TERNAK_KECIL",
        "TERNAK_BESAR",
    ]
    asset_labels = [
        "Smartphone",
        "Sepeda Motor",
        "TV Layar Datar",
        "Lemari Es",
        "Komputer/Laptop",
        "Sepeda",
        "AC",
        "Mobil",
        "Ternak Kecil",
        "Ternak Besar",
    ]
    percentages = np.array(
        [
            pd.to_numeric(features[column], errors="coerce").fillna(0).mean()
            for column in asset_columns
        ],
        dtype=float,
    )
    y_position = np.arange(len(asset_columns))
    bars = ax.barh(y_position, percentages, color=class_colors[0])
    ax.set_yticks(y_position, asset_labels)
    ax.invert_yaxis()
    ax.set_title("(c) Persentase Kepemilikan Aset", fontsize=11)
    ax.set_xlabel("Persentase Observasi")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, 1.14)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, percentage in zip(bars, percentages):
        ax.text(
            percentage + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{percentage * 100:.1f}%".replace(".", ","),
            va="center",
            ha="left",
            fontsize=8.4,
        )

    ax = axes[1, 1]
    electricity_values = pd.to_numeric(
        features["DAYA_LISTRIK_TERPASANG"], errors="coerce"
    )
    electricity_order = [3500, 2200, 1300, 900, 450]
    electricity = _stacked_status_table(
        electricity_values,
        target,
        electricity_order,
    )
    y_position = np.arange(len(electricity))
    ax.barh(
        y_position,
        electricity[0],
        color=class_colors[0],
        label=class_labels[0],
    )
    ax.barh(
        y_position,
        electricity[1],
        left=electricity[0],
        color=class_colors[1],
        label=class_labels[1],
    )
    electricity_labels = [
        f"{_format_count_id(float(value))} VA" for value in electricity.index
    ]
    ax.set_yticks(y_position, electricity_labels)
    ax.invert_yaxis()
    ax.set_title("(d) Distribusi Daya Listrik Berdasarkan Status", fontsize=11)
    ax.set_xlabel("Jumlah Data")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    _add_positive_percentage_labels(ax, electricity)

    legend_handles = [
        Patch(facecolor=class_colors[0], label=class_labels[0]),
        Patch(facecolor=class_colors[1], label=class_labels[1]),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 0.015),
    )

    for axis in axes.flat:
        _apply_normal_font(axis)

    _save_composite(fig, path, bottom=0.065)


def plot_class_distribution(y: pd.Series, path: Path) -> None:
    target = pd.Series(y).astype(int)
    counts = [int((target == 0).sum()), int((target == 1).sum())]
    labels = [
        _display_class_label(NEGATIVE_LABEL),
        _display_class_label(POSITIVE_LABEL),
    ]

    total = sum(counts)
    if total == 0:
        raise ValueError(
            "Distribusi kelas tidak dapat dibuat karena data target kosong."
        )

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    bars = ax.bar(labels, counts)

    ax.set_title("Distribusi Kelas Target", fontsize=14, pad=12)
    ax.set_xlabel("Status Kelayakan", fontsize=11)
    ax.set_ylabel("Jumlah Data", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(counts) * 1.17)

    for bar, count in zip(bars, counts):
        percentage = count / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.015,
            (
                f"{_format_count_id(count)}\n"
                f"({_format_percent_id(percentage)})"
            ),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="normal",
        )

    _apply_normal_font(ax)
    _save(fig, path)


def plot_data_split(
    y_train: pd.Series,
    y_test: pd.Series,
    path: Path,
) -> None:
    train = pd.Series(y_train).astype(int)
    test = pd.Series(y_test).astype(int)

    labels = ["Data Latih", "Data Uji"]
    negative = [int((train == 0).sum()), int((test == 0).sum())]
    positive = [int((train == 1).sum()), int((test == 1).sum())]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    x = np.arange(len(labels))

    ax.bar(
        x,
        negative,
        label=_display_class_label(NEGATIVE_LABEL),
    )
    ax.bar(
        x,
        positive,
        bottom=negative,
        label=_display_class_label(POSITIVE_LABEL),
    )

    ax.set_title("Pembagian Data Latih dan Data Uji", fontsize=14, pad=12)
    ax.set_xlabel("Bagian Dataset", fontsize=11)
    ax.set_ylabel("Jumlah Data", fontsize=11)
    ax.set_xticks(x, labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    for index, (neg, pos) in enumerate(zip(negative, positive)):
        total = neg + pos
        ax.text(
            index,
            neg / 2,
            _format_count_id(neg),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="normal",
        )
        ax.text(
            index,
            neg + pos / 2,
            _format_count_id(pos),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="normal",
        )
        ax.text(
            index,
            total + max(total * 0.012, 3),
            _format_count_id(total),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="normal",
        )

    _apply_normal_font(ax)
    _save(fig, path)


def plot_threshold_curve(
    curve: pd.DataFrame,
    selected_threshold: float,
    model_name: str,
    path: Path,
) -> None:
    required_columns = {"threshold", "precision", "recall", "f1"}
    missing = required_columns.difference(curve.columns)
    if missing:
        raise ValueError(
            "Kolom kurva cut-off tidak lengkap: "
            + ", ".join(sorted(missing))
        )

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(curve["threshold"], curve["precision"], linewidth=1.8, label="Precision")
    ax.plot(curve["threshold"], curve["recall"], linewidth=1.8, label="Recall")
    ax.plot(curve["threshold"], curve["f1"], linewidth=1.8, label="F1-score")
    ax.axvline(
        selected_threshold,
        linestyle="--",
        linewidth=1.5,
        label=(
            "Cut-off terpilih = "
            f"{_format_decimal_id(selected_threshold, 2)}"
        ),
    )

    ax.set_title(f"Pemilihan Cut-off {model_name}", fontsize=14, pad=12)
    ax.set_xlabel("Cut-off", fontsize=11)
    ax.set_ylabel("Nilai Metrik (%)", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(FuncFormatter(_decimal_tick_id))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="best")

    _apply_normal_font(ax)
    _save(fig, path)


def plot_confusion_matrix(
    result: EvaluationResult,
    model_name: str,
    path: Path,
) -> None:
    matrix = np.array(
        [[result.tn, result.fp], [result.fn, result.tp]],
        dtype=int,
    )

    labels = [
        _display_class_label(NEGATIVE_LABEL),
        _display_class_label(POSITIVE_LABEL),
    ]

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    image = ax.imshow(matrix, cmap="Blues")

    ax.set_title(f"Confusion Matrix {model_name}", fontsize=14, pad=12)
    ax.set_xlabel("Prediksi", fontsize=11)
    ax.set_ylabel("Aktual", fontsize=11)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)

    threshold = matrix.max() / 2
    cell_names = np.array([["TN", "FP"], ["FN", "TP"]])

    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                (
                    f"{cell_names[row, column]}\n"
                    f"{_format_count_id(matrix[row, column])}"
                ),
                ha="center",
                va="center",
                fontsize=13,
                fontweight="normal",
                color=(
                    "white"
                    if matrix[row, column] > threshold
                    else "black"
                ),
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _apply_normal_font(ax)
    _save(fig, path)


def plot_metric_comparison(
    pmt: EvaluationResult,
    rf: EvaluationResult,
    path: Path,
) -> None:
    metric_keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Balanced\nAccuracy", "Precision", "Recall", "F1-score"]

    pmt_values = [getattr(pmt, key) * 100 for key in metric_keys]
    rf_values = [getattr(rf, key) * 100 for key in metric_keys]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    pmt_bars = ax.bar(
        x - width / 2,
        pmt_values,
        width,
        label="PMT Regresi Logistik",
    )
    rf_bars = ax.bar(
        x + width / 2,
        rf_values,
        width,
        label="Random Forest",
    )

    ax.set_title("Perbandingan Metrik Evaluasi", fontsize=14, pad=12)
    ax.set_xlabel("Metrik", fontsize=11)
    ax.set_ylabel("Nilai (%)", fontsize=11)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    for bars in (pmt_bars, rf_bars):
        for bar in bars:
            value = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.8,
                _format_percent_id(value, decimals=2, value_is_ratio=False),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="normal",
            )

    _apply_normal_font(ax)
    _save(fig, path)


def plot_error_comparison(
    pmt: EvaluationResult,
    rf: EvaluationResult,
    path: Path,
) -> None:
    labels = ["Inclusion Error", "Exclusion Error"]
    pmt_values = [pmt.inclusion_error * 100, pmt.exclusion_error * 100]
    rf_values = [rf.inclusion_error * 100, rf.exclusion_error * 100]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    pmt_bars = ax.bar(
        x - width / 2,
        pmt_values,
        width,
        label="PMT Regresi Logistik",
    )
    rf_bars = ax.bar(
        x + width / 2,
        rf_values,
        width,
        label="Random Forest",
    )

    ax.set_title("Perbandingan Kesalahan Penargetan", fontsize=14, pad=12)
    ax.set_xlabel("Jenis Kesalahan", fontsize=11)
    ax.set_ylabel("Nilai (%)", fontsize=11)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(pmt_values + rf_values) * 1.25)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    for bars in (pmt_bars, rf_bars):
        for bar in bars:
            value = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.3,
                _format_percent_id(value, decimals=2, value_is_ratio=False),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="normal",
            )

    _apply_normal_font(ax)
    _save(fig, path)


def plot_roc_curves(
    y_true: pd.Series,
    pmt_probability: np.ndarray,
    rf_probability: np.ndarray,
    pmt_auc: float,
    rf_auc: float,
    path: Path,
) -> None:
    actual = np.asarray(y_true, dtype=int)
    pmt_prob = np.asarray(pmt_probability, dtype=float)
    rf_prob = np.asarray(rf_probability, dtype=float)

    pmt_fpr, pmt_tpr, _ = roc_curve(actual, pmt_prob)
    rf_fpr, rf_tpr, _ = roc_curve(actual, rf_prob)

    fig, ax = plt.subplots(figsize=(7.6, 6.2))

    ax.plot(
        pmt_fpr,
        pmt_tpr,
        linewidth=2,
        label=f"PMT (ROC-AUC = {_format_percent_id(pmt_auc)})",
    )
    ax.plot(
        rf_fpr,
        rf_tpr,
        linewidth=2,
        label=f"Random Forest (ROC-AUC = {_format_percent_id(rf_auc)})",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.4,
        label="Klasifikasi acak (ROC-AUC = 50,00%)",
    )

    ax.set_title("Kurva ROC PMT dan Random Forest", fontsize=14, pad=12)
    ax.set_xlabel("False Positive Rate (%)", fontsize=11)
    ax.set_ylabel("True Positive Rate (%)", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")

    _apply_normal_font(ax)
    _save(fig, path)


def plot_precision_recall_curves(
    y_true: pd.Series,
    pmt_probability: np.ndarray,
    rf_probability: np.ndarray,
    pmt_average_precision: float,
    rf_average_precision: float,
    path: Path,
) -> None:
    actual = np.asarray(y_true, dtype=int)
    pmt_prob = np.asarray(pmt_probability, dtype=float)
    rf_prob = np.asarray(rf_probability, dtype=float)

    pmt_precision, pmt_recall, _ = precision_recall_curve(actual, pmt_prob)
    rf_precision, rf_recall, _ = precision_recall_curve(actual, rf_prob)
    prevalence = float(np.mean(actual))

    fig, ax = plt.subplots(figsize=(7.6, 6.2))

    ax.plot(
        pmt_recall,
        pmt_precision,
        linewidth=2,
        label=(
            "PMT (Average Precision = "
            f"{_format_percent_id(pmt_average_precision)})"
        ),
    )
    ax.plot(
        rf_recall,
        rf_precision,
        linewidth=2,
        label=(
            "Random Forest (Average Precision = "
            f"{_format_percent_id(rf_average_precision)})"
        ),
    )
    ax.axhline(
        prevalence,
        linestyle="--",
        linewidth=1.4,
        label=(
            "Proporsi kelas Layak = "
            f"{_format_percent_id(prevalence)}"
        ),
    )

    ax.set_title("Kurva Precision-Recall PMT dan Random Forest", fontsize=14, pad=12)
    ax.set_xlabel("Recall (%)", fontsize=11)
    ax.set_ylabel("Precision (%)", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left")

    _apply_normal_font(ax)
    _save(fig, path)


def plot_pmt_coefficients(
    feature_names: np.ndarray,
    coefficients: np.ndarray,
    path: Path,
    top_n: int = 15,
) -> None:
    frame = pd.DataFrame(
        {"feature": feature_names, "coefficient": coefficients}
    ).assign(
        feature=lambda item: item["feature"].map(_format_feature_name),
        absolute=lambda item: item["coefficient"].abs(),
    )
    frame = frame.nlargest(top_n, "absolute").sort_values("coefficient")

    fig, ax = plt.subplots(figsize=(10, 7.2))
    bars = ax.barh(frame["feature"], frame["coefficient"])

    ax.axvline(0, linewidth=1)
    ax.set_title(
        f"{top_n} Koefisien Dominan PMT Regresi Logistik",
        fontsize=14,
        pad=12,
    )
    ax.set_xlabel("Nilai Koefisien", fontsize=11)
    ax.set_ylabel("Fitur", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, position: _format_decimal_id(value, 1))
    )

    maximum = float(frame["coefficient"].abs().max())
    if maximum == 0:
        maximum = 1.0

    ax.set_xlim(
        min(float(frame["coefficient"].min()) - maximum * 0.15, -0.1),
        max(float(frame["coefficient"].max()) + maximum * 0.15, 0.1),
    )

    for bar, value in zip(bars, frame["coefficient"]):
        numeric_value = float(value)
        alignment = "left" if numeric_value >= 0 else "right"
        offset = maximum * 0.018 if numeric_value >= 0 else -maximum * 0.018
        ax.text(
            numeric_value + offset,
            bar.get_y() + bar.get_height() / 2,
            _format_decimal_id(numeric_value, 3),
            va="center",
            ha=alignment,
            fontsize=8,
            fontweight="normal",
        )

    _apply_normal_font(ax)
    _save(fig, path)


def plot_rf_importance(
    feature_names: np.ndarray,
    importance: np.ndarray,
    path: Path,
    top_n: int = 15,
) -> None:
    frame = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .assign(feature=lambda item: item["feature"].map(_format_feature_name))
        .nlargest(top_n, "importance")
        .sort_values("importance")
    )

    fig, ax = plt.subplots(figsize=(10, 7.2))
    bars = ax.barh(frame["feature"], frame["importance"])

    ax.set_title(
        f"{top_n} Tingkat Kepentingan Fitur Random Forest",
        fontsize=14,
        pad=12,
    )
    ax.set_xlabel("Nilai Feature Importance", fontsize=11)
    ax.set_ylabel("Fitur", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, position: _format_decimal_id(value, 2))
    )

    max_importance = float(frame["importance"].max())
    ax.set_xlim(0, max_importance * 1.18 if max_importance > 0 else 1)

    for bar, value in zip(bars, frame["importance"]):
        numeric_value = float(value)
        ax.text(
            numeric_value + max_importance * 0.01,
            bar.get_y() + bar.get_height() / 2,
            _format_decimal_id(numeric_value, 3),
            va="center",
            ha="left",
            fontsize=8,
            fontweight="normal",
        )

    _apply_normal_font(ax)
    _save(fig, path)


def plot_mcnemar(
    mcnemar: dict[str, int | float],
    path: Path,
) -> None:
    labels = [
        "Keduanya\nBenar",
        "PMT Benar\nRF Salah",
        "PMT Salah\nRF Benar",
        "Keduanya\nSalah",
    ]
    values = [
        int(mcnemar["both_correct"]),
        int(mcnemar["pmt_correct_rf_wrong"]),
        int(mcnemar["pmt_wrong_rf_correct"]),
        int(mcnemar["both_wrong"]),
    ]
    p_value = float(mcnemar["p_value"])

    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    bars = ax.bar(labels, values)

    ax.set_title(
        "Exact McNemar Test "
        f"(p-value = {_format_decimal_id(p_value, 4)})",
        fontsize=14,
        pad=12,
    )
    ax.set_xlabel("Kombinasi Ketepatan Prediksi", fontsize=11)
    ax.set_ylabel("Jumlah Data Uji", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(values) * 1.12)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.012,
            _format_count_id(value),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="normal",
        )

    _apply_normal_font(ax)
    _save(fig, path)


def plot_result_summary(
    dataset_size: int,
    train_size: int,
    test_size: int,
    pmt: EvaluationResult,
    rf: EvaluationResult,
    pmt_threshold: float,
    rf_threshold: float,
    pmt_params: dict[str, Any],
    rf_params: dict[str, Any],
    p_value: float,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 8.2))
    ax.axis("off")

    ax.text(
        0.5,
        0.96,
        "Ringkasan Hasil Komparasi Model",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="normal",
        transform=ax.transAxes,
    )

    ax.text(
        0.5,
        0.91,
        (
            f"Dataset: {_format_count_id(dataset_size)} data | "
            f"Latih: {_format_count_id(train_size)} | "
            f"Uji: {_format_count_id(test_size)}"
        ),
        ha="center",
        va="top",
        fontsize=11,
        fontweight="normal",
        transform=ax.transAxes,
    )

    box_style = {
        "boxstyle": "round,pad=0.8",
        "facecolor": "white",
        "edgecolor": "0.65",
    }

    pmt_parameter_text = "\n".join(
        f"{key.replace('model__', '')} = {value}"
        for key, value in pmt_params.items()
    )
    rf_parameter_text = "\n".join(
        f"{key.replace('model__', '')} = {value}"
        for key, value in rf_params.items()
    )

    pmt_text = (
        "PMT REGRESI LOGISTIK\n\n"
        f"Cut-off                 : {_format_decimal_id(pmt_threshold, 2)}\n"
        f"Accuracy                : {_format_percent_id(pmt.accuracy)}\n"
        f"Balanced accuracy       : {_format_percent_id(pmt.balanced_accuracy)}\n"
        f"Precision               : {_format_percent_id(pmt.precision)}\n"
        f"Recall                  : {_format_percent_id(pmt.recall)}\n"
        f"F1-score                : {_format_percent_id(pmt.f1)}\n"
        f"Inclusion error         : {_format_percent_id(pmt.inclusion_error)}\n"
        f"Exclusion error         : {_format_percent_id(pmt.exclusion_error)}\n"
        f"ROC-AUC                 : {_format_percent_id(pmt.roc_auc)}\n"
        f"Average Precision       : {_format_percent_id(pmt.average_precision)}\n\n"
        f"Parameter terbaik\n{pmt_parameter_text}"
    )

    rf_text = (
        "RANDOM FOREST\n\n"
        f"Cut-off                 : {_format_decimal_id(rf_threshold, 2)}\n"
        f"Accuracy                : {_format_percent_id(rf.accuracy)}\n"
        f"Balanced accuracy       : {_format_percent_id(rf.balanced_accuracy)}\n"
        f"Precision               : {_format_percent_id(rf.precision)}\n"
        f"Recall                  : {_format_percent_id(rf.recall)}\n"
        f"F1-score                : {_format_percent_id(rf.f1)}\n"
        f"Inclusion error         : {_format_percent_id(rf.inclusion_error)}\n"
        f"Exclusion error         : {_format_percent_id(rf.exclusion_error)}\n"
        f"ROC-AUC                 : {_format_percent_id(rf.roc_auc)}\n"
        f"Average Precision       : {_format_percent_id(rf.average_precision)}\n\n"
        f"Parameter terbaik\n{rf_parameter_text}"
    )

    ax.text(
        0.04,
        0.84,
        pmt_text,
        va="top",
        fontsize=10,
        fontweight="normal",
        family="DejaVu Sans Mono",
        bbox=box_style,
        transform=ax.transAxes,
    )

    ax.text(
        0.53,
        0.84,
        rf_text,
        va="top",
        fontsize=10,
        fontweight="normal",
        family="DejaVu Sans Mono",
        bbox=box_style,
        transform=ax.transAxes,
    )

    if p_value >= 0.05:
        conclusion = (
            "Exact McNemar Test menghasilkan p-value "
            f"{_format_decimal_id(p_value, 4)} ≥ 0,05. "
            "Tidak terdapat perbedaan kesalahan klasifikasi yang "
            "signifikan antara PMT dan Random Forest."
        )
    else:
        conclusion = (
            "Exact McNemar Test menghasilkan p-value "
            f"{_format_decimal_id(p_value, 4)} < 0,05. "
            "Terdapat perbedaan kesalahan klasifikasi yang "
            "signifikan antara PMT dan Random Forest."
        )

    ax.text(
        0.5,
        0.055,
        conclusion,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="normal",
        wrap=True,
        transform=ax.transAxes,
    )

    _save(fig, path)
