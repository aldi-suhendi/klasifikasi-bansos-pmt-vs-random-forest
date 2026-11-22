from __future__ import annotations

import html
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, QDate, QLocale, QTimer, QPointF, QRectF, QSize
from PyQt5.QtGui import (QBrush, QColor, QFont, QIcon, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QPolygonF)
from PyQt5.QtWidgets import (
    QApplication,
    QCalendarWidget,
    QComboBox,
    QDateEdit,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplashScreen,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.auth import AuthenticationError, change_operator_password, verify_operator_credentials
from src.environment import validate_environment
from src.integrity import dataset_fingerprint, validate_artifact_hashes
from src.model_io import load_joblib_model


APP_TITLE = "Sistem Klasifikasi Bantuan Sosial Kelurahan Pakojan"
DEFAULT_DATA_FILE = "Data-Dummy.xlsx"
TARGET = "KETERANGAN"
POSITIVE_LABEL = "LAYAK"
NEGATIVE_LABEL = "TIDAK LAYAK"
RECOMMEND_POSITIVE = "REKOMENDASI LAYAK"
RECOMMEND_NEGATIVE = "REKOMENDASI TIDAK LAYAK"
NEEDS_VERIFICATION = "PERLU VERIFIKASI"
REFERENCE_DATE = "2025-12-31"
RANDOM_STATE = 42

DATASET_COLUMNS: List[str] = [
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
    TARGET,
]

FEATURES: List[str] = [
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

MODEL_FEATURES: List[str] = ["USIA"] + [c for c in FEATURES if c != "TANGGAL_LAHIR"]
NUMERIC_FEATURES: List[str] = [
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
CATEGORICAL_FEATURES: List[str] = [c for c in MODEL_FEATURES if c not in NUMERIC_FEATURES]

IDENTITY_ADMIN_COLS: List[str] = [
    "NAMA_KEPALA_KELUARGA",
    "KECAMATAN",
    "DESA/KELURAHAN",
    "ALAMAT",
]
CONSTANT_EXCLUDED_COLS: List[str] = ["SUMBER_PENERANGAN_UTAMA", "PERAHU", "KAPAL_MOTOR"]
ASSET_COLS: List[str] = [
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
ALL_ASSET_COLS: List[str] = ASSET_COLS + ["PERAHU", "KAPAL_MOTOR"]

NEW_IDENTITY_REQUIRED: List[str] = ["NAMA_KEPALA_KELUARGA", "ALAMAT"]
NEW_TEMPLATE_COLUMNS: List[str] = [
    column
    for column in DATASET_COLUMNS
    if column not in CONSTANT_EXCLUDED_COLS + [TARGET]
]
CONSTANT_DEFAULTS: Dict[str, Any] = {
    "SUMBER_PENERANGAN_UTAMA": "LISTRIK PLN",
    "PERAHU": 0,
    "KAPAL_MOTOR": 0,
}

CATEGORY_OPTIONS: Dict[str, List[Any]] = {
    "PENDIDIKAN": [
        "TIDAK TAMAT SD/SEDERAJAT", "SD/SEDERAJAT", "SMP/SEDERAJAT",
        "SMA/SEDERAJAT", "DIPLOMA/SARJANA",
    ],
    "PEKERJAAN": [
        "TIDAK/BELUM BEKERJA", "BURUH HARIAN LEPAS", "BURUH PABRIK",
        "BURUH BANGUNAN", "ASISTEN RUMAH TANGGA", "TUKANG", "PETANI",
        "PEDAGANG", "PEKERJA JASA", "WIRASWASTA", "KURIR", "SOPIR",
        "MEKANIK", "SATPAM", "PENATA BUSANA", "PEGAWAI HONORER",
        "KARYAWAN SWASTA", "KARYAWAN BUMD", "KARYAWAN BUMN",
        "APARATUR PEMERINTAHAN", "PNS", "TNI", "POLRI", "BIDAN",
        "DOKTER", "PENSIUNAN",
    ],
    "JUMLAH-ART": [1, 2, 3, 4, 5, 6, 7],
    "KEPEMILIKAN_RUMAH": ["BEBAS SEWA/NUMPANG", "KONTRAK/SEWA", "MILIK SENDIRI"],
    "LANSIA": [0, 1, 2],
    "DISABILITAS": ["TIDAK ADA", "FISIK", "SENSORIK", "RUNGU/WICARA", "MENTAL"],
    "SAKIT_KRONIS": ["TIDAK ADA", "ASMA KRONIS", "DIABETES", "HIPERTENSI", "JANTUNG", "REMATIK", "TBC"],
    "LANTAI_TERLUAS": ["SEMEN", "UBIN/TEGEL/TERASO", "KERAMIK", "GRANIT/MARMER"],
    "DINDING_TERLUAS": ["KAYU/PAPAN", "TEMBOK"],
    "ATAP_TERLUAS": ["RUMBIA", "SENG", "ASBES", "GENTENG", "BETON"],
    "DAYA_LISTRIK_TERPASANG": [450, 900, 1300, 2200, 3500],
    "BAHAN_BAKAR_MEMASAK": ["KAYU BAKAR", "GAS 3 KG", "GAS 5,5 KG"],
    "FASILITAS_BAB": ["BERSAMA", "SENDIRI"],
    "JENIS_KLOSET": ["CEMPLUNG", "LEHER ANGSA"],
    "PEMBUANGAN_AKHIR_TINJA": ["LUBANG TANAH", "TANGKI SEPTIK"],
    "SUMBER_AIR_MINUM": ["SUMUR BOR/POMPA", "AIR ISI ULANG", "AIR KEMASAN BERMEREK"],
}
for _binary_col in ASSET_COLS:
    CATEGORY_OPTIONS[_binary_col] = [0, 1]

FEATURE_LABELS: Dict[str, str] = {
    "NAMA_KEPALA_KELUARGA": "Nama Lengkap Kepala Keluarga",
    "TANGGAL_LAHIR": "Tanggal Lahir",
    "USIA": "Usia",
    "KECAMATAN": "Kecamatan",
    "DESA/KELURAHAN": "Desa/Kelurahan",
    "ALAMAT": "Alamat",
    "PENDIDIKAN": "Pendidikan",
    "PEKERJAAN": "Pekerjaan",
    "JUMLAH-ART": "Jumlah Anggota Rumah Tangga",
    "KEPEMILIKAN_RUMAH": "Kepemilikan Rumah",
    "LANSIA": "Jumlah Lansia",
    "DISABILITAS": "Disabilitas",
    "SAKIT_KRONIS": "Sakit Kronis",
    "LANTAI_TERLUAS": "Jenis Lantai Terluas",
    "DINDING_TERLUAS": "Jenis Dinding Terluas",
    "ATAP_TERLUAS": "Jenis Atap Terluas",
    "SUMBER_PENERANGAN_UTAMA": "Sumber Penerangan Utama",
    "DAYA_LISTRIK_TERPASANG": "Daya Listrik Terpasang (VA)",
    "BAHAN_BAKAR_MEMASAK": "Bahan Bakar Memasak",
    "FASILITAS_BAB": "Fasilitas BAB",
    "JENIS_KLOSET": "Jenis Kloset",
    "PEMBUANGAN_AKHIR_TINJA": "Pembuangan Akhir Tinja",
    "SUMBER_AIR_MINUM": "Sumber Air Minum",
    "LEMARI_ES": "Lemari Es",
    "AC": "AC",
    "TV_LAYAR_DATAR": "TV Layar Datar",
    "KOMPUTER/LAPTOP": "Komputer/Laptop",
    "SEPEDA_MOTOR": "Sepeda Motor",
    "SEPEDA": "Sepeda",
    "MOBIL": "Mobil",
    "PERAHU": "Perahu",
    "KAPAL_MOTOR": "Kapal Motor",
    "SMARTPHONE": "Smartphone",
    "TERNAK_BESAR": "Ternak Besar",
    "TERNAK_KECIL": "Ternak Kecil",
    TARGET: "Status Kelayakan",
}

NAVY = "#0B2A4A"
NAVY_2 = "#0F3A63"
PRIMARY = "#2563EB"
PRIMARY_DARK = "#1D4ED8"
GOLD = "#D6A742"
GREEN = "#16A34A"
RED = "#DC2626"
ORANGE = "#F59E0B"
BG = "#F5F7FA"
SURFACE = "#FFFFFF"
SURFACE_2 = "#F8FAFC"
TEXT = "#0F172A"
MUTED = "#64748B"
BORDER = "#E2E8F0"
BORDER_DARK = "#CBD5E1"


def friendly_column_name(column: str) -> str:
    return FEATURE_LABELS.get(column, str(column).replace("_", " ").replace("-", " ").title())


def fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}".replace(",", ".")
    except Exception:
        return str(value)


def fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.2f}%".replace(".", ",")
    except Exception:
        return str(value)


def display_ui_value(value: Any, default: str = "-") -> str:
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat", "null", "<na>"}:
        return default
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


PASSWORD_MIN_LENGTH = 8


def password_policy_checks(password: str) -> Dict[str, bool]:
    return {
        "length": len(password) >= PASSWORD_MIN_LENGTH,
        "uppercase": any(char.isupper() for char in password),
        "lowercase": any(char.islower() for char in password),
        "digit": any(char.isdigit() for char in password),
        "symbol": any(not char.isalnum() and not char.isspace() for char in password),
        "no_space": not any(char.isspace() for char in password),
    }


def password_policy_valid(password: str) -> bool:
    return all(password_policy_checks(password).values())


def password_policy_html(password: str) -> str:
    checks = password_policy_checks(password)
    requirements = [
        ("length", f"Minimal {PASSWORD_MIN_LENGTH} karakter"),
        ("uppercase", "Mengandung huruf besar (A-Z)"),
        ("lowercase", "Mengandung huruf kecil (a-z)"),
        ("digit", "Mengandung angka (0-9)"),
        ("symbol", "Mengandung simbol, misalnya ! @ # $ %"),
        ("no_space", "Tidak menggunakan spasi"),
    ]
    lines = []
    for key, label in requirements:
        passed = checks[key]
        icon = "✓" if passed else "•"
        color = "#15803D" if passed else "#64748B"
        lines.append(f"<span style='color:{color}'>{icon} {html.escape(label)}</span>")
    return "<b>Persyaratan kata sandi:</b><br>" + "<br>".join(lines)


def password_policy_error_text(password: str) -> str:
    checks = password_policy_checks(password)
    missing = []
    if not checks["length"]:
        missing.append(f"minimal {PASSWORD_MIN_LENGTH} karakter")
    if not checks["uppercase"]:
        missing.append("huruf besar")
    if not checks["lowercase"]:
        missing.append("huruf kecil")
    if not checks["digit"]:
        missing.append("angka")
    if not checks["symbol"]:
        missing.append("simbol")
    if not checks["no_space"]:
        missing.append("tanpa spasi")
    return "Kata sandi baru harus menggunakan " + ", ".join(missing) + "."


def info_table_html(rows: Iterable[Tuple[str, Any]], note: str = "", accent: str = "#2563EB") -> str:
    body = "".join(
        "<tr>"
        f"<td style='width:165px; padding:3px 8px 3px 0; font-weight:650; color:#334155'>{html.escape(str(label))}</td>"
        "<td style='width:12px; padding:3px 8px 3px 0; text-align:center; font-weight:700'>:</td>"
        f"<td style='padding:3px 0; color:#0F172A'>{html.escape(display_ui_value(value)).replace(chr(10), '<br>')}</td>"
        "</tr>"
        for label, value in rows
    )
    note_html = (
        f"<div style='margin-top:9px; padding-top:8px; border-top:1px solid #E2E8F0; color:#64748B'>"
        f"{html.escape(note)}</div>" if note else ""
    )
    return (
        f"<div style='border-left:4px solid {accent}; padding-left:10px'>"
        f"<table cellspacing='0' cellpadding='0' style='width:100%'>{body}</table>{note_html}</div>"
    )


def make_password_eye_icon(hidden: bool) -> QIcon:
    pixmap = QPixmap(28, 22)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    pen = QPen(QColor("#2C6EC8"), 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    painter.setPen(pen)

    eye = QPainterPath()
    eye.moveTo(2.5, 11)
    eye.cubicTo(7.0, 4.5, 12.0, 3.0, 14.0, 3.0)
    eye.cubicTo(19.0, 3.0, 23.0, 6.0, 25.5, 11)
    eye.cubicTo(22.0, 16.5, 18.0, 19.0, 14.0, 19.0)
    eye.cubicTo(9.5, 19.0, 5.5, 16.5, 2.5, 11)
    painter.drawPath(eye)
    painter.drawEllipse(QRectF(10.5, 7.5, 7.0, 7.0))

    if hidden:
        slash = QPen(QColor("#2C6EC8"), 2.4, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(slash)
        painter.drawLine(QPointF(4.0, 3.0), QPointF(24.0, 19.0))

    painter.end()
    return QIcon(pixmap)


def normalize_column_name(value: Any) -> str:
    return str(value).replace("\n", " ").strip().upper()


def normalize_text_series(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "NULL": pd.NA})
    )


def format_date(value: Any, default: str = "-") -> str:
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return default
    return date.strftime("%d-%m-%Y")


def calculate_age(series: pd.Series, reference_date: str = REFERENCE_DATE) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce")
    ref = pd.Timestamp(reference_date)
    age = ref.year - dates.dt.year
    before_birthday = (ref.month < dates.dt.month) | (
        (ref.month == dates.dt.month) & (ref.day < dates.dt.day)
    )
    return pd.to_numeric(age - before_birthday.astype("Int64"), errors="coerce")


def _looks_like_dataset_header(values: Iterable[Any], require_target: bool = True) -> bool:
    normalized = {normalize_column_name(v) for v in values if pd.notna(v)}
    core = {"NAMA_KEPALA_KELUARGA", "TANGGAL_LAHIR", "PENDIDIKAN", "PEKERJAAN"}
    return core.issubset(normalized) and (TARGET in normalized if require_target else True)


def read_verval_excel(path: str | Path, require_target: bool = True) -> Tuple[pd.DataFrame, int]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    preview = pd.read_excel(path, header=None, nrows=20)
    header_row: Optional[int] = None
    for idx in range(min(20, len(preview))):
        if _looks_like_dataset_header(preview.iloc[idx].tolist(), require_target=require_target):
            header_row = idx
            break
    if header_row is None:
        raise ValueError(
            "Header dataset tidak dikenali. Pastikan file memakai struktur kolom yang didukung aplikasi, "
            "termasuk NAMA_KEPALA_KELUARGA, TANGGAL_LAHIR, PENDIDIKAN, dan KETERANGAN."
        )

    df = pd.read_excel(path, header=header_row)
    df.columns = [normalize_column_name(c) for c in df.columns]
    df = df.dropna(how="all").copy()
    unnamed = [c for c in df.columns if c.startswith("UNNAMED:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    duplicate_columns = sorted({c for c in df.columns if list(df.columns).count(c) > 1})
    if duplicate_columns:
        raise ValueError("Terdapat nama kolom ganda: " + ", ".join(duplicate_columns))

    missing_full = [c for c in DATASET_COLUMNS if c not in df.columns]
    if missing_full:
        raise ValueError("Kolom dataset belum lengkap: " + ", ".join(missing_full))
    unexpected = [c for c in df.columns if c not in DATASET_COLUMNS]
    if unexpected:
        raise ValueError(
            "Dataset memiliki kolom tambahan yang tidak termasuk skema aplikasi: "
            + ", ".join(unexpected)
        )

    source_column_count = len(df.columns)
    df = df[DATASET_COLUMNS].copy()
    df.attrs["source_column_count"] = source_column_count
    df["NAMA_KEPALA_KELUARGA"] = normalize_text_series(df["NAMA_KEPALA_KELUARGA"])
    df["KECAMATAN"] = normalize_text_series(df["KECAMATAN"])
    df["DESA/KELURAHAN"] = normalize_text_series(df["DESA/KELURAHAN"])
    df["ALAMAT"] = normalize_text_series(df["ALAMAT"])
    df["TANGGAL_LAHIR"] = pd.to_datetime(df["TANGGAL_LAHIR"], errors="coerce")

    if require_target:
        target = normalize_text_series(df[TARGET])
        invalid = ~target.isin({POSITIVE_LABEL, NEGATIVE_LABEL})
        if invalid.any():
            bad = sorted(target[invalid].dropna().astype(str).unique().tolist())
            raise ValueError("Nilai KETERANGAN tidak valid: " + ", ".join(bad[:10]))
        df[TARGET] = target

    return df.reset_index(drop=True), header_row


def prepare_model_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Kolom prediktor belum lengkap: " + ", ".join(missing))

    out = pd.DataFrame(index=df.index)
    out["USIA"] = calculate_age(df["TANGGAL_LAHIR"])
    for col in FEATURES:
        if col == "TANGGAL_LAHIR":
            continue
        if col in NUMERIC_FEATURES:
            out[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[col] = normalize_text_series(df[col])
    return out[MODEL_FEATURES]


def encode_target(series: pd.Series) -> pd.Series:
    normalized = normalize_text_series(series)
    mapped = normalized.map({NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1})
    if mapped.isna().any():
        raise ValueError("Target hanya boleh berisi LAYAK atau TIDAK LAYAK.")
    return mapped.astype(int)


def read_new_prediction_excel(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    preview = pd.read_excel(path, header=None, nrows=4, dtype=str)
    header_row = 0
    for idx in range(min(4, len(preview))):
        normalized = {normalize_column_name(v) for v in preview.iloc[idx].tolist() if pd.notna(v)}
        if {"NAMA_KEPALA_KELUARGA", "TANGGAL_LAHIR"}.issubset(normalized):
            header_row = idx
            break
    df = pd.read_excel(path, header=header_row)
    df.columns = [normalize_column_name(c) for c in df.columns]
    duplicate_columns = sorted({c for c in df.columns if list(df.columns).count(c) > 1})
    if duplicate_columns:
        raise ValueError("Terdapat nama kolom ganda: " + ", ".join(duplicate_columns))
    df = df.dropna(how="all")
    excel_rows = (df.index + header_row + 2).astype(int).tolist()
    df = df.reset_index(drop=True)
    df.attrs["excel_rows"] = excel_rows
    if df.empty:
        raise ValueError("File data baru tidak memiliki baris data.")

    missing = [c for c in NEW_TEMPLATE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Kolom data baru belum lengkap: " + ", ".join(missing))
    unexpected = [c for c in df.columns if c not in NEW_TEMPLATE_COLUMNS]
    if unexpected:
        raise ValueError(
            "File data baru memiliki kolom tambahan yang tidak diperlukan: "
            + ", ".join(unexpected)
        )
    result = df[NEW_TEMPLATE_COLUMNS].copy()
    result.attrs["excel_rows"] = excel_rows
    return result


def evaluate_predictions(y_true: pd.Series, pred: np.ndarray, prob: np.ndarray) -> Dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, pred) * 100,
        "Balanced Accuracy": balanced_accuracy_score(y_true, pred) * 100,
        "Precision": precision_score(y_true, pred, pos_label=1, zero_division=0) * 100,
        "Recall": recall_score(y_true, pred, pos_label=1, zero_division=0) * 100,
        "F1-Score": f1_score(y_true, pred, pos_label=1, zero_division=0) * 100,
        "Inclusion Error": (fp / (tp + fp) * 100) if (tp + fp) else 0.0,
        "Exclusion Error": (fn / (tp + fn) * 100) if (tp + fn) else 0.0,
        "ROC-AUC": roc_auc_score(y_true, prob) * 100,
        "Average Precision": average_precision_score(y_true, prob) * 100,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def exact_mcnemar(y_true: pd.Series, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, Any]:
    actual = np.asarray(y_true)
    correct_a = np.asarray(pred_a) == actual
    correct_b = np.asarray(pred_b) == actual
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    discordant = a_only + b_only
    p_value = float(binomtest(min(a_only, b_only), discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
    return {
        "both_correct": int(np.sum(correct_a & correct_b)),
        "pmt_only": a_only,
        "rf_only": b_only,
        "both_wrong": int(np.sum(~correct_a & ~correct_b)),
        "discordant": discordant,
        "p_value": p_value,
    }


def human_feature_name(encoded_name: str) -> str:
    name = str(encoded_name).replace("numeric__", "").replace("categorical__", "")
    for feature in sorted(MODEL_FEATURES, key=len, reverse=True):
        if name == feature:
            return friendly_column_name(feature)
        prefix = feature + "_"
        if name.startswith(prefix):
            return f"{friendly_column_name(feature)}: {name[len(prefix):]}"
    return name.replace("_", " ").title()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class ClickOnlyTabBar(QTabBar):

    def wheelEvent(self, event):
        event.ignore()


class ClickOnlyTabWidget(QTabWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabBar(ClickOnlyTabBar(self))


class ModernDateEdit(QWidget):

    MIN_DATE = QDate(1925, 1, 1)
    MAX_DATE = QDate(2007, 12, 31)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._selected_date: Optional[QDate] = None
        self.locale_id = QLocale(QLocale.Indonesian, QLocale.Indonesia)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.display = QLineEdit()
        self.display.setReadOnly(True)
        self.display.setPlaceholderText("Belum memilih tanggal lahir")
        self.display.setMinimumHeight(44)
        self.display.setMinimumWidth(225)

        self.pick_button = QPushButton("Pilih Tanggal")
        self.pick_button.setObjectName("SecondaryButton")
        self.pick_button.setCursor(Qt.PointingHandCursor)
        self.pick_button.setMinimumHeight(44)
        self.pick_button.setMinimumWidth(125)
        self.pick_button.clicked.connect(self.open_calendar_dialog)

        self.setMinimumWidth(370)
        layout.addWidget(self.display, 1)
        layout.addWidget(self.pick_button)

    def open_calendar_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Pilih Tanggal Lahir")
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dialog.resize(470, 500)
        dialog.setMinimumSize(440, 470)

        root = QVBoxLayout(dialog)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(14)

        title = QLabel("Pilih tanggal lahir warga")
        title.setObjectName("DialogTitle")
        root.addWidget(title)

        hint = QLabel("Rentang tanggal mengikuti validasi usia 18–100 tahun pada data tahun 2025.")
        hint.setObjectName("SmallText")
        hint.setWordWrap(True)
        root.addWidget(hint)

        calendar = QCalendarWidget(dialog)
        calendar.setLocale(self.locale_id)
        calendar.setGridVisible(True)
        calendar.setFirstDayOfWeek(Qt.Monday)
        calendar.setVerticalHeaderFormat(QCalendarWidget.NoVerticalHeader)
        calendar.setHorizontalHeaderFormat(QCalendarWidget.ShortDayNames)
        calendar.setMinimumDate(self.MIN_DATE)
        calendar.setMaximumDate(self.MAX_DATE)
        calendar.setNavigationBarVisible(True)
        calendar.setMinimumHeight(330)

        initial = self._selected_date or QDate(1980, 1, 1)
        calendar.setSelectedDate(initial)
        calendar.setCurrentPage(initial.year(), initial.month())
        root.addWidget(calendar, 1)

        button_row = QHBoxLayout()
        clear_btn = make_button("Hapus Pilihan", "secondary")
        cancel_btn = make_button("Batal", "secondary")
        select_btn = make_button("Gunakan Tanggal", "primary")
        button_row.addWidget(clear_btn)
        button_row.addStretch(1)
        button_row.addWidget(cancel_btn)
        button_row.addWidget(select_btn)
        root.addLayout(button_row)

        def choose_date():
            self.set_selected_date(calendar.selectedDate())
            dialog.accept()

        clear_btn.clicked.connect(lambda: (self.reset_value(), dialog.accept()))
        cancel_btn.clicked.connect(dialog.reject)
        select_btn.clicked.connect(choose_date)
        calendar.activated.connect(lambda _date: choose_date())
        dialog.exec_()

    def set_selected_date(self, qdate: QDate) -> None:
        self._selected_date = QDate(qdate)
        self.display.setText(self.locale_id.toString(qdate, "dd MMMM yyyy"))

    def selected_date_or_none(self) -> Optional[pd.Timestamp]:
        if self._selected_date is None:
            return None
        qdate = self._selected_date
        return pd.Timestamp(qdate.year(), qdate.month(), qdate.day())

    def reset_value(self) -> None:
        self._selected_date = None
        self.display.clear()


class DataEngine:
    def __init__(self):
        self.pmt_model = None
        self.rf_model = None
        self.metadata: Dict[str, Any] = {}
        self.model_dir: Optional[Path] = None
        self.model_ready = False
        self.model_error = ""
        self.load_models()
        self.reset_dataset()

    def candidate_model_dirs(self) -> List[Path]:
        script_dir = Path(__file__).resolve().parent
        cwd = Path.cwd()
        return [
            script_dir / "output" / "model",
            cwd / "output" / "model",
            script_dir / "models",
            cwd / "models",
        ]

    def load_models(self) -> bool:
        self.model_ready = False
        self.model_error = ""
        try:
            validate_environment(include_gui=True)
        except Exception as exc:
            self.model_error = str(exc)
            return False

        for directory in self.candidate_model_dirs():
            pmt_path = directory / "model_pmt.joblib"
            rf_path = directory / "model_random_forest.joblib"
            metadata_candidates = [
                directory / "metadata.json",
                directory / "model_metadata.json",
            ]
            meta_path = next((path for path in metadata_candidates if path.exists()), None)
            if pmt_path.exists() and rf_path.exists() and meta_path is not None:
                try:
                    with meta_path.open("r", encoding="utf-8") as file:
                        self.metadata = json.load(file)
                    validate_artifact_hashes(directory, self.metadata)
                    self.pmt_model = load_joblib_model(pmt_path)
                    self.rf_model = load_joblib_model(rf_path)
                    self.model_dir = directory
                    self.model_ready = True
                    return True
                except Exception as exc:
                    self.model_error = f"Model ditemukan tetapi gagal dimuat: {exc}"
        self.model_error = (
            "File model belum ditemukan. Pastikan folder output/model berisi "
            "model_pmt.joblib, model_random_forest.joblib, dan metadata.json."
        )
        return False

    def reset_dataset(self) -> None:
        self.loaded = False
        self.error_message = "Dataset belum dimuat. Buka menu Import Data untuk memilih dan memproses dataset."
        self.path: Optional[str] = None
        self.df: Optional[pd.DataFrame] = None
        self.header_row = 0
        self.original_column_count = 0
        self.duplicate_count = 0
        self.missing_main_features = 0
        self.X_clean: Optional[pd.DataFrame] = None
        self.y_encoded: Optional[pd.Series] = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_indices: Optional[np.ndarray] = None
        self.best_cutoff = None
        self.best_f1_train = None
        self.rf_cutoff = None
        self.y_pred_pmt = None
        self.y_pred_rf = None
        self.y_prob_pmt = None
        self.y_prob_rf = None
        self.cm_pmt = None
        self.cm_rf = None
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.mcnemar: Dict[str, Any] = {}
        self.pmt_coefficients = pd.DataFrame()
        self.feature_importance = pd.DataFrame()
        self.classification_results = pd.DataFrame()
        self.new_prediction_results = pd.DataFrame()

    def clear_dataset(self) -> None:
        self.reset_dataset()
        self.error_message = "Dataset dan hasil evaluasi telah dikosongkan. Model tersimpan tetap aktif."

    @property
    def pmt_threshold(self) -> float:
        return float(self.metadata.get("pmt", {}).get("threshold", 0.67))

    @property
    def rf_threshold(self) -> float:
        return float(self.metadata.get("random_forest", {}).get("threshold", 0.51))

    def inspect_dataset(self, path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df, header_row = read_verval_excel(path, require_target=True)
        missing_predictor = int(df[FEATURES].isna().sum().sum())
        duplicate_count = int(df.duplicated().sum())
        layak = int((df[TARGET] == POSITIVE_LABEL).sum())
        tidak = int((df[TARGET] == NEGATIVE_LABEL).sum())
        unexpected_columns = []
        info = {
            "header_row": header_row,
            "rows": len(df),
            "columns": int(df.attrs.get("source_column_count", len(DATASET_COLUMNS))),
            "predictors": len(FEATURES),
            "identity_admin": len(IDENTITY_ADMIN_COLS),
            "constant_excluded": len(CONSTANT_EXCLUDED_COLS),
            "missing_predictor": missing_predictor,
            "duplicates": duplicate_count,
            "layak": layak,
            "tidak_layak": tidak,
            "unexpected_columns": unexpected_columns,
        }
        return df, info

    def load_dataset(self, path: str) -> bool:
        if not self.model_ready:
            raise RuntimeError(self.model_error)
        df, info = self.inspect_dataset(path)

        self.path = str(path)
        self.df = df.copy()
        df = self.df
        self.header_row = info["header_row"]
        self.original_column_count = info["columns"]
        self.duplicate_count = info["duplicates"]
        self.missing_main_features = info["missing_predictor"]

        X = prepare_model_features(df)
        y = encode_target(df[TARGET])

        expected_fingerprint = self.metadata.get("dataset", {}).get(
            "feature_target_fingerprint_sha256"
        )
        actual_fingerprint = dataset_fingerprint(X, y)
        if not expected_fingerprint or actual_fingerprint != expected_fingerprint:
            self.reset_dataset()
            raise ValueError(
                "Dataset berlabel tidak sesuai dengan dataset acuan model. "
                "Gunakan data/Data-Dummy.xlsx untuk evaluasi. "
                "Data warga baru dimasukkan melalui menu Prediksi Data Baru."
            )

        indices = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        self.test_indices = np.asarray(test_idx)
        self.X_clean = X
        self.y_encoded = y
        self.X_train = X.iloc[train_idx].copy()
        self.X_test = X.iloc[test_idx].copy()
        self.y_train = y.iloc[train_idx].copy()
        self.y_test = y.iloc[test_idx].copy()

        self.best_cutoff = self.pmt_threshold
        self.rf_cutoff = self.rf_threshold
        self.best_f1_train = float(self.metadata.get("pmt", {}).get("cv_f1", 0.0))

        self.y_prob_pmt = self.pmt_model.predict_proba(self.X_test)[:, 1]
        self.y_prob_rf = self.rf_model.predict_proba(self.X_test)[:, 1]
        self.y_pred_pmt = (self.y_prob_pmt >= self.pmt_threshold).astype(int)
        self.y_pred_rf = (self.y_prob_rf >= self.rf_threshold).astype(int)

        pmt_metrics = evaluate_predictions(self.y_test, self.y_pred_pmt, self.y_prob_pmt)
        rf_metrics = evaluate_predictions(self.y_test, self.y_pred_rf, self.y_prob_rf)
        self.metrics = {
            "Proxy Means Test": pmt_metrics,
            "Random Forest": rf_metrics,
        }
        self.cm_pmt = np.array([[pmt_metrics["TN"], pmt_metrics["FP"]], [pmt_metrics["FN"], pmt_metrics["TP"]]])
        self.cm_rf = np.array([[rf_metrics["TN"], rf_metrics["FP"]], [rf_metrics["FN"], rf_metrics["TP"]]])
        self.mcnemar = exact_mcnemar(self.y_test, self.y_pred_pmt, self.y_pred_rf)
        self.pmt_coefficients = self.build_pmt_coefficients()
        self.feature_importance = self.build_feature_importance()
        self.classification_results = self.build_classification_results()
        self.loaded = True
        self.error_message = ""
        return True

    def build_pmt_coefficients(self) -> pd.DataFrame:
        try:
            preprocess = self.pmt_model.named_steps["preprocess"]
            classifier = self.pmt_model.named_steps["model"]
            names = preprocess.get_feature_names_out()
            coefficients = classifier.coef_[0]
            frame = pd.DataFrame({"Fitur": names, "Koefisien": coefficients})
            frame["Nilai Absolut"] = frame["Koefisien"].abs()
            frame = frame.sort_values("Nilai Absolut", ascending=False).reset_index(drop=True)
            frame.insert(0, "Peringkat", np.arange(1, len(frame) + 1))
            frame["Fitur Ringkas"] = frame["Fitur"].apply(human_feature_name)
            return frame
        except Exception:
            return pd.DataFrame(
                columns=["Peringkat", "Fitur", "Koefisien", "Nilai Absolut", "Fitur Ringkas"]
            )

    def build_feature_importance(self) -> pd.DataFrame:
        try:
            preprocess = self.rf_model.named_steps["preprocess"]
            classifier = self.rf_model.named_steps["model"]
            names = preprocess.get_feature_names_out()
            df = pd.DataFrame({"Fitur": names, "Importance": classifier.feature_importances_})
            df = df.sort_values("Importance", ascending=False).reset_index(drop=True)
            df.insert(0, "Peringkat", np.arange(1, len(df) + 1))
            df["Fitur Ringkas"] = df["Fitur"].apply(human_feature_name)
            return df
        except Exception:
            return pd.DataFrame(columns=["Peringkat", "Fitur", "Importance", "Fitur Ringkas"])

    def build_classification_results(self) -> pd.DataFrame:
        base = self.df.iloc[self.test_indices].reset_index(drop=True).copy()
        pmt_label = np.where(self.y_pred_pmt == 1, POSITIVE_LABEL, NEGATIVE_LABEL)
        rf_label = np.where(self.y_pred_rf == 1, POSITIVE_LABEL, NEGATIVE_LABEL)
        actual = np.where(np.asarray(self.y_test) == 1, POSITIVE_LABEL, NEGATIVE_LABEL)
        consistent = pmt_label == rf_label
        final = np.where(
            ~consistent,
            NEEDS_VERIFICATION,
            np.where(
                rf_label == POSITIVE_LABEL,
                RECOMMEND_POSITIVE,
                RECOMMEND_NEGATIVE,
            ),
        )
        result = pd.DataFrame({
            "No": np.arange(1, len(base) + 1),
            "Nama Lengkap Kepala Keluarga": base["NAMA_KEPALA_KELUARGA"].astype(str),
            "Tanggal Lahir": base["TANGGAL_LAHIR"].apply(format_date),
            "Usia": calculate_age(base["TANGGAL_LAHIR"]).astype("Int64"),
            "Alamat": base["ALAMAT"].astype(str),
            "Pendidikan": base["PENDIDIKAN"].astype(str),
            "Pekerjaan": base["PEKERJAAN"].astype(str),
            "Probabilitas PMT (%)": np.round(self.y_prob_pmt * 100, 2),
            "Prediksi PMT": pmt_label,
            "Probabilitas RF (%)": np.round(self.y_prob_rf * 100, 2),
            "Prediksi Random Forest": rf_label,
            "Status Aktual": actual,
            "Keterangan Hasil": final,
            "Keterangan": np.where(
                ~consistent,
                "Hasil prediksi PMT dan Random Forest berbeda; perlu verifikasi lebih lanjut",
                np.where(
                    rf_label == POSITIVE_LABEL,
                    "Kedua model memberikan rekomendasi Layak",
                    "Kedua model memberikan rekomendasi Tidak Layak",
                ),
            ),
        })
        return result

    def validate_new_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            raise ValueError("Data baru masih kosong.")
        source = data.copy()
        excel_rows = list(source.attrs.get("excel_rows", []))
        source.columns = [normalize_column_name(c) for c in source.columns]
        duplicate_columns = sorted({c for c in source.columns if list(source.columns).count(c) > 1})
        if duplicate_columns:
            raise ValueError("Terdapat nama kolom ganda: " + ", ".join(duplicate_columns))

        allowed_columns = NEW_TEMPLATE_COLUMNS + CONSTANT_EXCLUDED_COLS
        missing = [c for c in NEW_TEMPLATE_COLUMNS if c not in source.columns]
        if missing:
            raise ValueError("Kolom wajib data baru belum lengkap: " + ", ".join(missing))
        unexpected = [c for c in source.columns if c not in allowed_columns]
        if unexpected:
            raise ValueError(
                "Data baru memiliki kolom tambahan yang tidak diperlukan: "
                + ", ".join(unexpected)
            )

        empty_details: List[str] = []
        for column in NEW_TEMPLATE_COLUMNS:
            series = source[column]
            text_values = series.astype("string").str.strip().str.lower()
            empty_mask = series.isna() | text_values.isna() | text_values.isin(
                {"", "nan", "none", "null", "<na>", "nat"}
            )
            if not empty_mask.any():
                continue

            positions = np.flatnonzero(empty_mask.to_numpy()).tolist()
            if len(excel_rows) == len(source):
                rows = [int(excel_rows[pos]) for pos in positions]
            else:
                rows = [int(pos + 2) for pos in positions]

            shown_rows = rows[:30]
            row_text = ", ".join(str(row) for row in shown_rows)
            if len(rows) > len(shown_rows):
                row_text += f" (dan {len(rows) - len(shown_rows)} baris lainnya)"
            empty_details.append(
                f"{friendly_column_name(column)}: baris Excel {row_text}"
            )

        if empty_details:
            raise ValueError(
                "Data wajib belum lengkap:\n- " + "\n- ".join(empty_details)
            )

        source["NAMA_KEPALA_KELUARGA"] = normalize_text_series(source["NAMA_KEPALA_KELUARGA"])
        source["ALAMAT"] = normalize_text_series(source["ALAMAT"])
        if source["NAMA_KEPALA_KELUARGA"].isna().any():
            raise ValueError("Nama lengkap kepala keluarga wajib diisi pada seluruh baris.")
        if source["ALAMAT"].isna().any():
            raise ValueError("Alamat wajib diisi pada seluruh baris.")

        source["KECAMATAN"] = normalize_text_series(source["KECAMATAN"])
        source["DESA/KELURAHAN"] = normalize_text_series(source["DESA/KELURAHAN"])
        if source["KECAMATAN"].isna().any() or source["DESA/KELURAHAN"].isna().any():
            raise ValueError("Kecamatan dan Desa/Kelurahan wajib diisi pada seluruh baris.")
        if (~source["KECAMATAN"].eq("PINANG")).any() or (~source["DESA/KELURAHAN"].eq("PAKOJAN")).any():
            raise ValueError("Model ini khusus Kecamatan Pinang, Kelurahan Pakojan.")

        for column, expected_value in CONSTANT_DEFAULTS.items():
            if column in source.columns:
                if isinstance(expected_value, str):
                    normalized = normalize_text_series(source[column])
                    invalid = normalized.isna() | ~normalized.eq(expected_value)
                else:
                    normalized = pd.to_numeric(source[column], errors="coerce").astype("Int64")
                    invalid = normalized.isna() | ~normalized.eq(expected_value)
                if invalid.any():
                    rows = (source.index[invalid] + 2).tolist()[:10]
                    raise ValueError(
                        f"Nilai {column} harus tetap {expected_value} sesuai konfigurasi aplikasi. "
                        f"Periksa baris: {rows}"
                    )
            source[column] = expected_value

        source["TANGGAL_LAHIR"] = pd.to_datetime(source["TANGGAL_LAHIR"], errors="coerce")
        if source["TANGGAL_LAHIR"].isna().any():
            raise ValueError("Tanggal lahir wajib diisi dengan format tanggal yang valid.")
        age = calculate_age(source["TANGGAL_LAHIR"])
        invalid_age = age.lt(18) | age.gt(100) | age.isna()
        if invalid_age.any():
            rows = (source.index[invalid_age] + 2).tolist()[:10]
            raise ValueError(f"Usia harus berada pada rentang 18–100 tahun. Periksa baris: {rows}")

        for feature, options in CATEGORY_OPTIONS.items():
            if feature in ASSET_COLS or feature in {"JUMLAH-ART", "LANSIA", "DAYA_LISTRIK_TERPASANG"}:
                source[feature] = pd.to_numeric(source[feature], errors="coerce")
                valid_values = set(int(v) for v in options)
                invalid = source[feature].isna() | ~source[feature].astype("Int64").isin(valid_values)
            else:
                source[feature] = normalize_text_series(source[feature])
                invalid = source[feature].isna() | ~source[feature].isin(set(options))
            if invalid.any():
                rows = (source.index[invalid] + 2).tolist()[:10]
                allowed = ", ".join(map(str, options))
                raise ValueError(f"Nilai {feature} tidak sesuai dataset pada baris {rows}. Pilihan: {allowed}")

        internal_columns = [c for c in DATASET_COLUMNS if c != TARGET]
        validated = source[internal_columns].reset_index(drop=True)
        validated.attrs["excel_rows"] = excel_rows
        return validated

    def predict_new_data(self, new_df: pd.DataFrame) -> pd.DataFrame:
        if not self.model_ready:
            raise RuntimeError(self.model_error)
        source = self.validate_new_data(new_df)
        X = prepare_model_features(source)
        pmt_prob = self.pmt_model.predict_proba(X)[:, 1]
        rf_prob = self.rf_model.predict_proba(X)[:, 1]
        pmt_pred = (pmt_prob >= self.pmt_threshold).astype(int)
        rf_pred = (rf_prob >= self.rf_threshold).astype(int)
        pmt_label = np.where(pmt_pred == 1, POSITIVE_LABEL, NEGATIVE_LABEL)
        rf_label = np.where(rf_pred == 1, POSITIVE_LABEL, NEGATIVE_LABEL)
        age = calculate_age(source["TANGGAL_LAHIR"]).astype("Int64")
        disagreement = pmt_label != rf_label
        final = np.where(
            disagreement,
            NEEDS_VERIFICATION,
            np.where(
                rf_label == POSITIVE_LABEL,
                RECOMMEND_POSITIVE,
                RECOMMEND_NEGATIVE,
            ),
        )

        notes = [
            "Hasil prediksi PMT dan Random Forest berbeda; perlu verifikasi lebih lanjut"
            if disagree
            else (
                "Kedua model memberikan rekomendasi Layak"
                if rf_value == POSITIVE_LABEL
                else "Kedua model memberikan rekomendasi Tidak Layak"
            )
            for disagree, rf_value in zip(disagreement, rf_label)
        ]

        result = pd.DataFrame({
            "No": np.arange(1, len(source) + 1),
            "Nama Lengkap Kepala Keluarga": source["NAMA_KEPALA_KELUARGA"],
            "Tanggal Lahir": source["TANGGAL_LAHIR"].apply(format_date),
            "Usia": age,
            "Alamat": source["ALAMAT"],
            "Probabilitas PMT (%)": np.round(pmt_prob * 100, 2),
            "Cut-off PMT (%)": round(self.pmt_threshold * 100, 2),
            "Prediksi PMT": pmt_label,
            "Probabilitas RF (%)": np.round(rf_prob * 100, 2),
            "Cut-off RF (%)": round(self.rf_threshold * 100, 2),
            "Prediksi Random Forest": rf_label,
            "Keterangan Hasil": final,
            "Keterangan": notes,
            "Tanggal Prediksi": datetime.now().strftime("%d-%m-%Y %H:%M"),
        })
        self.new_prediction_results = result.copy()
        return result

    def target_distribution(self) -> Tuple[int, int, int]:
        if self.y_encoded is None:
            return 0, 0, 0
        total = len(self.y_encoded)
        layak = int(self.y_encoded.sum())
        return layak, total - layak, total

    def metrics_dataframe(self) -> pd.DataFrame:
        rows = []
        for method, values in self.metrics.items():
            rows.append({
                "Metode": method,
                "Accuracy": round(values["Accuracy"], 2),
                "Balanced Accuracy": round(values["Balanced Accuracy"], 2),
                "Precision": round(values["Precision"], 2),
                "Recall": round(values["Recall"], 2),
                "F1-Score": round(values["F1-Score"], 2),
                "Inclusion Error": round(values["Inclusion Error"], 2),
                "Exclusion Error": round(values["Exclusion Error"], 2),
                "ROC-AUC": round(values["ROC-AUC"], 2),
                "Average Precision": round(values["Average Precision"], 2),
            })
        return pd.DataFrame(rows)

    def best_method_by_metric(self, metric: str = "F1-Score") -> Tuple[str, float, str, float, str]:
        if not self.metrics:
            return "-", 0.0, "-", 0.0, "Belum ada hasil evaluasi."
        pmt = float(self.metrics["Proxy Means Test"].get(metric, 0))
        rf = float(self.metrics["Random Forest"].get(metric, 0))
        if abs(pmt - rf) < 1e-12:
            return "Keduanya", pmt, "Keduanya", rf, f"Kedua metode memiliki {metric} yang sama."
        if rf > pmt:
            return "Random Forest", rf, "Proxy Means Test", pmt, (
                f"Random Forest lebih tinggi secara numerik pada {metric}: {fmt_pct(rf)} dibandingkan {fmt_pct(pmt)}."
            )
        return "Proxy Means Test", pmt, "Random Forest", rf, (
            f"Proxy Means Test lebih tinggi secara numerik pada {metric}: {fmt_pct(pmt)} dibandingkan {fmt_pct(rf)}."
        )

    def report_conclusion_text(self) -> str:
        _, _, _, _, sentence = self.best_method_by_metric("F1-Score")
        p_value = self.mcnemar.get("p_value", 1.0)
        significance = "tidak menunjukkan perbedaan pola kesalahan klasifikasi yang signifikan" if p_value >= 0.05 else "menunjukkan perbedaan pola kesalahan klasifikasi yang signifikan"
        pmt = self.metrics.get("Proxy Means Test", {})
        rf = self.metrics.get("Random Forest", {})
        comparison = (
            f"Pada data uji, PMT memiliki precision {fmt_pct(pmt.get('Precision', 0))} dan recall {fmt_pct(pmt.get('Recall', 0))}, "
            f"sedangkan Random Forest memiliki precision {fmt_pct(rf.get('Precision', 0))} dan recall {fmt_pct(rf.get('Recall', 0))}. "
        )
        return (
            f"{sentence} Uji Exact McNemar menghasilkan p-value {p_value:.4f}, sehingga {significance}. "
            + comparison
            + "Aplikasi merupakan sistem pendukung analisis dan bukan penetapan resmi penerima bantuan."
        )

    def export_excel(self, df: pd.DataFrame, default_name: str) -> Optional[str]:
        if df is None or df.empty:
            raise ValueError("Tidak ada data untuk diekspor.")
        path, _ = QFileDialog.getSaveFileName(None, "Simpan File", default_name, "Excel Files (*.xlsx);;CSV Files (*.csv)")
        if not path:
            return None
        if path.lower().endswith(".csv"):
            df.to_csv(path, index=False)
        else:
            if not path.lower().endswith(".xlsx"):
                path += ".xlsx"
            df.to_excel(path, index=False)
        return path

    def export_prediction_template(self) -> Optional[str]:
        path, _ = QFileDialog.getSaveFileName(
            None,
            "Simpan Template Prediksi",
            "template_prediksi_data_baru_pakojan.xlsx",
            "Excel Files (*.xlsx)",
        )
        if not path:
            return None
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"

        packaged_template = (
            Path(__file__).resolve().parent
            / "template"
            / "template_prediksi_data_baru_pakojan.xlsx"
        )
        if not packaged_template.exists():
            raise FileNotFoundError(
                "Template bawaan tidak ditemukan: " + str(packaged_template)
            )
        shutil.copyfile(packaged_template, path)
        return path

    def export_report_excel(self, default_name: str = "laporan_evaluasi_pakojan.xlsx") -> Optional[str]:
        if not self.loaded:
            raise ValueError("Dataset belum diproses.")
        path, _ = QFileDialog.getSaveFileName(None, "Export Laporan", default_name, "Excel Files (*.xlsx)")
        if not path:
            return None
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            self.metrics_dataframe().to_excel(writer, sheet_name="Evaluasi Model", index=False)
            pd.DataFrame(self.cm_pmt, index=["Aktual Tidak Layak", "Aktual Layak"], columns=["Prediksi Tidak Layak", "Prediksi Layak"]).to_excel(writer, sheet_name="CM PMT")
            pd.DataFrame(self.cm_rf, index=["Aktual Tidak Layak", "Aktual Layak"], columns=["Prediksi Tidak Layak", "Prediksi Layak"]).to_excel(writer, sheet_name="CM RF")
            pd.DataFrame([self.mcnemar]).to_excel(writer, sheet_name="Uji McNemar", index=False)
            self.pmt_coefficients.to_excel(writer, sheet_name="Koefisien PMT", index=False)
            self.feature_importance.to_excel(writer, sheet_name="Feature Importance RF", index=False)
            self.classification_results.to_excel(writer, sheet_name="Hasil Data Uji", index=False)
        return path


class ChartCanvas(FigureCanvas):
    def __init__(self, width=6, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor=SURFACE)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


def add_shadow(widget, blur=20, x=0, y=5, alpha=24):
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(blur)
    shadow.setOffset(x, y)
    shadow.setColor(QColor(15, 23, 42, alpha))
    widget.setGraphicsEffect(shadow)


class Card(QFrame):
    def __init__(self, title=None, subtitle=None, min_height=0):
        super().__init__()
        self.setObjectName("Card")
        self.setMinimumHeight(min_height)
        add_shadow(self, blur=22, y=6, alpha=18)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 14, 16, 14)
        self.layout.setSpacing(8)
        self.layout.setAlignment(Qt.AlignTop)
        if title:
            title_label = QLabel(title)
            title_label.setObjectName("CardTitle")
            self.layout.addWidget(title_label)
        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setObjectName("SmallText")
            subtitle_label.setWordWrap(True)
            self.layout.addWidget(subtitle_label)


class StatCard(QFrame):
    def __init__(self, title, value, subtitle, accent=PRIMARY):
        super().__init__()
        self.setObjectName("StatCard")
        add_shadow(self, blur=22, y=5, alpha=16)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(5)
        self.setMinimumHeight(92)
        self.setMaximumHeight(108)
        title_label = QLabel(title)
        title_label.setObjectName("StatTitle")
        title_label.setWordWrap(True)
        value_label = QLabel(value)
        value_label.setObjectName("StatValue")
        value_label.setStyleSheet(f"color:{accent};")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("StatSubtitle")
        subtitle_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addWidget(subtitle_label)


class ScrollPage(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setObjectName("PageScroll")
        root.addWidget(self.scroll)
        self.container = QWidget()
        self.scroll.setWidget(self.container)
        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(20, 16, 20, 22)
        self.layout.setSpacing(14)
        self.layout.setAlignment(Qt.AlignTop)

    def header(self, title, subtitle):
        box = QFrame()
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        left = QVBoxLayout()
        left.setSpacing(6)
        title_label = QLabel(title)
        title_label.setObjectName("PageTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("PageSubtitle")
        subtitle_label.setWordWrap(True)
        left.addWidget(title_label)
        left.addWidget(subtitle_label)
        row.addLayout(left, 1)
        self.layout.addWidget(box)


def make_button(text, role="primary"):
    button = QPushButton(text)
    button.setCursor(Qt.PointingHandCursor)
    button.setMinimumHeight(36)
    button.setObjectName({"primary": "PrimaryButton", "danger": "DangerButton"}.get(role, "SecondaryButton"))
    return button


def configure_table(table):
    table.setAlternatingRowColors(True)
    table.setSelectionBehavior(QTableWidget.SelectRows)
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setStretchLastSection(True)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
    table.setSortingEnabled(True)
    table.setWordWrap(False)
    table.setMinimumHeight(260)


def set_table_data(table, df, max_rows=500):
    sorting = table.isSortingEnabled()
    table.setSortingEnabled(False)
    table.clear()
    if df is None or df.empty:
        table.setRowCount(0)
        table.setColumnCount(0)
        table.setSortingEnabled(sorting)
        return
    show_df = df.head(max_rows).copy()
    table.setRowCount(len(show_df))
    table.setColumnCount(len(show_df.columns))
    table.setHorizontalHeaderLabels([str(c) for c in show_df.columns])
    for row in range(len(show_df)):
        for col_index, col in enumerate(show_df.columns):
            value = show_df.iloc[row, col_index]
            if "TANGGAL" in str(col).upper() or isinstance(value, (pd.Timestamp, datetime)):
                text = format_date(value) if not isinstance(value, str) else display_ui_value(value)
            elif isinstance(value, (float, np.floating)) and not pd.isna(value):
                text = f"{value:.2f}"
            else:
                text = display_ui_value(value)
            table.setItem(row, col_index, QTableWidgetItem(text))
    table.resizeColumnsToContents()
    table.setSortingEnabled(sorting)


class MetricCardRow(QWidget):
    def __init__(self):
        super().__init__()
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(16)

    def set_cards(self, cards):
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for index, (title, value, subtitle, color) in enumerate(cards):
            self.grid.addWidget(StatCard(title, value, subtitle, color), 0, index)


class DashboardPage(ScrollPage):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.header("Dashboard", "Ringkasan data, distribusi status kelayakan, dan performa PMT serta Random Forest.")
        self.cards = MetricCardRow()
        self.layout.addWidget(self.cards)
        grid = QGridLayout()
        grid.setSpacing(18)
        distribution_card = Card("Distribusi Status Kelayakan", "Komposisi LAYAK dan TIDAK LAYAK pada dataset.", 375)
        self.distribution_chart = ChartCanvas(width=6.4, height=3.5)
        distribution_card.layout.addWidget(self.distribution_chart)
        metric_card = Card("Perbandingan Metrik", "Perbandingan hasil evaluasi PMT dan Random Forest.", 375)
        self.metric_chart = ChartCanvas(width=8.0, height=3.5)
        metric_card.layout.addWidget(self.metric_chart)
        grid.addWidget(distribution_card, 0, 0)
        grid.addWidget(metric_card, 0, 1)
        self.layout.addLayout(grid)
        summary_card = Card("Status Dataset dan Model", "Informasi proses yang aktif di aplikasi.", 205)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setObjectName("ReadOnlyText")
        self.summary_text.setMaximumHeight(185)
        summary_card.layout.addWidget(self.summary_text)
        self.layout.addWidget(summary_card)
        self.refresh()

    def refresh(self):
        e = self.engine
        if not e.loaded:
            model_value = "Aktif" if e.model_ready else "Belum ditemukan"
            model_color = GREEN if e.model_ready else RED
            self.cards.set_cards([
                ("Dataset", "Belum dimuat", "Import dataset untuk melihat evaluasi", ORANGE),
                ("Model PMT", model_value, "Regresi logistik tersimpan", model_color),
                ("Model RF", model_value, "Random Forest tersimpan", model_color),
                ("Target", "KETERANGAN", "LAYAK sebagai kelas positif", PRIMARY),
            ])
            self.distribution_chart.fig.clear(); self.distribution_chart.draw()
            self.metric_chart.fig.clear(); self.metric_chart.draw()
            self.summary_text.setHtml(info_table_html(
                [
                    ("Dataset", "Belum dimuat"),
                    ("Model PMT", "Siap digunakan" if e.model_ready else e.model_error),
                    ("Model Random Forest", "Siap digunakan" if e.model_ready else e.model_error),
                ],
                "Buka menu Import Data, pilih dataset, lalu klik Validasi Data dan Proses Data.",
                GREEN if e.model_ready else RED,
            ))
            return
        layak, tidak, total = e.target_distribution()
        best, value, _, _, _ = e.best_method_by_metric("F1-Score")
        self.cards.set_cards([
            ("Total Data", fmt_int(total), "Data verifikasi Kelurahan Pakojan", PRIMARY),
            ("LAYAK", fmt_int(layak), f"{fmt_pct(layak / total * 100)} dari dataset", GREEN),
            ("TIDAK LAYAK", fmt_int(tidak), f"{fmt_pct(tidak / total * 100)} dari dataset", RED),
            ("F1-Score Terbaik", fmt_pct(value), best, GOLD),
        ])
        self.draw_distribution()
        self.draw_metrics()
        self.summary_text.setHtml(info_table_html(
            [
                ("Dataset aktif", os.path.basename(e.path)),
                ("Struktur", f"{e.original_column_count} kolom sumber; {len(FEATURES)} prediktor final"),
                ("Target", TARGET),
                ("Pembagian data", f"Latih {fmt_int(len(e.X_train))}; uji {fmt_int(len(e.X_test))}"),
                ("Cut-off PMT", f"{e.pmt_threshold:.2f}"),
                ("Cut-off Random Forest", f"{e.rf_threshold:.2f}"),
                ("Missing value prediktor", e.missing_main_features),
                ("Duplikat baris penuh", e.duplicate_count),
                ("Exact McNemar p-value", f"{e.mcnemar.get('p_value', 1.0):.4f}"),
            ],
            "Identitas/administrasi dan tiga kolom konstan tidak digunakan sebagai prediktor model.",
            GREEN,
        ))

    def draw_distribution(self):
        layak, tidak, total = self.engine.target_distribution()
        fig = self.distribution_chart.fig
        fig.clear()
        ax = fig.add_subplot(111)
        wedges, _ = ax.pie(
            [layak, tidak],
            startangle=90,
            colors=[GREEN, RED],
            radius=0.74,
            center=(0, 0.12),
            wedgeprops=dict(width=0.27, edgecolor="white", linewidth=4),
        )
        ax.text(0, 0.19, fmt_int(total), ha="center", va="center", fontsize=22, fontweight="bold", color=TEXT)
        ax.text(0, -0.02, "Total data", ha="center", va="center", fontsize=9.5, color=MUTED)
        ax.axis("equal")
        labels = [
            f"LAYAK  {fmt_int(layak)} ({fmt_pct(layak / total * 100)})",
            f"TIDAK LAYAK  {fmt_int(tidak)} ({fmt_pct(tidak / total * 100)})",
        ]
        ax.legend(
            wedges,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.04),
            frameon=False,
            fontsize=8.8,
            ncol=2,
            columnspacing=1.6,
            handlelength=1.15,
            handletextpad=0.55,
        )
        fig.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.25)
        self.distribution_chart.draw()

    def draw_metrics(self):
        e = self.engine
        metrics = ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1-Score"]
        display_metrics = ["Accuracy", "Balanced\nAccuracy", "Precision", "Recall", "F1-Score"]
        pmt = [e.metrics["Proxy Means Test"][m] for m in metrics]
        rf = [e.metrics["Random Forest"][m] for m in metrics]
        fig = self.metric_chart.fig
        fig.clear()
        ax = fig.add_subplot(111)
        y = np.arange(len(metrics)); height = 0.30
        bars1 = ax.barh(y + height/2, pmt, height, label="PMT", color=PRIMARY)
        bars2 = ax.barh(y - height/2, rf, height, label="Random Forest", color=GREEN)
        ax.set_yticks(y); ax.set_yticklabels(display_metrics, fontsize=9.0); ax.invert_yaxis(); ax.set_xlim(0, 107)
        ax.grid(axis="x", linestyle="--", alpha=0.22); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        for bar in list(bars1) + list(bars2):
            ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.1f}", va="center", fontsize=8)
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2)
        fig.subplots_adjust(left=0.27, right=0.985, top=0.97, bottom=0.24)
        self.metric_chart.draw()


class ImportDataPage(ScrollPage):
    def __init__(self, engine, main_window):
        super().__init__()
        self.engine = engine
        self.main_window = main_window
        self.selected_file: Optional[str] = None
        self.header("Import Data", "Pilih, validasi, dan proses dataset warga. Isi data dapat diperiksa melalui menu Data Warga.")
        top = QGridLayout(); top.setSpacing(18)
        file_card = Card("Pilih Dataset", "Mendukung file .xlsx atau .xls dengan judul dokumen pada baris pertama.", 225)
        row = QHBoxLayout()
        self.file_name = QLineEdit(); self.file_name.setReadOnly(True); self.file_name.setPlaceholderText("Belum ada file dipilih")
        browse = make_button("Pilih File", "primary"); browse.clicked.connect(self.choose_file)
        row.addWidget(self.file_name, 1); row.addWidget(browse)
        file_card.layout.addLayout(row)
        note = QLabel("Pilih dataset berlabel yang sesuai dengan struktur aplikasi, lalu lakukan validasi sebelum proses data.")
        note.setObjectName("SmallText"); note.setWordWrap(True); file_card.layout.addWidget(note)

        validation_card = Card("Status Validasi", "Pemeriksaan header, kolom, target, missing value, dan duplikat.", 225)
        self.validation_text = QLabel(info_table_html([("Status", "Belum divalidasi"), ("File", os.path.basename(self.selected_file) if self.selected_file else "Belum dipilih")], "Klik Validasi Data untuk memeriksa struktur dataset.", ORANGE))
        self.validation_text.setObjectName("InfoLabel"); self.validation_text.setWordWrap(True)
        self.progress = QProgressBar(); self.progress.setValue(0)
        buttons = QHBoxLayout()
        validate_btn = make_button("Validasi Data", "secondary"); validate_btn.clicked.connect(self.validate_file)
        process_btn = make_button("Proses Data", "primary"); process_btn.clicked.connect(self.process_file)
        reset_btn = make_button("Kosongkan Data", "danger"); reset_btn.clicked.connect(self.clear_data)
        buttons.addWidget(validate_btn); buttons.addWidget(process_btn); buttons.addWidget(reset_btn)
        validation_card.layout.addWidget(self.validation_text); validation_card.layout.addWidget(self.progress); validation_card.layout.addLayout(buttons)
        top.addWidget(file_card, 0, 0); top.addWidget(validation_card, 0, 1)
        self.layout.addLayout(top)

        self.refresh()

    def choose_file(self):
        default_dir = str((Path(__file__).resolve().parent / "data").resolve())
        path, _ = QFileDialog.getOpenFileName(self, "Pilih Dataset Verval Pakojan", default_dir, "Excel Files (*.xlsx *.xls)")
        if path:
            self.selected_file = path
            self.file_name.setText(os.path.basename(path))
            self.progress.setValue(0)
            self.validation_text.setText(info_table_html([("Status", "File dipilih"), ("File", os.path.basename(path))], "Klik Validasi Data untuk memeriksa kesesuaiannya.", PRIMARY))

    def validation_message(self, info: Dict[str, Any]) -> str:
        return info_table_html(
            [
                ("Status", "Dataset sesuai format Pakojan"),
                ("Header Excel", f"Baris ke-{info['header_row'] + 1}"),
                ("Observasi", fmt_int(info['rows'])),
                ("Jumlah kolom", info['columns']),
                ("Prediktor model", info['predictors']),
                ("Identitas/administrasi", f"{info['identity_admin']} kolom; tidak digunakan model"),
                ("Kolom konstan", f"{info['constant_excluded']} kolom; dikeluarkan"),
                ("Distribusi target", f"LAYAK {fmt_int(info['layak'])}; TIDAK LAYAK {fmt_int(info['tidak_layak'])}"),
                ("Missing value prediktor", info['missing_predictor']),
                ("Duplikat baris penuh", info['duplicates']),
            ],
            "Struktur kolom, tipe data, dan label target telah sesuai dengan kebutuhan aplikasi.",
            GREEN,
        )

    def validate_file(self):
        path = self.selected_file or self.engine.path
        if not path:
            QMessageBox.warning(self, "File belum dipilih", "Pilih file dataset terlebih dahulu.")
            return
        try:
            self.progress.setValue(35); QApplication.processEvents()
            df, info = self.engine.inspect_dataset(path)
            self.validation_text.setText(self.validation_message(info))
            self.progress.setValue(100)
        except Exception as exc:
            self.progress.setValue(0)
            self.validation_text.setText(info_table_html([("Status", "Validasi gagal"), ("Kesalahan", str(exc))], accent=RED))
            QMessageBox.critical(self, "Validasi gagal", str(exc))

    def process_file(self):
        path = self.selected_file or self.engine.path
        if not path:
            QMessageBox.warning(self, "File belum dipilih", "Pilih file dataset terlebih dahulu.")
            return
        try:
            self.progress.setValue(25); QApplication.processEvents()
            self.engine.load_dataset(path)
            self.progress.setValue(100)
            _, info = self.engine.inspect_dataset(path)
            self.validation_text.setText(self.validation_message(info) + "<div style='margin-top:8px; font-weight:700; color:#15803D'>Dataset dan evaluasi model berhasil dimuat.</div>")
            self.main_window.refresh_all_pages("Dataset dan hasil evaluasi berhasil dimuat")
            QMessageBox.information(self, "Berhasil", "Dataset berhasil diproses. Model tersimpan tidak dilatih ulang.")
        except Exception as exc:
            self.progress.setValue(0)
            QMessageBox.critical(self, "Proses gagal", str(exc))

    def clear_data(self):
        reply = QMessageBox.question(self, "Kosongkan Dataset", "Kosongkan dataset dan hasil evaluasi dari aplikasi? File asli dan model tidak akan dihapus.", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.selected_file = None
            self.engine.clear_dataset()
            self.file_name.clear(); self.progress.setValue(0)
            self.validation_text.setText(info_table_html([("Status dataset", "Dikosongkan"), ("Status model", "Tetap aktif")], "Prediksi data baru masih dapat digunakan.", ORANGE))
            self.main_window.refresh_all_pages("Dataset telah dikosongkan")

    def refresh(self):
        if self.engine.loaded and self.engine.df is not None:
            self.file_name.setText(os.path.basename(self.engine.path) if self.engine.path else "")
        elif not self.selected_file:
            self.file_name.clear()


class DataWargaPage(ScrollPage):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.filtered = pd.DataFrame()
        self.header("Data Warga", "Cari dan periksa data warga yang telah dimuat ke dalam aplikasi.")
        info = QLabel("Tabel menampilkan kolom utama agar mudah dibaca. Klik dua kali pada baris untuk melihat rincian seluruh variabel.")
        info.setObjectName("InfoLabel"); info.setWordWrap(True); self.layout.addWidget(info)
        filter_card = Card("Filter dan Pencarian", "Status menggunakan nilai asli KETERANGAN: LAYAK dan TIDAK LAYAK.", 100)
        row = QHBoxLayout()
        self.search = QLineEdit(); self.search.setPlaceholderText("Cari nama, alamat, pendidikan, pekerjaan, atau status...")
        self.status_filter = NoWheelComboBox(); self.status_filter.addItems(["Semua Status", POSITIVE_LABEL, NEGATIVE_LABEL])
        refresh_btn = make_button("Reset Filter", "secondary")
        refresh_btn.clicked.connect(self.refresh)
        row.addWidget(self.search, 4)
        row.addWidget(self.status_filter, 1)
        row.addWidget(refresh_btn)
        filter_card.layout.addLayout(row)
        self.result_count = QLabel("Menampilkan 0 data")
        self.result_count.setObjectName("SmallText")
        filter_card.layout.addWidget(self.result_count)
        self.layout.addWidget(filter_card)

        table_card = Card(
            "Tabel Data Warga",
            "Kolom utama ditampilkan agar mudah dibaca. Klik dua kali pada baris untuk melihat seluruh variabel.",
            445,
        )
        self.table = QTableWidget()
        configure_table(self.table)
        self.table.cellDoubleClicked.connect(self.show_detail)
        table_card.layout.addWidget(self.table)
        self.layout.addWidget(table_card)

        export_card = Card(
            "Export Data Warga",
            "Simpan data setelah selesai meninjau tabel dan menerapkan filter yang diperlukan.",
            105,
        )
        export_row = QHBoxLayout()
        export_row.addStretch(1)
        export_view_btn = make_button("Export Tampilan", "primary")
        export_view_btn.setToolTip(
            "Mengekspor data warga yang sedang terlihat setelah pencarian atau filter diterapkan."
        )
        export_view_btn.clicked.connect(self.export_filtered_data)
        export_all_btn = make_button("Export Semua Data", "secondary")
        export_all_btn.setToolTip("Mengekspor seluruh data warga yang telah dimuat.")
        export_all_btn.clicked.connect(self.export_all_data)
        export_row.addWidget(export_view_btn)
        export_row.addWidget(export_all_btn)
        export_card.layout.addLayout(export_row)
        self.layout.addWidget(export_card)
        self.search.textChanged.connect(self.apply_filters)
        self.status_filter.currentIndexChanged.connect(self.apply_filters)
        self.refresh()

    def refresh(self):
        self.search.blockSignals(True); self.status_filter.blockSignals(True)
        self.search.clear(); self.status_filter.setCurrentIndex(0)
        self.search.blockSignals(False); self.status_filter.blockSignals(False)
        self.apply_filters()

    def apply_filters(self):
        if not self.engine.loaded:
            self.filtered = pd.DataFrame()
            set_table_data(self.table, pd.DataFrame())
            self.result_count.setText("Menampilkan 0 data")
            return
        df = self.engine.df.copy()
        query = self.search.text().strip().lower()
        if query:
            search_cols = ["NAMA_KEPALA_KELUARGA", "ALAMAT", "PENDIDIKAN", "PEKERJAAN", TARGET]
            mask = pd.Series(False, index=df.index)
            for col in search_cols:
                mask |= df[col].astype(str).str.lower().str.contains(query, na=False)
            df = df[mask]
        if self.status_filter.currentText() != "Semua Status":
            df = df[df[TARGET] == self.status_filter.currentText()]
        self.filtered = df.reset_index(drop=True)
        cols = [
            "NAMA_KEPALA_KELUARGA", "TANGGAL_LAHIR", "ALAMAT",
            "PENDIDIKAN", "PEKERJAAN", "JUMLAH-ART", "KEPEMILIKAN_RUMAH", TARGET,
        ]
        show = self.filtered[cols].copy()
        show["TANGGAL_LAHIR"] = show["TANGGAL_LAHIR"].apply(format_date)
        show.columns = [friendly_column_name(c) for c in show.columns]
        show.insert(0, "No", np.arange(1, len(show) + 1))
        set_table_data(self.table, show, 1000)
        total = len(self.engine.df) if self.engine.df is not None else 0
        self.result_count.setText(
            f"Menampilkan {fmt_int(len(self.filtered))} dari {fmt_int(total)} data warga"
        )

    def show_detail(self, row, _col):
        if self.filtered.empty or row >= len(self.filtered):
            return
        data = self.filtered.iloc[row]
        dialog = QDialog(self)
        dialog.setWindowTitle("Detail Lengkap Data Warga")
        dialog.resize(1080, 780); dialog.setMinimumSize(980, 700)
        root = QVBoxLayout(dialog); root.setContentsMargins(20, 18, 20, 18); root.setSpacing(14)
        title = QLabel("Detail Lengkap Data Warga"); title.setObjectName("DialogTitle")
        root.addWidget(title)
        subtitle = QLabel(display_ui_value(data.get("NAMA_KEPALA_KELUARGA"), "Warga"))
        subtitle.setObjectName("PageSubtitle")
        root.addWidget(subtitle)
        tabs = QTabWidget(); tabs.setUsesScrollButtons(False); tabs.tabBar().setExpanding(True)
        groups = [
            ("Identitas", ["NAMA_KEPALA_KELUARGA", "TANGGAL_LAHIR"]),
            ("Lokasi", ["KECAMATAN", "DESA/KELURAHAN", "ALAMAT"]),
            ("Kondisi Individu", ["PENDIDIKAN", "PEKERJAAN", "JUMLAH-ART", "LANSIA", "DISABILITAS", "SAKIT_KRONIS"]),
            ("Hunian & Sanitasi", ["KEPEMILIKAN_RUMAH", "LANTAI_TERLUAS", "DINDING_TERLUAS", "ATAP_TERLUAS", "SUMBER_PENERANGAN_UTAMA", "DAYA_LISTRIK_TERPASANG", "BAHAN_BAKAR_MEMASAK", "FASILITAS_BAB", "JENIS_KLOSET", "PEMBUANGAN_AKHIR_TINJA", "SUMBER_AIR_MINUM"]),
            ("Kepemilikan Aset", ALL_ASSET_COLS),
            ("Status", [TARGET]),
        ]
        for tab_name, columns in groups:
            scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
            content = QWidget(); form = QFormLayout(content); form.setContentsMargins(18, 16, 18, 16); form.setHorizontalSpacing(28); form.setVerticalSpacing(12)
            for key in columns:
                key_label = QLabel(friendly_column_name(key)); key_label.setObjectName("DetailFieldLabel"); key_label.setMinimumWidth(235)
                value = format_date(data.get(key)) if key == "TANGGAL_LAHIR" else display_ui_value(data.get(key))
                value_label = QLabel(value); value_label.setObjectName("DetailFieldValue"); value_label.setWordWrap(True); value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                form.addRow(key_label, value_label)
            scroll.setWidget(content); tabs.addTab(scroll, tab_name)
        root.addWidget(tabs, 1)
        close = make_button("Tutup", "primary"); close.clicked.connect(dialog.accept)
        bottom = QHBoxLayout(); bottom.addStretch(1); bottom.addWidget(close); root.addLayout(bottom)
        dialog.exec_()

    def export_filtered_data(self):
        if not self.engine.loaded:
            QMessageBox.warning(self, "Dataset belum tersedia", "Muat dan proses dataset terlebih dahulu.")
            return
        if self.filtered.empty:
            QMessageBox.warning(self, "Tidak ada data", "Tidak ada data warga yang sesuai dengan pencarian atau filter aktif.")
            return
        try:
            path = self.engine.export_excel(self.filtered, "data_warga_sesuai_tampilan.xlsx")
            if path:
                QMessageBox.information(self, "Export berhasil", f"Data sesuai tampilan tersimpan di:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export gagal", str(exc))

    def export_all_data(self):
        if not self.engine.loaded or self.engine.df.empty:
            QMessageBox.warning(self, "Dataset belum tersedia", "Muat dan proses dataset terlebih dahulu.")
            return
        try:
            path = self.engine.export_excel(self.engine.df.copy(), "seluruh_data_warga_pakojan.xlsx")
            if path:
                QMessageBox.information(self, "Export berhasil", f"Seluruh data warga tersimpan di:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export gagal", str(exc))


class HasilKlasifikasiPage(ScrollPage):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.filtered = pd.DataFrame()
        self.header("Hasil Klasifikasi", "Tinjau hasil PMT dan Random Forest pada data yang telah diproses.")
        purpose = QLabel("Menu ini menampilkan hasil klasifikasi pada data berlabel. Untuk warga baru yang belum memiliki status aktual, gunakan menu Prediksi Data Baru.")
        purpose.setObjectName("InfoLabel"); purpose.setWordWrap(True); self.layout.addWidget(purpose)
        self.summary = MetricCardRow(); self.layout.addWidget(self.summary)
        filter_card = Card("Pencarian Hasil", "Pencarian mengikuti kolom Nama, Probabilitas PMT, dan Prediksi PMT.", 154)

        search_row = QHBoxLayout()
        search_row.setSpacing(10)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Cari nama, Probabilitas PMT, atau Prediksi PMT...")
        self.search.setMinimumWidth(360)
        self.search.setMaximumWidth(500)

        self.filter = NoWheelComboBox()
        self.filter.addItems(["Semua", RECOMMEND_POSITIVE, RECOMMEND_NEGATIVE, NEEDS_VERIFICATION])
        self.filter.setMinimumWidth(245)
        self.filter.setMaximumWidth(285)

        reset_btn = make_button("Reset Pencarian", "secondary")
        reset_btn.setMinimumWidth(150)
        reset_btn.setToolTip("Mengosongkan kata pencarian dan mengembalikan filter ke Semua.")
        reset_btn.clicked.connect(self.reset_filters)

        search_row.addWidget(self.search)
        search_row.addWidget(self.filter)
        search_row.addWidget(reset_btn)
        search_row.addStretch(1)
        filter_card.layout.addLayout(search_row)

        self.result_count = QLabel("0 hasil ditampilkan")
        self.result_count.setObjectName("SmallText")
        filter_card.layout.addWidget(self.result_count)
        self.layout.addWidget(filter_card)

        splitter = QSplitter(Qt.Horizontal)
        table_card = Card("Tabel Hasil Data Uji", "Klik baris untuk melihat rincian prediksi.", 445)
        self.table = QTableWidget()
        configure_table(self.table)
        self.table.cellClicked.connect(self.update_detail)
        table_card.layout.addWidget(self.table)
        detail_card = Card("Detail Hasil", "Probabilitas, cut-off, dan status aktual.", 445)
        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setObjectName("ReadOnlyText")
        detail_card.layout.addWidget(self.detail)
        splitter.addWidget(table_card)
        splitter.addWidget(detail_card)
        splitter.setSizes([950, 430])
        self.layout.addWidget(splitter)

        export_card = Card(
            "Export Hasil Klasifikasi",
            "Simpan hasil setelah selesai meninjau tabel dan detail klasifikasi.",
            105,
        )
        export_row = QHBoxLayout()
        export_row.addStretch(1)
        export_view = make_button("Export Tampilan", "primary")
        export_view.setToolTip(
            "Mengekspor hasil yang sedang terlihat setelah pencarian dan filter diterapkan."
        )
        export_view.clicked.connect(self.export_filtered_results)
        export_all = make_button("Export Semua Hasil", "secondary")
        export_all.setToolTip("Mengekspor seluruh hasil klasifikasi data uji.")
        export_all.clicked.connect(self.export_all_results)
        export_row.addWidget(export_view)
        export_row.addWidget(export_all)
        export_card.layout.addLayout(export_row)
        self.layout.addWidget(export_card)
        self.search.textChanged.connect(self.apply_filters); self.filter.currentIndexChanged.connect(self.apply_filters)
        self.refresh()

    def reset_filters(self):
        self.search.blockSignals(True)
        self.filter.blockSignals(True)
        self.search.clear()
        self.filter.setCurrentIndex(0)
        self.search.blockSignals(False)
        self.filter.blockSignals(False)
        self.apply_filters()
        self.search.setFocus()

    def refresh(self):
        if not self.engine.loaded:
            self.summary.set_cards([
                ("Data Uji", "-", "Dataset belum diproses", PRIMARY),
                ("LAYAK", "-", "Belum tersedia", GREEN),
                ("TIDAK LAYAK", "-", "Belum tersedia", RED),
                ("Perlu Verifikasi", "-", "Belum tersedia", ORANGE),
            ])
            set_table_data(self.table, pd.DataFrame())
            self.detail.setText("Dataset belum diproses.")
            self.result_count.setText("0 hasil ditampilkan")
            return
        df = self.engine.classification_results
        self.summary.set_cards([
            ("Data Uji", fmt_int(len(df)), "Stratified split 20%", PRIMARY),
            ("REKOMENDASI LAYAK", fmt_int((df["Keterangan Hasil"] == RECOMMEND_POSITIVE).sum()), "Kedua model konsisten", GREEN),
            ("REKOMENDASI TIDAK LAYAK", fmt_int((df["Keterangan Hasil"] == RECOMMEND_NEGATIVE).sum()), "Kedua model konsisten", RED),
            ("PERLU VERIFIKASI", fmt_int((df["Keterangan Hasil"] == NEEDS_VERIFICATION).sum()), "Hasil model berbeda", ORANGE),
        ])
        self.apply_filters()

    def apply_filters(self):
        if not self.engine.loaded:
            return
        df = self.engine.classification_results.copy()
        query = self.search.text().strip().lower()
        if query:
            search_columns = [
                "Nama Lengkap Kepala Keluarga",
                "Probabilitas PMT (%)",
                "Prediksi PMT",
            ]
            mask = pd.Series(False, index=df.index)
            for col in search_columns:
                mask |= df[col].astype(str).str.lower().str.contains(
                    query,
                    na=False,
                    regex=False,
                )
            df = df[mask]
        if self.filter.currentText() != "Semua":
            df = df[df["Keterangan Hasil"] == self.filter.currentText()]
        self.filtered = df.reset_index(drop=True)
        self.result_count.setText(f"{fmt_int(len(self.filtered))} hasil ditampilkan")
        cols = ["No", "Nama Lengkap Kepala Keluarga", "Probabilitas PMT (%)", "Prediksi PMT", "Probabilitas RF (%)", "Prediksi Random Forest", "Status Aktual", "Keterangan Hasil"]
        set_table_data(self.table, self.filtered[cols], 700)
        if not self.filtered.empty:
            self.table.selectRow(0)
            self.update_detail(0, 0)
        else:
            self.detail.setText("Tidak ada hasil yang sesuai dengan pencarian atau filter.")

    def update_detail(self, row, _col):
        if self.filtered.empty or row >= len(self.filtered):
            return
        d = self.filtered.iloc[row]

        identity = info_table_html([
            ("Nama Kepala Keluarga", d.get("Nama Lengkap Kepala Keluarga")),
            ("Tanggal Lahir", d.get("Tanggal Lahir")),
            ("Usia", f"{display_ui_value(d.get('Usia'))} tahun"),
            ("Alamat", d.get("Alamat")),
            ("Pendidikan", d.get("Pendidikan")),
            ("Pekerjaan", d.get("Pekerjaan")),
        ], accent=PRIMARY)
        prediction = info_table_html([
            ("Probabilitas PMT", f"{display_ui_value(d.get('Probabilitas PMT (%)'))}%"),
            ("Cut-off PMT", f"{self.engine.pmt_threshold * 100:.0f}%"),
            ("Prediksi PMT", d.get("Prediksi PMT")),
            ("Probabilitas Random Forest", f"{display_ui_value(d.get('Probabilitas RF (%)'))}%"),
            ("Cut-off Random Forest", f"{self.engine.rf_threshold * 100:.0f}%"),
            ("Prediksi Random Forest", d.get("Prediksi Random Forest")),
        ], accent=GREEN)
        conclusion = info_table_html([
            ("Status Aktual", d.get("Status Aktual")),
            ("Keterangan Hasil", d.get("Keterangan Hasil")),
            ("Keterangan", d.get("Keterangan")),
        ], "Keputusan resmi tetap berada pada pihak kelurahan.", ORANGE)

        self.detail.setHtml(
            "<h3 style='margin:0 0 8px 0; color:#0B2A4A'>Identitas Warga</h3>" + identity +
            "<h3 style='margin:16px 0 8px 0; color:#0B2A4A'>Hasil Kedua Model</h3>" + prediction +
            "<h3 style='margin:16px 0 8px 0; color:#0B2A4A'>Kesimpulan Hasil</h3>" + conclusion
        )

    def export_filtered_results(self):
        try:
            if self.filtered.empty:
                raise ValueError("Tidak ada hasil pada tampilan yang dapat diekspor.")
            path = self.engine.export_excel(
                self.filtered,
                "hasil_klasifikasi_sesuai_pencarian.xlsx",
            )
            if path:
                QMessageBox.information(
                    self,
                    "Export berhasil",
                    f"Hasil yang sedang ditampilkan tersimpan di:\n{path}",
                )
        except Exception as exc:
            QMessageBox.critical(self, "Export gagal", str(exc))

    def export_all_results(self):
        try:
            path = self.engine.export_excel(
                self.engine.classification_results,
                "hasil_klasifikasi_semua_data_uji.xlsx",
            )
            if path:
                QMessageBox.information(
                    self,
                    "Export berhasil",
                    f"Seluruh hasil klasifikasi tersimpan di:\n{path}",
                )
        except Exception as exc:
            QMessageBox.critical(self, "Export gagal", str(exc))


class PrediksiDataBaruPage(ScrollPage):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.identity_widgets: Dict[str, Any] = {}
        self.feature_widgets: Dict[str, Any] = {}
        self.batch_source = pd.DataFrame()
        self.batch_result = pd.DataFrame()
        self.manual_result = pd.DataFrame()
        self.header(
            "Prediksi Data Baru",
            "Pilih cara input, lalu isi data warga pada tampilan yang lebih fokus.",
        )

        self.prediction_mode_stack = QStackedWidget()
        self.prediction_mode_stack.setObjectName("PredictionModeStack")
        self.prediction_mode_stack.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        self.prediction_mode_stack.addWidget(self.build_mode_choice_page())
        self.prediction_mode_stack.addWidget(self.build_manual_tab())
        self.prediction_mode_stack.addWidget(self.build_batch_tab())
        self.prediction_mode_stack.setCurrentIndex(0)
        self.layout.addWidget(self.prediction_mode_stack)

        self.refresh()

    def build_mode_choice_page(self):
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        status_card = Card(
            "Status Model",
            "Kesiapan model sebelum data warga diproses.",
            145,
        )
        self.model_status = QLabel()
        self.model_status.setObjectName("InfoLabel")
        self.model_status.setWordWrap(True)
        status_card.layout.addWidget(self.model_status)
        root.addWidget(status_card)

        choice_section = QFrame()
        choice_section.setObjectName("PredictionChoiceSection")
        choice_layout = QVBoxLayout(choice_section)
        choice_layout.setContentsMargins(22, 20, 22, 22)
        choice_layout.setSpacing(16)

        section_title = QLabel("Pilih Metode Input Data")
        section_title.setObjectName("PredictionSectionTitle")
        section_subtitle = QLabel(
            "Gunakan formulir untuk satu warga atau berkas Excel untuk memproses beberapa warga sekaligus."
        )
        section_subtitle.setObjectName("PageSubtitle")
        section_subtitle.setWordWrap(True)
        choice_layout.addWidget(section_title)
        choice_layout.addWidget(section_subtitle)

        option_row = QHBoxLayout()
        option_row.setSpacing(18)
        option_row.addWidget(
            self.create_prediction_option_card(
                "01",
                "Input Satu Warga",
                "Masukkan identitas, kondisi individu, hunian, sanitasi, dan aset melalui formulir terstruktur.",
                "Cocok untuk pemeriksaan individual",
                "Buka Formulir",
                0,
            ),
            1,
        )
        option_row.addWidget(
            self.create_prediction_option_card(
                "02",
                "Import Banyak Warga",
                "Unggah berkas Excel sesuai template, lakukan validasi, kemudian proses seluruh baris data.",
                "Cocok untuk pemrosesan massal",
                "Buka Import Excel",
                1,
            ),
            1,
        )
        choice_layout.addLayout(option_row)
        root.addWidget(choice_section)
        root.addStretch(1)
        return page

    def create_prediction_option_card(
        self,
        number: str,
        title: str,
        description: str,
        note: str,
        button_text: str,
        index: int,
    ) -> QFrame:
        card = QFrame()
        card.setObjectName("PredictionOptionCard")
        card.setMinimumHeight(245)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(10)

        top_row = QHBoxLayout()
        badge = QLabel(number)
        badge.setObjectName("PredictionOptionBadge")
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedSize(42, 42)
        top_row.addWidget(badge)
        top_row.addStretch(1)
        type_label = QLabel("METODE INPUT")
        type_label.setObjectName("PredictionOptionType")
        top_row.addWidget(type_label)
        layout.addLayout(top_row)

        title_label = QLabel(title)
        title_label.setObjectName("PredictionOptionTitle")
        layout.addWidget(title_label)

        description_label = QLabel(description)
        description_label.setObjectName("PredictionOptionDescription")
        description_label.setWordWrap(True)
        layout.addWidget(description_label)

        note_label = QLabel("✓  " + note)
        note_label.setObjectName("PredictionOptionNote")
        note_label.setWordWrap(True)
        layout.addWidget(note_label)
        layout.addStretch(1)

        open_button = QPushButton(button_text + "  →")
        open_button.setObjectName("PredictionOptionButton")
        open_button.setCursor(Qt.PointingHandCursor)
        open_button.setMinimumHeight(44)
        open_button.clicked.connect(lambda: self.switch_prediction_mode(index))
        layout.addWidget(open_button)
        return card

    def required_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("RequiredLabel")
        return label

    def switch_prediction_mode(
        self,
        index: int,
    ) -> None:
        if index not in (0, 1):
            return
        self.prediction_mode_stack.setCurrentIndex(index + 1)
        self.scroll.verticalScrollBar().setValue(0)

    def back_to_prediction_choices(self) -> None:
        self.prediction_mode_stack.setCurrentIndex(0)
        self.scroll.verticalScrollBar().setValue(0)

    def create_focus_header(self, title: str, subtitle: str) -> QFrame:
        bar = QFrame()
        bar.setObjectName("PredictionFocusBar")

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(18)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(3)

        title_label = QLabel(title)
        title_label.setObjectName("PredictionFocusTitle")

        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("PredictionFocusSubtitle")
        subtitle_label.setWordWrap(True)

        text_layout.addWidget(title_label)
        text_layout.addWidget(subtitle_label)
        layout.addLayout(text_layout, 1)

        back_button = QPushButton("Kembali ke Pilihan Input")
        back_button.setObjectName("PredictionBackTextButton")
        back_button.setToolTip("Kembali untuk memilih metode input data")
        back_button.setCursor(Qt.PointingHandCursor)
        back_button.setMinimumHeight(40)
        back_button.setMinimumWidth(190)
        back_button.clicked.connect(self.back_to_prediction_choices)
        layout.addWidget(back_button, 0, Qt.AlignVCenter)

        return bar

    def build_manual_tab(self):
        tab = QWidget()
        root = QVBoxLayout(tab)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        root.addWidget(self.create_focus_header(
            "Input Satu Warga",
            "Isi data secara bertahap, jalankan prediksi, lalu tinjau hasil pada bagian bawah.",
        ))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(2, 2, 8, 8)
        content_layout.setSpacing(14)

        identity_card = Card("Identitas dan Lokasi Warga", "Data ini digunakan untuk penelusuran dan tidak menjadi prediktor model.", 215)
        identity_grid = QGridLayout()
        identity_grid.setHorizontalSpacing(18)
        identity_grid.setVerticalSpacing(9)
        name = QLineEdit(); name.setPlaceholderText("Nama lengkap kepala keluarga")
        address = QLineEdit(); address.setPlaceholderText("Alamat domisili di Pakojan")
        district = QLineEdit("PINANG"); district.setReadOnly(True)
        village = QLineEdit("PAKOJAN"); village.setReadOnly(True)
        self.identity_widgets = {
            "NAMA_KEPALA_KELUARGA": name,
            "ALAMAT": address,
            "KECAMATAN": district,
            "DESA/KELURAHAN": village,
        }
        identity_fields = [
            ("Nama Kepala Keluarga", name),
            ("Alamat", address),
            ("Kecamatan", district),
            ("Kelurahan", village),
        ]
        for i, (label_text, widget) in enumerate(identity_fields):
            row, col = divmod(i, 2)
            identity_grid.addWidget(self.required_label(label_text), row * 2, col)
            identity_grid.addWidget(widget, row * 2 + 1, col)
        identity_grid.setColumnStretch(0, 1)
        identity_grid.setColumnStretch(1, 1)
        identity_card.layout.addLayout(identity_grid)
        content_layout.addWidget(identity_card)

        content_layout.addWidget(self.create_feature_card(
            "Kondisi Individu",
            "Karakteristik anggota rumah tangga yang diperlukan model.",
            ["TANGGAL_LAHIR", "PENDIDIKAN", "PEKERJAAN", "JUMLAH-ART", "LANSIA", "DISABILITAS", "SAKIT_KRONIS"],
        ))
        content_layout.addWidget(self.create_feature_card(
            "Hunian dan Sanitasi",
            "Karakteristik tempat tinggal, listrik, air, dan fasilitas sanitasi.",
            ["KEPEMILIKAN_RUMAH", "LANTAI_TERLUAS", "DINDING_TERLUAS", "ATAP_TERLUAS", "DAYA_LISTRIK_TERPASANG", "BAHAN_BAKAR_MEMASAK", "FASILITAS_BAB", "JENIS_KLOSET", "PEMBUANGAN_AKHIR_TINJA", "SUMBER_AIR_MINUM"],
        ))
        content_layout.addWidget(self.create_feature_card(
            "Kepemilikan Aset",
            "Gunakan nilai 0 untuk tidak memiliki dan 1 untuk memiliki.",
            ASSET_COLS,
        ))

        action_card = Card("Proses Prediksi", "Pastikan seluruh isian telah lengkap dan benar sebelum menjalankan prediksi.", 145)
        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        predict = make_button("Prediksi Sekarang", "primary")
        predict.clicked.connect(self.predict_manual)
        reset = make_button("Reset Form", "secondary")
        reset.clicked.connect(self.reset_manual_form)
        buttons.addWidget(predict)
        buttons.addWidget(reset)
        buttons.addStretch(1)
        action_card.layout.addLayout(buttons)
        content_layout.addWidget(action_card)

        result_card = Card("Hasil Prediksi", "Probabilitas dan rekomendasi dari kedua metode.", 430)
        self.manual_result_text = QTextEdit()
        self.manual_result_text.setReadOnly(True)
        self.manual_result_text.setObjectName("ReportPreview")
        self.manual_result_text.setMinimumHeight(330)
        self.manual_result_text.setText("Belum ada data yang diprediksi.")
        result_card.layout.addWidget(self.manual_result_text)
        content_layout.addWidget(result_card)

        export_card = Card(
            "Export Hasil Prediksi",
            "Simpan hasil setelah prediksi selesai dan hasil telah diperiksa.",
            105,
        )
        export_row = QHBoxLayout()
        export_row.addStretch(1)
        export = make_button("Export Hasil Satu Warga", "primary")
        export.clicked.connect(self.export_manual)
        export_row.addWidget(export)
        export_card.layout.addLayout(export_row)
        content_layout.addWidget(export_card)
        content_layout.addStretch(1)

        scroll.setWidget(content)
        root.addWidget(scroll, 1)
        return tab

    def create_feature_card(self, title: str, subtitle: str, columns: List[str]):
        card = Card(title, subtitle, 0)
        grid = QGridLayout()
        grid.setHorizontalSpacing(22)
        grid.setVerticalSpacing(12)
        for i, feature in enumerate(columns):
            row, col = divmod(i, 2)
            widget = self.create_feature_widget(feature)
            self.feature_widgets[feature] = widget
            grid.addWidget(self.required_label(friendly_column_name(feature)), row * 2, col)
            grid.addWidget(widget, row * 2 + 1, col)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        card.layout.addLayout(grid)
        return card

    def create_feature_widget(self, feature: str):
        if feature == "TANGGAL_LAHIR":
            return ModernDateEdit()
        combo = NoWheelComboBox(); combo.addItem("-- Pilih --", None)
        if feature in ASSET_COLS or feature in {"PERAHU", "KAPAL_MOTOR"}:
            combo.addItem("0", 0)
            combo.addItem("1", 1)
        elif feature == "SUMBER_PENERANGAN_UTAMA":
            combo.addItem("LISTRIK PLN", "LISTRIK PLN")
        else:
            for option in CATEGORY_OPTIONS[feature]:
                label = f"{option} VA" if feature == "DAYA_LISTRIK_TERPASANG" else str(option)
                if feature == "LANSIA": label = f"{option} orang"
                if feature == "JUMLAH-ART": label = f"{option} orang"
                combo.addItem(label, option)
        combo.setMinimumWidth(220)
        return combo

    def build_batch_tab(self):
        tab = QWidget()
        root = QVBoxLayout(tab)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(14)

        root.addWidget(self.create_focus_header(
            "Import Banyak Warga",
            "Pilih file Excel, validasi struktur, lalu jalankan prediksi untuk seluruh baris.",
        ))

        card = Card("Import Data Baru dari Excel", "Gunakan template agar nama kolom dan pilihan nilai sesuai dengan format aplikasi.", 235)
        file_row = QHBoxLayout()
        file_row.setSpacing(10)
        self.batch_path = QLineEdit()
        self.batch_path.setReadOnly(True)
        self.batch_path.setPlaceholderText("Belum ada file dipilih")
        choose = make_button("Pilih File", "primary"); choose.clicked.connect(self.choose_batch_file)
        template = make_button("Unduh Template", "secondary"); template.clicked.connect(self.save_template)
        file_row.addWidget(self.batch_path, 1)
        file_row.addWidget(choose)
        file_row.addWidget(template)
        card.layout.addLayout(file_row)

        self.batch_status = QLabel(info_table_html([
            ("Status", "Belum ada file dipilih"),
            ("Format", "31 kolom input"),
            ("Target", "Tidak diperlukan"),
        ], "Pilih file, lakukan validasi, kemudian jalankan prediksi.", ORANGE))
        self.batch_status.setObjectName("InfoLabel")
        self.batch_status.setWordWrap(True)
        card.layout.addWidget(self.batch_status)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        validate = make_button("Validasi File", "secondary")
        validate.clicked.connect(self.validate_batch)
        predict = make_button("Prediksi Data", "primary")
        predict.clicked.connect(self.predict_batch)
        button_row.addWidget(validate)
        button_row.addWidget(predict)
        button_row.addStretch(1)
        card.layout.addLayout(button_row)
        root.addWidget(card)

        preview_card = Card("Preview dan Hasil Prediksi", "Tabel menampilkan data setelah validasi atau hasil prediksi.", 460)
        self.batch_table = QTableWidget()
        configure_table(self.batch_table)
        preview_card.layout.addWidget(self.batch_table)
        root.addWidget(preview_card, 1)

        export_card = Card(
            "Export Hasil Prediksi",
            "Simpan hasil setelah seluruh data selesai diprediksi dan diperiksa.",
            105,
        )
        export_row = QHBoxLayout()
        export_row.addStretch(1)
        export = make_button("Export Hasil Banyak Warga", "primary")
        export.clicked.connect(self.export_batch)
        export_row.addWidget(export)
        export_card.layout.addLayout(export_row)
        root.addWidget(export_card)
        return tab

    def refresh(self):
        self.refresh_job_options_from_dataset()
        if self.engine.model_ready:
            self.model_status.setText(info_table_html(
                [
                    ("Status model", "Aktif dan siap digunakan"),
                    ("PMT", f"Regresi logistik; cut-off {self.engine.pmt_threshold:.2f}"),
                    ("Random Forest", f"Cut-off {self.engine.rf_threshold:.2f}"),
                    ("Pelatihan ulang", "Tidak dilakukan saat prediksi"),
                ],
                accent=GREEN,
            ))
        else:
            self.model_status.setText(info_table_html([("Status model", "Tidak siap"), ("Kesalahan", self.engine.model_error)], accent=RED))

    def refresh_job_options_from_dataset(self):
        combo = self.feature_widgets.get("PEKERJAAN")
        if combo is None or not isinstance(combo, QComboBox):
            return
        options = list(CATEGORY_OPTIONS["PEKERJAAN"])
        if self.engine.loaded and self.engine.df is not None and "PEKERJAAN" in self.engine.df.columns:
            options = (
                self.engine.df["PEKERJAAN"]
                .dropna()
                .astype(str)
                .str.strip()
                .drop_duplicates()
                .tolist()
            )
        current = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("-- Pilih --", None)
        for option in options:
            combo.addItem(str(option), option)
        if current in options:
            combo.setCurrentIndex(options.index(current) + 1)
        combo.blockSignals(False)

    def collect_manual_data(self) -> pd.DataFrame:
        name = self.identity_widgets["NAMA_KEPALA_KELUARGA"].text().strip()
        address = self.identity_widgets["ALAMAT"].text().strip()
        missing_identity = []
        if not name: missing_identity.append("Nama Kepala Keluarga")
        if not address: missing_identity.append("Alamat")
        if missing_identity:
            raise ValueError("Lengkapi identitas wajib: " + ", ".join(missing_identity))

        record: Dict[str, Any] = {
            "NAMA_KEPALA_KELUARGA": name,
            "KECAMATAN": "PINANG",
            "DESA/KELURAHAN": "PAKOJAN",
            "ALAMAT": address,
        }
        for feature, widget in self.feature_widgets.items():
            if isinstance(widget, ModernDateEdit):
                value = widget.selected_date_or_none()
            else:
                value = widget.currentData()
            if value is None:
                raise ValueError(f"{friendly_column_name(feature)} wajib dipilih.")
            record[feature] = value
        return pd.DataFrame([record])

    def predict_manual(self):
        try:
            source = self.collect_manual_data()
            self.manual_result = self.engine.predict_new_data(source)
            d = self.manual_result.iloc[0]
            status_color = "#B45309" if d["Keterangan Hasil"] == NEEDS_VERIFICATION else (GREEN if d["Keterangan Hasil"] == RECOMMEND_POSITIVE else RED)
            identity_html = info_table_html([
                ("Nama Kepala Keluarga", d.get("Nama Lengkap Kepala Keluarga")),
                ("Alamat", d.get("Alamat")),
                ("Usia", f"{display_ui_value(d.get('Usia'))} tahun"),
            ], accent=PRIMARY)
            pmt_html = info_table_html([
                ("Probabilitas LAYAK", f"{d['Probabilitas PMT (%)']:.2f}%"),
                ("Cut-off", f"{d['Cut-off PMT (%)']:.0f}%"),
                ("Prediksi", d.get("Prediksi PMT")),
            ], accent=PRIMARY)
            rf_html = info_table_html([
                ("Probabilitas LAYAK", f"{d['Probabilitas RF (%)']:.2f}%"),
                ("Cut-off", f"{d['Cut-off RF (%)']:.0f}%"),
                ("Prediksi", d.get("Prediksi Random Forest")),
            ], accent=GREEN)
            conclusion_html = info_table_html([
                ("Keterangan Hasil", d.get("Keterangan Hasil")),
                ("Keterangan", d.get("Keterangan")),
            ], "Hasil merupakan rekomendasi analitis. Keputusan resmi tetap melalui verifikasi kelurahan.", status_color)
            self.manual_result_text.setHtml(
                "<h3 style='margin:0 0 8px 0; color:#0B2A4A'>Identitas Warga</h3>" + identity_html +
                "<h3 style='margin:16px 0 8px 0; color:#0B2A4A'>PMT Berbasis Regresi Logistik</h3>" + pmt_html +
                "<h3 style='margin:16px 0 8px 0; color:#0B2A4A'>Random Forest</h3>" + rf_html +
                "<h3 style='margin:16px 0 8px 0; color:#0B2A4A'>Rekomendasi</h3>" + conclusion_html
            )
        except Exception as exc:
            QMessageBox.warning(self, "Prediksi gagal", str(exc))

    def reset_manual_form(self):
        for key, widget in self.identity_widgets.items():
            if key in {"KECAMATAN", "DESA/KELURAHAN"}: continue
            if isinstance(widget, QLineEdit): widget.clear()
            elif isinstance(widget, QComboBox): widget.setCurrentIndex(0)
        for widget in self.feature_widgets.values():
            if isinstance(widget, ModernDateEdit): widget.reset_value()
            else: widget.setCurrentIndex(0)
        self.manual_result = pd.DataFrame(); self.manual_result_text.setText("Belum ada data yang diprediksi.")

    def export_manual(self):
        try:
            path = self.engine.export_excel(self.manual_result, "hasil_prediksi_satu_warga.xlsx")
            if path: QMessageBox.information(self, "Export berhasil", f"Hasil tersimpan di:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export gagal", str(exc))

    def choose_batch_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Pilih Data Baru", "", "Excel Files (*.xlsx *.xls)")
        if path:
            self.batch_path.setText(path); self.batch_source = pd.DataFrame(); self.batch_result = pd.DataFrame()
            self.batch_status.setText(info_table_html([
                ("Status", "File dipilih"),
                ("File", os.path.basename(path)),
                ("Tahap berikutnya", "Validasi file"),
            ], "Klik Validasi File sebelum menjalankan prediksi.", PRIMARY))
            set_table_data(self.batch_table, pd.DataFrame())

    def validate_batch(self):
        path = self.batch_path.text().strip()
        if not path:
            QMessageBox.warning(self, "File belum dipilih", "Pilih file data baru terlebih dahulu."); return
        try:
            raw = read_new_prediction_excel(path)
            self.batch_source = self.engine.validate_new_data(raw)
            self.batch_status.setText(info_table_html([
                ("Status", "File valid"),
                ("Jumlah warga", fmt_int(len(self.batch_source))),
                ("Struktur kolom", "Sesuai"),
                ("Kelengkapan data", "Sesuai"),
            ], "File siap diproses menggunakan kedua model.", GREEN))
            preview_cols = ["NAMA_KEPALA_KELUARGA", "ALAMAT", "TANGGAL_LAHIR", "PENDIDIKAN", "PEKERJAAN"]
            preview = self.batch_source[preview_cols].copy(); preview["TANGGAL_LAHIR"] = preview["TANGGAL_LAHIR"].apply(format_date)
            set_table_data(self.batch_table, preview, 500)
        except Exception as exc:
            self.batch_source = pd.DataFrame()
            self.batch_result = pd.DataFrame()
            self.batch_status.setText(info_table_html([
                ("Status", "Validasi gagal"),
                ("Keterangan", str(exc)),
            ], "Perbaiki kolom dan baris yang disebutkan, lalu lakukan validasi kembali.", RED))
            QMessageBox.critical(self, "Validasi gagal", str(exc))

    def predict_batch(self):
        try:
            if self.batch_source.empty:
                path = self.batch_path.text().strip()
                if not path: raise ValueError("Pilih dan validasi file data baru terlebih dahulu.")
                self.batch_source = self.engine.validate_new_data(read_new_prediction_excel(path))
            self.batch_result = self.engine.predict_new_data(self.batch_source)
            review = int(self.batch_result["Keterangan Hasil"].str.contains("VERIFIKASI").sum())
            self.batch_status.setText(info_table_html([
                ("Status", "Prediksi selesai"),
                ("Jumlah warga", fmt_int(len(self.batch_result))),
                ("Perlu verifikasi", fmt_int(review)),
            ], "Hasil dapat ditinjau pada tabel dan diekspor ke Excel.", GREEN))
            set_table_data(self.batch_table, self.batch_result, 1000)
        except Exception as exc:
            self.batch_source = pd.DataFrame()
            self.batch_result = pd.DataFrame()
            self.batch_status.setText(info_table_html([
                ("Status", "Prediksi gagal"),
                ("Keterangan", str(exc)),
            ], "Perbaiki kolom dan baris yang disebutkan, lalu validasi file kembali.", RED))
            QMessageBox.critical(self, "Prediksi gagal", str(exc))

    def export_batch(self):
        try:
            path = self.engine.export_excel(self.batch_result, "hasil_prediksi_data_baru_pakojan.xlsx")
            if path: QMessageBox.information(self, "Export berhasil", f"Hasil tersimpan di:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export gagal", str(exc))

    def save_template(self):
        try:
            path = self.engine.export_prediction_template()
            if path: QMessageBox.information(self, "Template tersimpan", f"Template tersimpan di:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Gagal membuat template", str(exc))


class EvaluasiPage(ScrollPage):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.header("Evaluasi Model", "Ringkasan performa, pola kesalahan, dan interpretasi kedua model.")
        tabs = QTabWidget(); tabs.setUsesScrollButtons(False); tabs.tabBar().setExpanding(True)

        summary_tab = QWidget(); summary_layout = QVBoxLayout(summary_tab); summary_layout.setContentsMargins(4, 8, 4, 4); summary_layout.setSpacing(10)
        self.cards_pmt = MetricCardRow(); summary_layout.addWidget(self.cards_pmt)
        self.cards_rf = MetricCardRow(); summary_layout.addWidget(self.cards_rf)
        mcnemar_card = Card("Uji Exact McNemar", "Menguji apakah pola kesalahan kedua model berbeda secara statistik.", 92)
        self.mcnemar_label = QLabel("-"); self.mcnemar_label.setObjectName("InfoLabel"); self.mcnemar_label.setWordWrap(True); mcnemar_card.layout.addWidget(self.mcnemar_label)
        summary_layout.addWidget(mcnemar_card)
        tabs.addTab(summary_tab, "Ringkasan")

        cm_tab = QWidget(); cm_grid = QGridLayout(cm_tab); cm_grid.setContentsMargins(4, 8, 4, 4); cm_grid.setSpacing(14)
        pmt_card = Card("Confusion Matrix PMT", "Baris aktual dan kolom prediksi.", 470)
        rf_card = Card("Confusion Matrix Random Forest", "Baris aktual dan kolom prediksi.", 470)
        self.cm_pmt_chart = ChartCanvas(5.4, 4.0); self.cm_pmt_chart.setMinimumSize(420, 340)
        self.cm_rf_chart = ChartCanvas(5.4, 4.0); self.cm_rf_chart.setMinimumSize(420, 340)
        pmt_card.layout.addWidget(self.cm_pmt_chart); rf_card.layout.addWidget(self.cm_rf_chart)
        cm_grid.addWidget(pmt_card, 0, 0); cm_grid.addWidget(rf_card, 0, 1)
        tabs.addTab(cm_tab, "Confusion Matrix")

        error_tab = QWidget(); error_layout = QVBoxLayout(error_tab); error_layout.setContentsMargins(4, 8, 4, 4)
        error_card = Card("Inclusion dan Exclusion Error", "Inclusion = FP/(TP+FP); Exclusion = FN/(TP+FN).", 470)
        self.error_chart = ChartCanvas(8.5, 4.0); error_card.layout.addWidget(self.error_chart)
        error_layout.addWidget(error_card)
        tabs.addTab(error_tab, "Kesalahan")

        interpretation_tab = QWidget(); interpretation_grid = QGridLayout(interpretation_tab); interpretation_grid.setContentsMargins(4, 8, 4, 4); interpretation_grid.setSpacing(14)
        coef_card = Card("Koefisien PMT Regresi Logistik", "Sepuluh fitur dengan nilai absolut koefisien terbesar.", 430)
        self.pmt_coef_chart = ChartCanvas(6.0, 3.6); coef_card.layout.addWidget(self.pmt_coef_chart)
        fi_card = Card("Feature Importance Random Forest", "Sepuluh fitur hasil encoding dengan importance terbesar.", 430)
        self.fi_chart = ChartCanvas(6.0, 3.6); fi_card.layout.addWidget(self.fi_chart)
        interpretation_grid.addWidget(coef_card, 0, 0); interpretation_grid.addWidget(fi_card, 0, 1)
        interpretation_grid.setColumnStretch(0, 1); interpretation_grid.setColumnStretch(1, 1)
        tabs.addTab(interpretation_tab, "Interpretasi")

        table_tab = QWidget(); table_layout = QVBoxLayout(table_tab); table_layout.setContentsMargins(4, 8, 4, 4)
        table_card = Card("Tabel Evaluasi", "Ringkasan metrik kedua model.", 500)
        self.table = QTableWidget(); configure_table(self.table); table_card.layout.addWidget(self.table); table_layout.addWidget(table_card)
        tabs.addTab(table_tab, "Tabel")

        self.layout.addWidget(tabs, 1)
        self.refresh()

    def refresh(self):
        if not self.engine.loaded:
            blank = [("Accuracy", "-", "Belum ada dataset", PRIMARY), ("Balanced Accuracy", "-", "Belum tersedia", GOLD), ("Precision", "-", "Belum tersedia", ORANGE), ("Recall", "-", "Belum tersedia", GREEN), ("F1-Score", "-", "Belum tersedia", PRIMARY)]
            self.cards_pmt.set_cards(blank); self.cards_rf.set_cards(blank); self.mcnemar_label.setText("Dataset belum diproses.")
            for canvas in [self.cm_pmt_chart, self.cm_rf_chart, self.error_chart, self.pmt_coef_chart, self.fi_chart]: canvas.fig.clear(); canvas.draw()
            set_table_data(self.table, pd.DataFrame()); return
        pmt = self.engine.metrics["Proxy Means Test"]; rf = self.engine.metrics["Random Forest"]
        self.cards_pmt.set_cards([
            ("PMT Accuracy", fmt_pct(pmt["Accuracy"]), "Ketepatan keseluruhan", PRIMARY),
            ("PMT Balanced Acc.", fmt_pct(pmt["Balanced Accuracy"]), "Rata-rata sensitivitas kelas", GOLD),
            ("PMT Precision", fmt_pct(pmt["Precision"]), "Ketepatan prediksi LAYAK", ORANGE),
            ("PMT Recall", fmt_pct(pmt["Recall"]), "LAYAK yang dikenali", GREEN),
            ("PMT F1-Score", fmt_pct(pmt["F1-Score"]), "Keseimbangan precision-recall", PRIMARY),
        ])
        self.cards_rf.set_cards([
            ("RF Accuracy", fmt_pct(rf["Accuracy"]), "Ketepatan keseluruhan", PRIMARY),
            ("RF Balanced Acc.", fmt_pct(rf["Balanced Accuracy"]), "Rata-rata sensitivitas kelas", GOLD),
            ("RF Precision", fmt_pct(rf["Precision"]), "Ketepatan prediksi LAYAK", ORANGE),
            ("RF Recall", fmt_pct(rf["Recall"]), "LAYAK yang dikenali", GREEN),
            ("RF F1-Score", fmt_pct(rf["F1-Score"]), "Keseimbangan precision-recall", PRIMARY),
        ])
        mc = self.engine.mcnemar
        self.mcnemar_label.setText(
            f"Keduanya benar: {mc['both_correct']} | PMT saja benar: {mc['pmt_only']} | RF saja benar: {mc['rf_only']} | "
            f"Keduanya salah: {mc['both_wrong']} | p-value: {mc['p_value']:.4f}. "
            + ("Tidak terdapat perbedaan pola kesalahan klasifikasi yang signifikan pada taraf 5%." if mc['p_value'] >= 0.05 else "Terdapat perbedaan pola kesalahan klasifikasi yang signifikan pada taraf 5%.")
        )
        self.draw_cm(self.cm_pmt_chart, self.engine.cm_pmt, "PMT"); self.draw_cm(self.cm_rf_chart, self.engine.cm_rf, "Random Forest")
        self.draw_errors(); self.draw_pmt_coefficients(); self.draw_feature_importance(); set_table_data(self.table, self.engine.metrics_dataframe(), 20)


    def draw_cm(self, canvas, cm, title):
        fig = canvas.fig; fig.clear(); ax = fig.add_subplot(111); image = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(["Tidak Layak", "Layak"], fontsize=9); ax.set_yticklabels(["Tidak Layak", "Layak"], fontsize=9)
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual"); ax.set_title(title, fontsize=12, fontweight="bold")
        for i in range(2):
            for j in range(2): ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=14, fontweight="bold", color=TEXT)
        fig.colorbar(image, ax=ax, fraction=.046, pad=.04); fig.tight_layout(); canvas.draw()

    def draw_errors(self):
        pmt = self.engine.metrics["Proxy Means Test"]; rf = self.engine.metrics["Random Forest"]
        inc = [pmt["Inclusion Error"], rf["Inclusion Error"]]; exc = [pmt["Exclusion Error"], rf["Exclusion Error"]]
        fig = self.error_chart.fig; fig.clear(); ax = fig.add_subplot(111); x = np.arange(2); width=.34
        ax.bar(x-width/2, inc, width, label="Inclusion Error", color=ORANGE); ax.bar(x+width/2, exc, width, label="Exclusion Error", color=RED)
        ax.set_xticks(x); ax.set_xticklabels(["PMT", "Random Forest"]); ax.set_ylabel("Nilai (%)"); ax.grid(axis="y", linestyle="--", alpha=.25); ax.legend(frameon=False)
        ax.set_ylim(0, max(inc+exc)+8)
        for i,val in enumerate(inc): ax.text(i-width/2,val+.5,f"{val:.2f}%",ha="center",fontsize=9)
        for i,val in enumerate(exc): ax.text(i+width/2,val+.5,f"{val:.2f}%",ha="center",fontsize=9)
        fig.tight_layout(); self.error_chart.draw()

    def draw_pmt_coefficients(self):
        top = self.engine.pmt_coefficients.head(10).sort_values("Koefisien", ascending=True)
        fig = self.pmt_coef_chart.fig; fig.clear(); ax = fig.add_subplot(111)
        if not top.empty:
            ax.barh(top["Fitur Ringkas"], top["Koefisien"], color=PRIMARY)
            ax.axvline(0, color=TEXT, linewidth=0.8)
            ax.set_xlabel("Nilai Koefisien")
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="x", linestyle="--", alpha=.25)
            maximum = max(float(top["Koefisien"].abs().max()), 0.01)
            minimum_value = min(float(top["Koefisien"].min()), 0.0)
            maximum_value = max(float(top["Koefisien"].max()), 0.0)
            ax.set_xlim(minimum_value - maximum * 0.18, maximum_value + maximum * 0.23)
            for i, value in enumerate(top["Koefisien"]):
                offset = maximum * 0.025 if value >= 0 else -maximum * 0.025
                ax.text(value + offset, i, f"{value:.4f}", va="center", ha="left" if value >= 0 else "right", fontsize=7.5)
        fig.subplots_adjust(left=.48, right=.97, top=.94, bottom=.14); self.pmt_coef_chart.draw()

    def draw_feature_importance(self):
        top = self.engine.feature_importance.head(10).sort_values("Importance", ascending=True)
        fig = self.fi_chart.fig; fig.clear(); ax = fig.add_subplot(111)
        if not top.empty:
            ax.barh(top["Fitur Ringkas"], top["Importance"], color=NAVY_2); ax.set_xlabel("Nilai Importance"); ax.tick_params(axis="y", labelsize=8); ax.grid(axis="x", linestyle="--", alpha=.25)
            maximum = max(float(top["Importance"].max()), 0.01)
            ax.set_xlim(0, maximum * 1.22)
            for i,val in enumerate(top["Importance"]): ax.text(val+maximum*.015,i,f"{val:.4f}",va="center",fontsize=7.5)
        fig.subplots_adjust(left=.48, right=.97, top=.94, bottom=.14); self.fi_chart.draw()


class LaporanPage(ScrollPage):
    def __init__(self, engine):
        super().__init__(); self.engine = engine
        self.header("Laporan", "Pratinjau dan export hasil evaluasi, confusion matrix, koefisien PMT, feature importance RF, serta hasil klasifikasi data uji.")

        preview_card = Card("Pratinjau Laporan", "Ringkasan hasil analisis yang siap disimpan sebagai laporan.", 560)
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setObjectName("ReportPreview")
        preview_card.layout.addWidget(self.preview)
        self.layout.addWidget(preview_card)

        export_card = Card("Export Laporan", "Simpan laporan lengkap ke dalam berkas Microsoft Excel.", 120)
        export_row = QHBoxLayout()
        export_row.addStretch(1)
        export = make_button("Export Laporan Excel", "primary")
        export.clicked.connect(self.export_report)
        export_row.addWidget(export)
        export_card.layout.addLayout(export_row)
        self.layout.addWidget(export_card)
        self.refresh()

    def refresh(self):
        if not self.engine.loaded:
            self.preview.setText("Laporan belum tersedia.")
            return

        e = self.engine
        pmt = e.metrics["Proxy Means Test"]
        rf = e.metrics["Random Forest"]
        layak, tidak, total = e.target_distribution()
        def rows(items):
            return "".join(
                f"<tr><td class='label'>{label}</td><td class='colon'>:</td><td class='value'>{value}</td></tr>"
                for label, value in items
            )

        dataset_rows = rows([
            ("File", os.path.basename(e.path)),
            ("Total", fmt_int(total)),
            ("LAYAK", f"{fmt_int(layak)} ({fmt_pct(layak / total * 100)})"),
            ("TIDAK LAYAK", f"{fmt_int(tidak)} ({fmt_pct(tidak / total * 100)})"),
            ("Prediktor final", str(len(FEATURES))),
            ("Data latih/uji", f"{len(e.X_train)} / {len(e.X_test)}"),
        ])
        pmt_rows = rows([
            ("Cut-off", f"{e.pmt_threshold:.2f}"),
            ("Accuracy", fmt_pct(pmt["Accuracy"])),
            ("Balanced Accuracy", fmt_pct(pmt["Balanced Accuracy"])),
            ("Precision", fmt_pct(pmt["Precision"])),
            ("Recall", fmt_pct(pmt["Recall"])),
            ("F1-Score", fmt_pct(pmt["F1-Score"])),
            ("Inclusion Error", fmt_pct(pmt["Inclusion Error"])),
            ("Exclusion Error", fmt_pct(pmt["Exclusion Error"])),
        ])
        rf_rows = rows([
            ("Cut-off", f"{e.rf_threshold:.2f}"),
            ("Accuracy", fmt_pct(rf["Accuracy"])),
            ("Balanced Accuracy", fmt_pct(rf["Balanced Accuracy"])),
            ("Precision", fmt_pct(rf["Precision"])),
            ("Recall", fmt_pct(rf["Recall"])),
            ("F1-Score", fmt_pct(rf["F1-Score"])),
            ("Inclusion Error", fmt_pct(rf["Inclusion Error"])),
            ("Exclusion Error", fmt_pct(rf["Exclusion Error"])),
        ])

        self.preview.setHtml(f"""
        <style>
            body {{ font-family: 'Segoe UI'; color: #0F172A; font-size: 10pt; line-height: 1.45; }}
            h1 {{ font-size: 16pt; margin: 0 0 4px 0; color: #0B2A4A; }}
            h2 {{ font-size: 11.5pt; margin: 18px 0 7px 0; color: #0B2A4A; }}
            .subtitle {{ color: #64748B; margin-bottom: 14px; }}
            .title-box {{ background: #F8FAFC; border: 1px solid #E2E8F0; padding: 10px 12px; margin: 8px 0 12px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin: 0; }}
            td {{ padding: 3px 5px; vertical-align: top; }}
            td.label {{ width: 180px; font-weight: 600; }}
            td.colon {{ width: 12px; text-align: center; font-weight: 700; }}
            td.value {{ width: auto; }}
            .conclusion {{ background: #EFF6FF; border-left: 4px solid #2563EB; padding: 10px 12px; margin-top: 6px; }}
        </style>
        <h1>Laporan Ringkas Klasifikasi Bantuan Sosial</h1>
        <div class='subtitle'>Kelurahan Pakojan, Kecamatan Pinang, Kota Tangerang</div>
        <div class='title-box'><b>Sistem Analisis</b><br>Komparasi Proxy Means Test dan Random Forest untuk Klasifikasi Kelayakan Bantuan Sosial Kelurahan Pakojan</div>
        <h2>A. Dataset</h2><table>{dataset_rows}</table>
        <h2>B. PMT Berbasis Regresi Logistik</h2><table>{pmt_rows}</table>
        <h2>C. Random Forest</h2><table>{rf_rows}</table>
        <h2>D. Exact McNemar</h2><table>{rows([("p-value", f"{e.mcnemar['p_value']:.4f}")])}</table>
        <h2>E. Kesimpulan</h2><div class='conclusion'>{e.report_conclusion_text()}</div>
        """)

    def export_report(self):
        try:
            path = self.engine.export_report_excel()
            if path: QMessageBox.information(self, "Export berhasil", f"Laporan tersimpan di:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export gagal", str(exc))


class TentangPage(ScrollPage):
    def __init__(self, engine):
        super().__init__(); self.engine=engine
        self.header("Tentang Aplikasi", "Informasi aplikasi, metode, pengembang, dan batasan penggunaan.")
        card = Card("Sistem Klasifikasi Bantuan Sosial Kelurahan Pakojan", "Aplikasi desktop berbasis Python dan PyQt5.", 510)
        text = QTextEdit(); text.setReadOnly(True); text.setObjectName("ReportPreview")
        text.setText("""
Status Aplikasi:
Purwarupa yang siap digunakan secara lokal untuk mendukung analisis dan verifikasi data bantuan sosial di Kelurahan Pakojan.

Pengembang:
ALDI SUHENDI
Teknik Informatika, Fakultas Teknik, Universitas Islam Syekh-Yusuf

Dasar Pengembangan:
Komparasi Proxy Means Test dan Random Forest pada Klasifikasi Kelayakan Bantuan Sosial Kelurahan Pakojan

Dataset Acuan Model:
Data-Dummy.xlsx
- 2.000 observasi (sintetis)
- 35 kolom sumber
- 27 prediktor
- Target KETERANGAN: LAYAK dan TIDAK LAYAK

Metode:
1. Proxy Means Test yang dioperasionalkan melalui regresi logistik
2. Random Forest

Fungsi Aplikasi:
- Memvalidasi dan menampilkan dataset warga
- Menampilkan hasil evaluasi model
- Menampilkan data warga dan detail seluruh variabel
- Memprediksi satu atau banyak warga baru
- Menandai perbedaan hasil model untuk verifikasi lebih lanjut
- Mengekspor hasil dan laporan

Batasan Penggunaan:
Hasil prediksi merupakan rekomendasi analitis, bukan keputusan resmi penerima bantuan. Penetapan tetap memerlukan pemeriksaan administrasi, verifikasi lapangan, dan ketentuan pihak berwenang.
""".strip())
        card.layout.addWidget(text); self.layout.addWidget(card)


class LoginIllustrationWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("LoginIllustration")
        self.setMinimumWidth(500)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    @staticmethod
    def _font(size: float, weight: int = QFont.Normal) -> QFont:
        font = QFont("Segoe UI")
        font.setPointSizeF(size)
        font.setWeight(weight)
        return font

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        w = float(self.width())
        h = float(self.height())

        gradient = QLinearGradient(0, 0, w, h)
        gradient.setColorAt(0.0, QColor("#F9FCFF"))
        gradient.setColorAt(0.58, QColor("#EAF4FF"))
        gradient.setColorAt(1.0, QColor("#DCEEFF"))
        painter.fillRect(self.rect(), QBrush(gradient))

        emblem_x, emblem_y = 48.0, 48.0
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#0B5CAD"))
        painter.drawEllipse(QRectF(emblem_x, emblem_y, 68, 68))
        painter.setBrush(QColor("#EAF4FF"))
        painter.drawEllipse(QRectF(emblem_x + 8, emblem_y + 8, 52, 52))
        painter.setPen(QColor("#0B5CAD"))
        painter.setFont(self._font(15, QFont.Black))
        painter.drawText(QRectF(emblem_x + 8, emblem_y + 8, 52, 52), Qt.AlignCenter, "KP")

        painter.setPen(QColor("#123B67"))
        painter.setFont(self._font(22, QFont.Black))
        painter.drawText(QPointF(138, 79), "Kelurahan Pakojan")
        painter.setFont(self._font(12, QFont.Medium))
        painter.drawText(QPointF(138, 108), "Kecamatan Pinang · Kota Tangerang")

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(205, 226, 248, 175))
        for cx, cy, scale in [(90, 235, 1.0), (380, 205, 0.85), (520, 260, 0.65)]:
            painter.drawEllipse(QRectF(cx, cy, 75 * scale, 32 * scale))
            painter.drawEllipse(QRectF(cx + 28 * scale, cy - 18 * scale, 58 * scale, 48 * scale))
            painter.drawEllipse(QRectF(cx + 64 * scale, cy + 2 * scale, 64 * scale, 30 * scale))

        ground = h * 0.74
        painter.setBrush(QColor(183, 213, 242, 150))
        buildings = [
            (32, 330, 54, 165), (102, 286, 58, 210), (177, 246, 72, 250),
            (267, 325, 54, 170), (337, 275, 62, 220), (416, 350, 55, 145),
            (492, 298, 68, 196), (582, 338, 44, 156),
        ]
        for x, y, bw, bh in buildings:
            if x > w - 20:
                continue
            painter.drawRect(QRectF(x, y, bw, min(bh, ground - y)))
            painter.setBrush(QColor(225, 239, 253, 185))
            for row in range(3):
                for col in range(2):
                    wx = x + 11 + col * 21
                    wy = y + 22 + row * 35
                    if wy + 12 < ground:
                        painter.drawRect(QRectF(wx, wy, 11, 12))
            painter.setBrush(QColor(183, 213, 242, 150))

        bx = max(265.0, w * 0.48)
        by = ground - 172
        painter.setBrush(QColor("#F7FBFF"))
        painter.setPen(QPen(QColor("#8DB7DF"), 2))
        painter.drawRoundedRect(QRectF(bx, by + 58, 220, 120), 5, 5)
        roof = QPolygonF([
            QPointF(bx - 24, by + 64), QPointF(bx + 110, by), QPointF(bx + 244, by + 64)
        ])
        painter.setBrush(QColor("#255D91"))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(roof)
        painter.setBrush(QColor("#D7E9FA"))
        for col in range(4):
            painter.drawRect(QRectF(bx + 18 + col * 50, by + 86, 28, 38))
        painter.setBrush(QColor("#467AA9"))
        painter.drawRect(QRectF(bx + 87, by + 105, 46, 73))
        painter.setBrush(QColor("#B7D4EE"))
        painter.drawRect(QRectF(bx + 20, by + 139, 47, 10))
        painter.drawRect(QRectF(bx + 153, by + 139, 47, 10))

        cx, cy = 96.0, ground - 182
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#214F7B"))
        painter.drawRoundedRect(QRectF(cx, cy, 126, 176), 13, 13)
        painter.setBrush(QColor("#F8FBFF"))
        painter.drawRoundedRect(QRectF(cx + 10, cy + 13, 106, 150), 9, 9)
        painter.setBrush(QColor("#467AA9"))
        painter.drawRoundedRect(QRectF(cx + 38, cy - 10, 52, 28), 7, 7)
        check_pen = QPen(QColor("#35A66F"), 5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        line_pen = QPen(QColor("#C6DDF3"), 5, Qt.SolidLine, Qt.RoundCap)
        for i in range(3):
            yy = cy + 45 + i * 40
            painter.setPen(check_pen)
            painter.drawLine(QPointF(cx + 27, yy + 5), QPointF(cx + 35, yy + 14))
            painter.drawLine(QPointF(cx + 35, yy + 14), QPointF(cx + 50, yy - 4))
            painter.setPen(line_pen)
            painter.drawLine(QPointF(cx + 61, yy + 4), QPointF(cx + 97, yy + 4))

        painter.setPen(QPen(QColor("#EAF4FF"), 4))
        people = [(194, ground - 65, 24), (238, ground - 72, 29), (285, ground - 65, 24)]
        for px, py, r in people:
            painter.setBrush(QColor("#2F6598"))
            painter.drawEllipse(QRectF(px - r, py - r * 2.1, r * 2, r * 2))
            painter.drawRoundedRect(QRectF(px - r * 1.15, py - 5, r * 2.3, r * 1.55), r, r)

        painter.setPen(Qt.NoPen)
        for tx in [58.0, bx + 238]:
            painter.setBrush(QColor("#4A9B70"))
            painter.drawEllipse(QRectF(tx, ground - 112, 56, 70))
            painter.drawEllipse(QRectF(tx - 15, ground - 88, 54, 52))
            painter.drawEllipse(QRectF(tx + 25, ground - 86, 50, 50))
            painter.setBrush(QColor("#3C7F60"))
            painter.drawRect(QRectF(tx + 25, ground - 49, 8, 48))

        wave_1 = QPainterPath()
        wave_1.moveTo(0, h * 0.80)
        wave_1.cubicTo(w * 0.24, h * 0.70, w * 0.43, h * 0.86, w * 0.68, h * 0.76)
        wave_1.cubicTo(w * 0.83, h * 0.70, w * 0.92, h * 0.79, w, h * 0.75)
        wave_1.lineTo(w, h)
        wave_1.lineTo(0, h)
        wave_1.closeSubpath()
        painter.setBrush(QColor(104, 164, 218, 210))
        painter.drawPath(wave_1)

        wave_2 = QPainterPath()
        wave_2.moveTo(0, h * 0.88)
        wave_2.cubicTo(w * 0.30, h * 0.78, w * 0.46, h * 0.96, w * 0.72, h * 0.84)
        wave_2.cubicTo(w * 0.86, h * 0.78, w * 0.95, h * 0.87, w, h * 0.82)
        wave_2.lineTo(w, h)
        wave_2.lineTo(0, h)
        wave_2.closeSubpath()
        painter.setBrush(QColor(51, 118, 180, 215))
        painter.drawPath(wave_2)

        painter.setPen(QColor("#FFFFFF"))
        painter.setFont(self._font(11, QFont.DemiBold))
        painter.drawText(QPointF(50, h - 52), "Sistem Klasifikasi Kelayakan Bantuan Sosial")
        painter.setFont(self._font(9.5, QFont.Normal))
        painter.drawText(QPointF(50, h - 27), "PMT berbasis Regresi Logistik × Random Forest")


class LoginLockIcon(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(92, 92)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#E9F3FF"))
        painter.drawEllipse(QRectF(2, 2, 88, 88))
        painter.setPen(QPen(QColor("#2465B3"), 5, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(QRectF(31, 18, 30, 40), 0, 180 * 16)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#2465B3"))
        painter.drawRoundedRect(QRectF(25, 40, 42, 34), 7, 7)
        painter.setBrush(QColor("#E9F3FF"))
        painter.drawEllipse(QRectF(42, 50, 8, 8))
        painter.drawRoundedRect(QRectF(45, 55, 3, 10), 1.5, 1.5)


class LoginDialog(QDialog):

    MAX_ATTEMPTS = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self.authenticated_user: Optional[Dict[str, str]] = None
        self.failed_attempts = 0
        self.setWindowTitle("Sistem Klasifikasi Kelayakan Bantuan Sosial")
        self.setObjectName("LoginDialog")
        self.setModal(True)
        self.resize(1360, 780)
        self.setMinimumSize(1180, 700)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        content = QFrame()
        content.setObjectName("LoginContent")
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.illustration = LoginIllustrationWidget()
        content_layout.addWidget(self.illustration, 5)

        form_panel = QFrame()
        form_panel.setObjectName("LoginFormPanel")
        form_panel_layout = QVBoxLayout(form_panel)
        form_panel_layout.setContentsMargins(30, 38, 30, 38)
        form_panel_layout.setSpacing(0)
        form_panel_layout.addStretch(1)

        card = QFrame()
        card.setObjectName("LoginCard")
        card.setMaximumWidth(660)
        card.setMinimumWidth(570)
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(33, 67, 101, 35))
        card.setGraphicsEffect(shadow)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(38, 32, 38, 32)
        card_layout.setSpacing(10)

        lock_icon = LoginLockIcon()
        lock_row = QHBoxLayout()
        lock_row.addStretch(1)
        lock_row.addWidget(lock_icon)
        lock_row.addStretch(1)
        card_layout.addLayout(lock_row)

        title = QLabel("Masuk ke Aplikasi")
        title.setObjectName("LoginTitle")
        title.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(title)

        subtitle = QLabel("Gunakan akun operator yang telah terdaftar")
        subtitle.setObjectName("LoginSubtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(subtitle)
        card_layout.addSpacing(10)

        username_label = QLabel("Nama Pengguna")
        username_label.setObjectName("LoginFieldLabel")
        card_layout.addWidget(username_label)

        self.username_input = QLineEdit()
        self.username_input.setObjectName("LoginField")
        self.username_input.setPlaceholderText("Masukkan nama pengguna")
        self.username_input.clear()
        self.username_input.setMaxLength(64)
        self.username_input.setMinimumHeight(60)
        card_layout.addWidget(self.username_input)

        card_layout.addSpacing(4)
        password_label = QLabel("Kata Sandi")
        password_label.setObjectName("LoginFieldLabel")
        card_layout.addWidget(password_label)

        password_frame = QFrame()
        password_frame.setObjectName("LoginPasswordFrame")
        password_frame.setMinimumHeight(60)
        password_layout = QHBoxLayout(password_frame)
        password_layout.setContentsMargins(0, 0, 7, 0)
        password_layout.setSpacing(0)
        self.password_input = QLineEdit()
        self.password_input.setObjectName("LoginPasswordField")
        self.password_input.setPlaceholderText("Masukkan kata sandi")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setMaxLength(128)
        self.password_input.setMinimumHeight(58)
        self.password_input.returnPressed.connect(self.attempt_login)
        password_layout.addWidget(self.password_input, 1)
        self.password_toggle = QToolButton()
        self.password_toggle.setObjectName("PasswordToggle")
        self.password_toggle.setIcon(make_password_eye_icon(hidden=True))
        self.password_toggle.setIconSize(QSize(28, 22))
        self.password_toggle.setCheckable(True)
        self.password_toggle.setToolTip("Tampilkan atau sembunyikan kata sandi")
        self.password_toggle.setFixedSize(42, 42)
        self.password_toggle.setCursor(Qt.PointingHandCursor)
        self.password_toggle.toggled.connect(self.toggle_password_visibility)
        password_layout.addWidget(self.password_toggle)
        card_layout.addWidget(password_frame)

        self.error_label = QLabel("")
        self.error_label.setObjectName("LoginError")
        self.error_label.setWordWrap(True)
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.hide()
        card_layout.addWidget(self.error_label)

        self.login_button = QPushButton("Masuk")
        self.login_button.setObjectName("LoginPrimaryButton")
        self.login_button.setCursor(Qt.PointingHandCursor)
        self.login_button.setMinimumHeight(54)
        self.login_button.clicked.connect(self.attempt_login)
        card_layout.addWidget(self.login_button)

        note = QLabel("Akses aplikasi terbatas untuk operator yang telah memiliki akun.")
        note.setObjectName("LoginSecurityNote")
        note.setWordWrap(True)
        note.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        card_layout.addWidget(note)

        exit_button = QPushButton("Keluar dari aplikasi")
        exit_button.setObjectName("LoginExitButton")
        exit_button.setCursor(Qt.PointingHandCursor)
        exit_button.clicked.connect(self.reject)
        card_layout.addWidget(exit_button, 0, Qt.AlignCenter)

        form_panel_layout.addWidget(card, 0, Qt.AlignCenter)
        form_panel_layout.addStretch(1)
        content_layout.addWidget(form_panel, 6)
        root.addWidget(content, 1)

        footer = QFrame()
        footer.setObjectName("LoginFooter")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(28, 12, 28, 12)
        footer_layout.setSpacing(12)
        footer_left = QLabel("Sistem Klasifikasi Kelayakan Bantuan Sosial")
        footer_left.setObjectName("LoginFooterStrong")
        footer_layout.addWidget(footer_left)
        footer_layout.addStretch(1)
        self.attempts_label = QLabel(f"Percobaan login maksimal {self.MAX_ATTEMPTS} kali")
        self.attempts_label.setObjectName("LoginFooterText")
        footer_layout.addWidget(self.attempts_label)
        footer_right = QLabel("© 2026 Kelurahan Pakojan")
        footer_right.setObjectName("LoginFooterText")
        footer_layout.addWidget(footer_right)
        root.addWidget(footer)

        self.lock_seconds_remaining = 0
        self.lock_timer = QTimer(self)
        self.lock_timer.setInterval(1000)
        self.lock_timer.timeout.connect(self.update_login_lock)
        self.login_button.setDefault(True)
        QTimer.singleShot(0, self.login_button.setFocus)

    def toggle_password_visibility(self, visible: bool) -> None:
        self.password_input.setEchoMode(QLineEdit.Normal if visible else QLineEdit.Password)
        self.password_toggle.setIcon(make_password_eye_icon(hidden=not visible))
        self.password_toggle.setToolTip(
            "Sembunyikan kata sandi" if visible else "Tampilkan kata sandi"
        )

    def start_login_lock(self, seconds: int = 30) -> None:
        self.lock_seconds_remaining = seconds
        for widget in (self.username_input, self.password_input, self.password_toggle, self.login_button):
            widget.setEnabled(False)
        self.show_error(
            f"Batas {self.MAX_ATTEMPTS} percobaan tercapai. Coba kembali dalam {seconds} detik."
        )
        self.attempts_label.setText(f"Login dikunci sementara: {seconds} detik")
        self.lock_timer.start()

    def update_login_lock(self) -> None:
        self.lock_seconds_remaining -= 1
        if self.lock_seconds_remaining > 0:
            self.attempts_label.setText(
                f"Login dikunci sementara: {self.lock_seconds_remaining} detik"
            )
            return

        self.lock_timer.stop()
        self.failed_attempts = 0
        for widget in (self.username_input, self.password_input, self.password_toggle, self.login_button):
            widget.setEnabled(True)
        self.attempts_label.setText(
            f"Percobaan login maksimal {self.MAX_ATTEMPTS} kali"
        )
        self.error_label.hide()
        self.password_input.clear()
        self.login_button.setFocus()

    def show_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.show()

    def attempt_login(self) -> None:
        if self.lock_timer.isActive():
            return
        username = self.username_input.text().strip()
        password = self.password_input.text()
        if not username or not password:
            self.show_error("Nama pengguna dan kata sandi wajib diisi.")
            return

        try:
            profile = verify_operator_credentials(username, password)
        except AuthenticationError as exc:
            QMessageBox.critical(self, "Konfigurasi login bermasalah", str(exc))
            return

        if profile is None:
            self.failed_attempts += 1
            remaining = self.MAX_ATTEMPTS - self.failed_attempts
            self.password_input.clear()
            self.password_input.setFocus()
            if remaining <= 0:
                self.start_login_lock(30)
                return
            self.attempts_label.setText(f"Sisa percobaan login: {remaining} kali")
            self.show_error(
                f"Nama pengguna atau kata sandi tidak sesuai. Sisa percobaan: {remaining}."
            )
            return

        self.authenticated_user = profile
        self.accept()


class ChangePasswordDialog(QDialog):

    MAX_OLD_PASSWORD_ATTEMPTS = 3

    def __init__(self, current_user: Dict[str, str], parent=None):
        super().__init__(parent)
        self.current_user = current_user
        self.failed_attempts = 0

        self.setWindowTitle("Ubah Kata Sandi Operator")
        self.setObjectName("ChangePasswordDialog")
        self.setModal(True)
        self.setFixedWidth(560)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 22, 22, 22)
        root.setSpacing(0)

        card = QFrame()
        card.setObjectName("PasswordChangeCard")
        add_shadow(card, blur=24, y=6, alpha=22)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(28, 26, 28, 26)
        card_layout.setSpacing(11)

        title = QLabel("Ubah Kata Sandi")
        title.setObjectName("PasswordChangeTitle")
        card_layout.addWidget(title)

        subtitle = QLabel(
            "Masukkan kata sandi lama untuk memastikan perubahan dilakukan "
            "oleh operator yang sedang login."
        )
        subtitle.setObjectName("PasswordChangeSubtitle")
        subtitle.setWordWrap(True)
        card_layout.addWidget(subtitle)
        card_layout.addSpacing(4)

        account_box = QFrame()
        account_box.setObjectName("AccountIdentityBox")
        account_layout = QHBoxLayout(account_box)
        account_layout.setContentsMargins(14, 11, 14, 11)
        account_layout.setSpacing(12)

        account_caption = QLabel("Akun aktif")
        account_caption.setObjectName("AccountIdentityCaption")
        account_layout.addWidget(account_caption)

        account_layout.addStretch(1)
        account_name = str(
            self.current_user.get("display_name")
            or self.current_user.get("username")
            or "Operator"
        )
        account_value = QLabel(account_name)
        account_value.setTextFormat(Qt.PlainText)
        account_value.setToolTip(
            f"Nama pengguna login: {self.current_user.get('username', '')}"
        )
        account_value.setObjectName("AccountIdentityValue")
        account_value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        account_layout.addWidget(account_value)
        card_layout.addWidget(account_box)
        card_layout.addSpacing(4)

        old_label = QLabel("Kata Sandi Lama")
        old_label.setObjectName("PasswordFieldLabel")
        card_layout.addWidget(old_label)
        old_frame, self.old_password = self.create_password_input(
            "Masukkan kata sandi yang sedang digunakan"
        )
        card_layout.addWidget(old_frame)

        new_label = QLabel("Kata Sandi Baru")
        new_label.setObjectName("PasswordFieldLabel")
        card_layout.addWidget(new_label)
        new_frame, self.new_password = self.create_password_input(
            "Gunakan huruf besar, huruf kecil, angka, dan simbol"
        )
        card_layout.addWidget(new_frame)

        confirmation_label = QLabel("Konfirmasi Kata Sandi Baru")
        confirmation_label.setObjectName("PasswordFieldLabel")
        card_layout.addWidget(confirmation_label)
        confirmation_frame, self.password_confirmation = self.create_password_input(
            "Ulangi kata sandi baru"
        )
        card_layout.addWidget(confirmation_frame)

        self.validation_label = QLabel(password_policy_html(""))
        self.validation_label.setObjectName("PasswordValidationHint")
        self.validation_label.setWordWrap(True)
        card_layout.addWidget(self.validation_label)

        self.error_label = QLabel("")
        self.error_label.setObjectName("PasswordChangeError")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        card_layout.addWidget(self.error_label)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 6, 0, 0)
        button_row.setSpacing(10)
        button_row.addStretch(1)

        cancel_button = make_button("Batal", "secondary")
        cancel_button.clicked.connect(self.reject)
        button_row.addWidget(cancel_button)

        self.save_button = make_button("Simpan Kata Sandi", "primary")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_password)
        button_row.addWidget(self.save_button)
        card_layout.addLayout(button_row)

        root.addWidget(card)

        self.old_password.textChanged.connect(self.update_form_state)
        self.new_password.textChanged.connect(self.update_form_state)
        self.password_confirmation.textChanged.connect(self.update_form_state)
        self.password_confirmation.returnPressed.connect(self.save_password)
        QTimer.singleShot(0, self.old_password.setFocus)

    def create_password_input(self, placeholder: str) -> Tuple[QFrame, QLineEdit]:
        frame = QFrame()
        frame.setObjectName("PasswordChangeFieldFrame")
        frame.setMinimumHeight(48)

        field_layout = QHBoxLayout(frame)
        field_layout.setContentsMargins(0, 0, 7, 0)
        field_layout.setSpacing(0)

        field = QLineEdit()
        field.setObjectName("PasswordChangeField")
        field.setEchoMode(QLineEdit.Password)
        field.setPlaceholderText(placeholder)
        field.setMinimumHeight(46)
        field.setMaxLength(128)
        field_layout.addWidget(field, 1)

        toggle = QToolButton()
        toggle.setObjectName("PasswordChangeToggle")
        toggle.setIcon(make_password_eye_icon(hidden=True))
        toggle.setIconSize(QSize(26, 20))
        toggle.setCheckable(True)
        toggle.setCursor(Qt.PointingHandCursor)
        toggle.setFixedSize(40, 40)
        toggle.setToolTip("Tampilkan kata sandi")
        toggle.toggled.connect(
            lambda visible, target=field, button=toggle: self.toggle_password_field(
                target,
                button,
                visible,
            )
        )
        field_layout.addWidget(toggle)
        return frame, field

    @staticmethod
    def toggle_password_field(
        field: QLineEdit,
        button: QToolButton,
        visible: bool,
    ) -> None:
        field.setEchoMode(QLineEdit.Normal if visible else QLineEdit.Password)
        button.setIcon(make_password_eye_icon(hidden=not visible))
        button.setToolTip("Sembunyikan kata sandi" if visible else "Tampilkan kata sandi")
        field.setFocus()
        field.setCursorPosition(len(field.text()))

    def update_form_state(self) -> None:
        old_password = self.old_password.text()
        new_password = self.new_password.text()
        confirmation = self.password_confirmation.text()

        valid_policy = password_policy_valid(new_password)
        passwords_match = bool(confirmation) and new_password == confirmation
        password_changed = bool(new_password) and new_password != old_password
        ready = (
            bool(old_password)
            and valid_policy
            and passwords_match
            and password_changed
        )
        self.save_button.setEnabled(ready)

        details = [password_policy_html(new_password)]
        if old_password and new_password == old_password:
            details.append(
                "<span style='color:#B45309'><b>Kata sandi baru harus berbeda "
                "dari kata sandi lama.</b></span>"
            )
        elif confirmation and not passwords_match:
            details.append(
                "<span style='color:#B45309'><b>Konfirmasi kata sandi belum sama.</b></span>"
            )
        elif valid_policy and passwords_match:
            details.append(
                "<span style='color:#15803D'><b>Kata sandi baru memenuhi seluruh "
                "persyaratan dan konfirmasi sudah sesuai.</b></span>"
            )
        elif valid_policy:
            details.append(
                "<span style='color:#64748B'>Ulangi kata sandi baru pada kolom "
                "konfirmasi.</span>"
            )

        state = "valid" if ready else ("warning" if new_password else "neutral")
        self.validation_label.setText("<br><br>".join(details))
        self.validation_label.setProperty("state", state)
        self.validation_label.style().unpolish(self.validation_label)
        self.validation_label.style().polish(self.validation_label)
        self.error_label.hide()

    def show_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.show()

    def save_password(self) -> None:
        if not self.save_button.isEnabled():
            return

        username = str(self.current_user.get("username") or "").strip()
        old_password = self.old_password.text()
        new_password = self.new_password.text()
        confirmation = self.password_confirmation.text()

        if new_password != confirmation:
            self.show_error("Konfirmasi kata sandi baru tidak sama.")
            return
        if not password_policy_valid(new_password):
            self.show_error(password_policy_error_text(new_password))
            return
        if new_password == old_password:
            self.show_error("Kata sandi baru harus berbeda dari kata sandi lama.")
            return

        try:
            change_operator_password(
                username=username,
                old_password=old_password,
                new_password=new_password,
            )
        except AuthenticationError:
            self.failed_attempts += 1
            remaining = self.MAX_OLD_PASSWORD_ATTEMPTS - self.failed_attempts
            self.old_password.clear()
            self.old_password.setFocus()
            if remaining <= 0:
                QMessageBox.critical(
                    self,
                    "Perubahan dibatalkan",
                    "Kata sandi lama tidak sesuai sebanyak tiga kali. "
                    "Dialog akan ditutup tanpa mengubah akun.",
                )
                self.reject()
                return
            self.show_error(
                f"Kata sandi lama tidak sesuai. Sisa percobaan: {remaining}."
            )
            return
        except (ValueError, OSError) as exc:
            self.show_error(str(exc))
            return

        QMessageBox.information(
            self,
            "Kata sandi berhasil diubah",
            "Kata sandi operator telah diperbarui. Gunakan kata sandi baru "
            "pada login berikutnya.",
        )
        self.accept()


class SidebarWidget(QFrame):
    def __init__(self, main_window):
        super().__init__(); self.main_window=main_window; self.setObjectName("Sidebar"); self.setFixedWidth(220)
        layout=QVBoxLayout(self); layout.setContentsMargins(14,16,14,14); layout.setSpacing(7)
        brand=QFrame(); brand.setObjectName("BrandBox"); brand.setFixedHeight(96)
        brand_layout=QVBoxLayout(brand); brand_layout.setContentsMargins(8,4,4,6); brand_layout.setSpacing(2)
        for text, align in [("BANSOS", Qt.AlignLeft|Qt.AlignBottom), ("PAKOJAN", Qt.AlignLeft|Qt.AlignTop)]:
            label=QLabel(text); label.setObjectName("BrandTitle"); label.setAlignment(align); label.setFixedHeight(27); brand_layout.addWidget(label)
        caption=QLabel("PMT × Random Forest"); caption.setObjectName("LogoCaption"); caption.setFixedHeight(22); brand_layout.addWidget(caption)
        layout.addWidget(brand); layout.addSpacing(8)
        self.buttons=[]
        menus=[("⌂  Dashboard","Dashboard"),("⇩  Import Data","Import Data"),("≡  Data Warga","Data Warga"),("✓  Hasil Klasifikasi","Hasil Klasifikasi"),("✦  Prediksi Data Baru","Prediksi Data Baru"),("▥  Evaluasi","Evaluasi"),("▤  Laporan","Laporan"),("⚙  Tentang","Tentang")]
        for index,(text,_key) in enumerate(menus):
            button=QPushButton(text); button.setObjectName("NavButton"); button.setCursor(Qt.PointingHandCursor); button.setMinimumHeight(40)
            button.clicked.connect(lambda checked=False, idx=index: self.main_window.set_page(idx)); layout.addWidget(button); self.buttons.append(button)
        layout.addStretch(1)
        account_button=QPushButton("🔒  Ubah Kata Sandi")
        account_button.setObjectName("AccountButton")
        account_button.setCursor(Qt.PointingHandCursor)
        account_button.setMinimumHeight(38)
        account_button.clicked.connect(self.main_window.open_change_password)
        layout.addWidget(account_button)
        logout_button=QPushButton("⇥  Keluar Akun")
        logout_button.setObjectName("LogoutButton")
        logout_button.setCursor(Qt.PointingHandCursor)
        logout_button.clicked.connect(self.main_window.logout)
        layout.addWidget(logout_button)
        note=QLabel("Pendukung analisis\nKeputusan final: Kelurahan"); note.setObjectName("SidebarNote"); note.setWordWrap(True); layout.addWidget(note)
        self.update_user(main_window.current_user)
        self.set_active(0)

    def update_user(self, user: Optional[Dict[str, str]]) -> None:
        return

    def set_active(self,index):
        for i,button in enumerate(self.buttons):
            button.setProperty("active", i==index); button.style().unpolish(button); button.style().polish(button)


class MainWindow(QMainWindow):
    def __init__(self, current_user: Optional[Dict[str, str]] = None):
        super().__init__(); self.current_user=current_user or {"username":"operator", "display_name":"Operator", "role":"operator"}; self.setWindowTitle(APP_TITLE); self.resize(1366,768); self.setMinimumSize(1100,650)
        self.engine=DataEngine(); root=QWidget(); self.setCentralWidget(root); main=QHBoxLayout(root); main.setContentsMargins(0,0,0,0); main.setSpacing(0)
        self.sidebar=SidebarWidget(self); main.addWidget(self.sidebar)
        self.stack=QStackedWidget(); self.stack.setObjectName("ContentStack"); main.addWidget(self.stack,1)
        self.dashboard=DashboardPage(self.engine)
        self.import_page=ImportDataPage(self.engine,self)
        self.data_warga=DataWargaPage(self.engine)
        self.hasil=HasilKlasifikasiPage(self.engine)
        self.prediksi_baru=PrediksiDataBaruPage(self.engine)
        self.evaluasi=EvaluasiPage(self.engine)
        self.laporan=LaporanPage(self.engine)
        self.tentang=TentangPage(self.engine)
        self.pages=[self.dashboard,self.import_page,self.data_warga,self.hasil,self.prediksi_baru,self.evaluasi,self.laporan,self.tentang]
        for page in self.pages: self.stack.addWidget(page)
        self.status=QStatusBar(); self.setStatusBar(self.status)
        self.status.showMessage(("Model siap digunakan" if self.engine.model_ready else self.engine.model_error) + "  |  F11: layar penuh")
        self.set_page(0)

    def set_page(self,index):
        self.stack.setCurrentIndex(index); self.sidebar.set_active(index)
        name=self.sidebar.buttons[index].text().split("  ",1)[-1]; self.status.showMessage(f"Halaman aktif: {name}")

    def open_change_password(self):
        dialog = ChangePasswordDialog(self.current_user, self)
        if dialog.exec_() == QDialog.Accepted:
            self.status.showMessage(
                "Kata sandi operator berhasil diperbarui",
                6000,
            )

    def logout(self):
        answer = QMessageBox.question(
            self,
            "Keluar akun",
            "Apakah Anda yakin ingin keluar dari akun operator?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        self.hide()
        login = LoginDialog()
        if login.exec_() == QDialog.Accepted and login.authenticated_user:
            self.current_user = login.authenticated_user
            self.sidebar.update_user(self.current_user)
            self.showMaximized()
            self.status.showMessage(
                f"Login berhasil sebagai {self.current_user.get('display_name', 'Operator Kelurahan')}"
            )
        else:
            self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
            return
        if event.key() == Qt.Key_Escape and self.isFullScreen():
            self.showNormal()
            return
        super().keyPressEvent(event)

    def refresh_all_pages(self,message="Data berhasil diperbarui"):
        for page in self.pages:
            if hasattr(page,"refresh"): page.refresh()
        self.status.showMessage(message)


STYLE = f"""
* {{ font-family: 'Segoe UI', 'Inter', 'Arial'; font-size: 9.5pt; color: {TEXT}; }}
QMainWindow {{ background: {BG}; }}
#LoginDialog {{ background:#F5F9FE; }}
#LoginContent {{ background:#F5F9FE; }}
#LoginIllustration {{ background:#EAF4FF; border:none; }}
#LoginFormPanel {{ background:#F5F9FE; }}
#LoginCard {{ background:white; border:1px solid #E3ECF6; border-radius:22px; }}
#LoginTitle {{ color:#101828; font-size:22pt; font-weight:900; }}
#LoginSubtitle {{ color:#475467; font-size:10.5pt; font-weight:500; }}
#LoginFieldLabel {{ color:#101828; font-size:9.5pt; font-weight:750; padding-top:2px; }}
#LoginField {{ background:white; border:1px solid #C7D3E1; border-radius:11px; padding:0 18px; font-size:11pt; color:#1D2939; min-height:58px; }}
#LoginField:focus {{ border:2px solid #2C6EC8; }}
#LoginPasswordFrame {{ background:white; border:1px solid #C7D3E1; border-radius:11px; min-height:58px; }}
#LoginPasswordFrame:focus-within {{ border:2px solid #2C6EC8; }}
#LoginPasswordField {{ background:transparent; border:none; border-radius:0; padding:0 16px; font-size:10.5pt; color:#1D2939; }}
#PasswordToggle {{ background:transparent; border:none; border-radius:9px; padding:4px; }}
#PasswordToggle:hover {{ background:#EAF3FF; }}
#LoginPrimaryButton {{ color:white; background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #1F64B9,stop:1 #2878D4); border:none; border-radius:10px; padding:11px 18px; font-size:11pt; font-weight:850; }}
#LoginPrimaryButton:hover {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #18539A,stop:1 #2168BA); }}
#LoginPrimaryButton:pressed {{ background:#174D8E; }}
#LoginSecurityNote {{ color:#183B64; background:#EFF7FF; border:1px solid #C7E0FA; border-radius:10px; padding:12px 14px; font-size:9pt; font-weight:650; }}
#LoginExitButton {{ color:#667085; background:transparent; border:none; padding:5px 10px; font-size:8.8pt; font-weight:650; }}
#LoginExitButton:hover {{ color:#B42318; text-decoration:underline; }}
#LoginError {{ color:#991B1B; background:#FEE2E2; border:1px solid #FCA5A5; border-radius:10px; padding:8px; font-weight:700; }}
#LoginFooter {{ background:rgba(255,255,255,0.90); border-top:1px solid #D8E5F2; }}
#LoginFooterStrong {{ color:#123B67; font-size:9pt; font-weight:800; }}
#LoginFooterText {{ color:#475467; font-size:8.7pt; font-weight:600; }}
#Sidebar {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {NAVY},stop:1 #061D35); border-right:1px solid #08243F; }}
#BrandBox {{ background:transparent; }}
#BrandTitle {{ color:white; font-size:16pt; font-weight:900; letter-spacing:1px; }}
#LogoCaption {{ color:#BFDBFE; font-size:8.5pt; font-weight:700; }}
#SidebarNote {{ color:#D1E8FF; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.12); border-radius:12px; padding:9px; font-size:8.5pt; }}
#SidebarUser {{ color:white; background:rgba(255,255,255,0.10); border:1px solid rgba(255,255,255,0.14); border-radius:12px; padding:9px; font-size:8.2pt; font-weight:700; }}
#LogoutButton {{ text-align:left; padding:8px 12px; border:1px solid rgba(255,255,255,0.20); border-radius:10px; color:#FEE2E2; background:rgba(220,38,38,0.18); font-weight:750; }}
#LogoutButton:hover {{ background:rgba(220,38,38,0.32); color:white; }}
#AccountButton {{ text-align:left; padding:8px 12px; border:1px solid rgba(191,219,254,0.38); border-radius:10px; color:#DBEAFE; background:rgba(37,99,235,0.18); font-weight:750; }}
#AccountButton:hover {{ background:rgba(37,99,235,0.34); color:white; }}
#ChangePasswordDialog {{ background:#F5F7FA; }}
#PasswordChangeCard {{ background:white; border:1px solid {BORDER}; border-radius:18px; }}
#PasswordChangeTitle {{ color:{TEXT}; font-size:18pt; font-weight:900; }}
#PasswordChangeSubtitle {{ color:{MUTED}; font-size:9.5pt; }}
#AccountIdentityBox {{ background:#EFF6FF; border:1px solid #BFDBFE; border-radius:11px; }}
#AccountIdentityCaption {{ color:{MUTED}; font-size:9pt; font-weight:750; }}
#AccountIdentityValue {{ color:{NAVY}; font-size:9.5pt; font-weight:850; }}
#PasswordFieldLabel {{ color:{TEXT}; font-weight:750; padding-top:3px; }}
#PasswordChangeFieldFrame {{
    background:white;
    border:1px solid {BORDER_DARK};
    border-radius:10px;
    min-height:46px;
}}
#PasswordChangeField {{
    background:transparent;
    border:none;
    border-radius:0;
    min-height:30px;
    padding:7px 11px;
}}
#PasswordChangeField:focus {{ border:none; }}
#PasswordChangeToggle {{
    background:transparent;
    border:none;
    border-radius:8px;
    padding:4px;
}}
#PasswordChangeToggle:hover {{ background:#EAF3FF; }}
#PasswordChangeToggle:pressed {{ background:#DBEAFE; }}
#PasswordValidationHint {{ color:{MUTED}; background:#F8FAFC; border:1px solid {BORDER}; border-radius:9px; padding:8px 10px; font-size:8.8pt; }}
#PasswordValidationHint[state="warning"] {{ color:#92400E; background:#FFFBEB; border:1px solid #FDE68A; }}
#PasswordValidationHint[state="valid"] {{ color:#166534; background:#F0FDF4; border:1px solid #BBF7D0; }}
#PasswordChangeError {{ color:#991B1B; background:#FEE2E2; border:1px solid #FCA5A5; border-radius:9px; padding:8px 10px; font-weight:700; }}
#NavButton {{ text-align:left; padding-left:14px; border:none; border-radius:11px; color:#DCEBFF; font-size:9.5pt; font-weight:650; background:transparent; }}
#NavButton:hover {{ background:rgba(255,255,255,0.10); color:white; }}
#NavButton[active="true"] {{ background:white; color:{NAVY}; }}
#ContentStack, #PageScroll {{ background:{BG}; border:none; }}
#PageTitle {{ font-size:21pt; font-weight:900; color:{TEXT}; }}
#PageSubtitle {{ font-size:9.5pt; color:{MUTED}; }}
#Card, #StatCard {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:16px; }}
#CardTitle {{ font-size:12.5pt; font-weight:800; color:{TEXT}; }}
#SmallText, #StatSubtitle {{ color:{MUTED}; font-size:8.7pt; }}
#StatTitle {{ color:{MUTED}; font-size:8.7pt; font-weight:700; }}
#StatValue {{ font-size:18pt; font-weight:900; }}
#InfoLabel {{ background:{SURFACE_2}; border:1px solid {BORDER}; border-radius:13px; padding:12px; color:{TEXT}; }}
#RequiredLabel {{ color:{TEXT}; font-weight:750; }}
QLineEdit, QComboBox, QDateEdit {{ background:white; border:1px solid {BORDER_DARK}; border-radius:10px; padding:7px 9px; min-height:20px; }}
QLineEdit:focus, QComboBox:focus, QDateEdit:focus {{ border:1px solid {PRIMARY}; }}
QLineEdit[readOnly="true"] {{ background:#F1F5F9; color:{MUTED}; }}
QComboBox::drop-down, QDateEdit::drop-down {{ border:none; width:34px; }}
QComboBox QAbstractItemView {{ background:white; border:1px solid {BORDER}; selection-background-color:#DBEAFE; selection-color:{TEXT}; padding:6px; }}
#PrimaryButton {{ background:{PRIMARY}; color:white; border:none; border-radius:12px; padding:10px 18px; font-weight:800; }}
#PrimaryButton:hover {{ background:{PRIMARY_DARK}; }}
#SecondaryButton {{ background:white; color:{NAVY}; border:1px solid {BORDER_DARK}; border-radius:12px; padding:10px 18px; font-weight:800; }}
#SecondaryButton:hover {{ background:#EFF6FF; }}
#DangerButton {{ background:{RED}; color:white; border:none; border-radius:12px; padding:10px 18px; font-weight:800; }}
QTableWidget {{ background:white; alternate-background-color:#F8FAFC; border:1px solid {BORDER}; border-radius:14px; gridline-color:#EEF2F7; selection-background-color:#DBEAFE; selection-color:{TEXT}; }}
QHeaderView::section {{ background:#EFF6FF; color:{NAVY}; padding:7px; border:none; border-right:1px solid {BORDER}; font-weight:800; }}
QProgressBar {{ border:1px solid {BORDER}; border-radius:10px; background:#E5E7EB; text-align:center; min-height:24px; }}
QProgressBar::chunk {{ background:{PRIMARY}; border-radius:10px; }}
#ReadOnlyText, #ReportPreview {{ background:{SURFACE_2}; border:1px solid {BORDER}; border-radius:14px; padding:14px; color:{TEXT}; }}
#DialogTitle {{ font-size:18pt; font-weight:900; color:{TEXT}; }}
QTabWidget::pane {{ border:none; background:transparent; }}
QTabBar::tab {{ background:white; color:{NAVY}; border:1px solid {BORDER}; border-radius:10px; min-width:120px; min-height:20px; padding:7px 12px; margin-right:6px; font-weight:800; }}
QTabBar::tab:selected {{ background:{PRIMARY}; color:white; border:1px solid {PRIMARY}; }}
#DetailSummary {{ background:#EFF6FF; border:1px solid #BFDBFE; border-radius:14px; }}
#DetailSummaryTitle {{ color:{MUTED}; font-size:9.5pt; font-weight:700; }}
#DetailSummaryValue {{ color:{NAVY}; font-size:11pt; font-weight:850; }}
#DetailFieldLabel {{ color:{MUTED}; font-weight:750; padding:5px 0; }}
#DetailFieldValue {{ background:white; border:1px solid {BORDER}; border-radius:10px; padding:9px 12px; color:{TEXT}; }}
QCalendarWidget {{ background:white; border:1px solid {BORDER}; border-radius:16px; }}
QCalendarWidget QWidget#qt_calendar_navigationbar {{ background:{NAVY}; border-top-left-radius:14px; border-top-right-radius:14px; min-height:44px; }}
QCalendarWidget QToolButton {{ color:white; background:transparent; border:none; border-radius:8px; padding:7px 10px; font-weight:800; }}
QCalendarWidget QToolButton:hover {{ background:rgba(255,255,255,0.16); }}
QCalendarWidget QMenu {{ background:white; color:{TEXT}; }}
QCalendarWidget QSpinBox {{ background:white; color:{TEXT}; border-radius:8px; padding:5px; }}
QCalendarWidget QAbstractItemView:enabled {{ background:white; color:{TEXT}; selection-background-color:{PRIMARY}; selection-color:white; outline:0; }}
QCalendarWidget QAbstractItemView:disabled {{ color:#CBD5E1; }}
#PredictionChoiceSection {{
    background:{SURFACE};
    border:1px solid {BORDER};
    border-radius:16px;
}}
#PredictionSectionTitle {{
    color:{TEXT};
    font-size:15pt;
    font-weight:900;
}}
#PredictionOptionCard {{
    background:#F8FAFC;
    border:1px solid {BORDER_DARK};
    border-radius:16px;
}}
#PredictionOptionCard:hover {{
    background:#F5F9FF;
    border:1px solid #93C5FD;
}}
#PredictionOptionBadge {{
    background:#DBEAFE;
    color:{PRIMARY_DARK};
    border:1px solid #BFDBFE;
    border-radius:12px;
    font-size:11pt;
    font-weight:900;
}}
#PredictionOptionType {{
    color:{MUTED};
    font-size:8pt;
    font-weight:850;
    letter-spacing:1px;
}}
#PredictionOptionTitle {{
    color:{NAVY};
    font-size:14pt;
    font-weight:900;
    padding-top:4px;
}}
#PredictionOptionDescription {{
    color:#475569;
    font-size:9.5pt;
    line-height:1.4;
}}
#PredictionOptionNote {{
    color:#166534;
    background:#F0FDF4;
    border:1px solid #BBF7D0;
    border-radius:9px;
    padding:7px 10px;
    font-size:8.8pt;
    font-weight:700;
}}
#PredictionOptionButton {{
    background:{PRIMARY};
    color:white;
    border:none;
    border-radius:11px;
    padding:10px 16px;
    font-weight:850;
    text-align:center;
}}
#PredictionOptionButton:hover {{
    background:{PRIMARY_DARK};
}}
#PredictionFocusBar {{
    background:{SURFACE};
    border:1px solid {BORDER};
    border-radius:14px;
}}
#PredictionBackTextButton {{
    background:#F8FAFC;
    color:{PRIMARY_DARK};
    border:1px solid #BFDBFE;
    border-radius:10px;
    padding:8px 16px;
    font-weight:800;
}}
#PredictionBackTextButton:hover {{
    background:#EFF6FF;
    border:1px solid #93C5FD;
}}
#PredictionBackTextButton:pressed {{
    background:#DBEAFE;
}}
#PredictionFocusTitle {{
    color:{NAVY};
    font-size:13.5pt;
    font-weight:900;
}}
#PredictionFocusSubtitle {{
    color:{MUTED};
    font-size:8.8pt;
}}
#PredictionModeStack {{
    background:transparent;
    border:none;
}}
QStatusBar {{ background:white; color:{MUTED}; border-top:1px solid {BORDER}; }}
"""


def create_splash():
    pix=QPixmap(620,320); pix.fill(QColor(NAVY)); painter=QPainter(pix); painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QColor("white")); painter.setFont(QFont("Segoe UI",25,QFont.Bold)); painter.drawText(42,88,"BANSOS PAKOJAN")
    painter.setFont(QFont("Segoe UI",13,QFont.Medium)); painter.setPen(QColor("#BFDBFE")); painter.drawText(44,125,"Komparasi Proxy Means Test dan Random Forest")
    painter.setPen(QColor("#D6A742")); painter.setFont(QFont("Segoe UI",11,QFont.Bold)); painter.drawText(44,190,"Memuat model...")
    painter.end(); return QSplashScreen(pix)


def main():
    app=QApplication(sys.argv); app.setStyleSheet(STYLE); app.setFont(QFont("Segoe UI",10))
    login=LoginDialog()
    if login.exec_() != QDialog.Accepted or not login.authenticated_user:
        sys.exit(0)
    splash=create_splash(); splash.show(); app.processEvents()
    window=MainWindow(login.authenticated_user); QTimer.singleShot(700,splash.close); QTimer.singleShot(730,window.showMaximized)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
