from __future__ import annotations

import os
import platform
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import Final

EXPECTED_PYTHON_MAJOR_MINOR: Final = (3, 13)
EXPECTED_PACKAGES: Final[dict[str, str]] = {
    "pandas": "3.0.3",
    "numpy": "2.5.1",
    "scikit-learn": "1.9.0",
    "scipy": "1.18.0",
    "joblib": "1.5.3",
    "matplotlib": "3.11.1",
    "openpyxl": "3.1.5",
    "xlrd": "2.0.2",
}
EXPECTED_GUI_PACKAGES: Final[dict[str, str]] = {
    "PyQt5": "5.15.11",
}


class EnvironmentMismatchError(RuntimeError):
    pass


def collect_environment(include_gui: bool = False) -> dict[str, str]:
    packages = dict(EXPECTED_PACKAGES)
    if include_gui:
        packages.update(EXPECTED_GUI_PACKAGES)

    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package_name in packages:
        try:
            result[package_name] = version(package_name)
        except PackageNotFoundError:
            result[package_name] = "TIDAK TERPASANG"
    return result


def validate_environment(include_gui: bool = False) -> dict[str, str]:
    errors: list[str] = []
    actual_python = sys.version_info[:2]
    if actual_python != EXPECTED_PYTHON_MAJOR_MINOR:
        errors.append(
            "Python harus 3.13.x, tetapi yang aktif adalah "
            f"{platform.python_version()}."
        )

    expected = dict(EXPECTED_PACKAGES)
    if include_gui:
        expected.update(EXPECTED_GUI_PACKAGES)

    actual = collect_environment(include_gui=include_gui)
    for package_name, expected_version in expected.items():
        actual_version = actual.get(package_name, "TIDAK TERPASANG")
        if actual_version != expected_version:
            errors.append(
                f"{package_name} harus {expected_version}, tetapi yang aktif "
                f"adalah {actual_version}."
            )

    if errors and os.environ.get("APK_RELAXED_ENV") == "1":
        warnings.warn(
            "APK_RELAXED_ENV=1: versi lingkungan tidak sama dengan versi terkunci. "
            + " ".join(errors),
            RuntimeWarning,
            stacklevel=2,
        )
        errors = []

    if errors:
        detail = "\n- ".join(errors)
        raise EnvironmentMismatchError(
            "Lingkungan Python tidak sama dengan lingkungan final penelitian.\n"
            f"- {detail}\n\n"
            "Aktifkan .venv proyek dan jalankan:\n"
            "python -m pip install -r requirements.txt"
        )
    return actual
