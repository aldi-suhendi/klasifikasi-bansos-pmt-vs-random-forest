from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib


def load_joblib_model(path: str | Path) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
        )
        return joblib.load(Path(path))
