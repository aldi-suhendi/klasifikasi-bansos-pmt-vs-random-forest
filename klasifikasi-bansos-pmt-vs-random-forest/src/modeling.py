from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PMT_PARAM_GRID,
    RANDOM_STATE,
    RF_PARAM_GRID,
)


@dataclass(frozen=True)
class TrainedModel:

    pipeline: Pipeline
    best_params: dict[str, Any]
    cv_f1: float
    threshold: float
    threshold_curve: pd.DataFrame


def create_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def create_pmt_search(cv: StratifiedKFold, n_jobs: int) -> GridSearchCV:
    pipeline = Pipeline(
        steps=[
            ("preprocess", create_preprocessor(scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid=PMT_PARAM_GRID,
        scoring="f1",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
    )


def create_random_forest_search(
    cv: StratifiedKFold,
    n_jobs: int,
) -> GridSearchCV:
    pipeline = Pipeline(
        steps=[
            ("preprocess", create_preprocessor(scale_numeric=False)),
            (
                "model",
                RandomForestClassifier(
                    criterion="gini",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid=RF_PARAM_GRID,
        scoring="f1",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
    )


def build_threshold_curve(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    y_array = np.asarray(y_true, dtype=int)

    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(
                    precision_score(y_array, predictions, zero_division=0)
                ),
                "recall": float(recall_score(y_array, predictions, zero_division=0)),
                "f1": float(f1_score(y_array, predictions, zero_division=0)),
            }
        )

    return pd.DataFrame(rows)


def select_best_threshold(curve: pd.DataFrame) -> float:
    ranked = curve.assign(distance_to_half=(curve["threshold"] - 0.50).abs())
    ranked = ranked.sort_values(
        by=["f1", "recall", "distance_to_half"],
        ascending=[False, False, True],
    )
    return float(ranked.iloc[0]["threshold"])


def train_optimized_model(
    search: GridSearchCV,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    n_jobs: int,
) -> TrainedModel:
    search.fit(X_train, y_train)
    best_pipeline: Pipeline = search.best_estimator_

    out_of_fold_probabilities = cross_val_predict(
        best_pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=n_jobs,
    )[:, 1]

    threshold_curve = build_threshold_curve(y_train, out_of_fold_probabilities)
    threshold = select_best_threshold(threshold_curve)

    return TrainedModel(
        pipeline=best_pipeline,
        best_params={
            key.replace("model__", ""): value
            for key, value in search.best_params_.items()
        },
        cv_f1=float(search.best_score_),
        threshold=threshold,
        threshold_curve=threshold_curve,
    )


def predict_with_threshold(
    pipeline: Pipeline,
    X: pd.DataFrame,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = pipeline.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities
