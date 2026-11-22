from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
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


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    inclusion_error: float
    exclusion_error: float
    roc_auc: float
    average_precision: float
    tn: int
    fp: int
    fn: int
    tp: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def evaluate_predictions(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_prob: Iterable[float],
) -> EvaluationResult:
    actual = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    probability = np.asarray(y_prob, dtype=float)

    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    inclusion_error = fp / (tp + fp) if (tp + fp) else 0.0
    exclusion_error = fn / (tp + fn) if (tp + fn) else 0.0

    return EvaluationResult(
        accuracy=float(accuracy_score(actual, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(actual, predicted)),
        precision=float(precision_score(actual, predicted, zero_division=0)),
        recall=float(recall_score(actual, predicted, zero_division=0)),
        f1=float(f1_score(actual, predicted, zero_division=0)),
        inclusion_error=float(inclusion_error),
        exclusion_error=float(exclusion_error),
        roc_auc=float(roc_auc_score(actual, probability)),
        average_precision=float(average_precision_score(actual, probability)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def exact_mcnemar(
    y_true: Iterable[int],
    pmt_prediction: Iterable[int],
    rf_prediction: Iterable[int],
) -> dict[str, int | float]:
    actual = np.asarray(y_true, dtype=int)
    pmt = np.asarray(pmt_prediction, dtype=int)
    rf = np.asarray(rf_prediction, dtype=int)

    pmt_correct = pmt == actual
    rf_correct = rf == actual

    both_correct = int(np.sum(pmt_correct & rf_correct))
    pmt_only = int(np.sum(pmt_correct & ~rf_correct))
    rf_only = int(np.sum(~pmt_correct & rf_correct))
    both_wrong = int(np.sum(~pmt_correct & ~rf_correct))
    discordant = pmt_only + rf_only

    p_value = 1.0
    if discordant:
        p_value = float(
            binomtest(
                min(pmt_only, rf_only),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    return {
        "both_correct": both_correct,
        "pmt_correct_rf_wrong": pmt_only,
        "pmt_wrong_rf_correct": rf_only,
        "both_wrong": both_wrong,
        "discordant": discordant,
        "p_value": p_value,
    }
