"""Cálculo de métricas globales (AUC-PR, F1, ROC-AUC, Recall@FPR) y equidad por subgrupos.

Implementa las métricas predeclaradas en el Hito 1 y la auditoría demográfica del Criterio 7
sobre `customer_age`, `employment_status` e `income` siguiendo Jesus et al. (NeurIPS 2022).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from src.config import OPERATIONAL_MAX_FPR


def threshold_at_max_fpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_fpr: float = OPERATIONAL_MAX_FPR,
) -> tuple[float, float, float]:
    """Obtiene el umbral operativo que maximiza Recall sujeto a FPR <= max_fpr."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid_idx = np.where(fpr <= max_fpr)[0]
    if len(valid_idx) == 0:
        return 0.5, 0.0, 0.0
    best_idx = valid_idx[np.argmax(tpr[valid_idx])]
    chosen_threshold = float(thresholds[best_idx])
    if np.isinf(chosen_threshold):
        chosen_threshold = 1.0
    return chosen_threshold, float(tpr[best_idx]), float(fpr[best_idx])


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Calcula AUC-PR (ap), F1-score, ROC-AUC, Recall@FPR<=5% y matriz de confusión."""
    y_arr = np.asarray(y_true, dtype=np.int8)
    prob_arr = np.asarray(y_prob, dtype=np.float64)
    y_pred = (prob_arr >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(y_arr, y_pred, labels=[0, 1]).ravel()
    fpr_val = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    recall_val = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    _, recall_at_fpr5, actual_fpr5 = threshold_at_max_fpr(y_arr, prob_arr, OPERATIONAL_MAX_FPR)

    return {
        "ap": float(average_precision_score(y_arr, prob_arr)),
        "f1": float(f1_score(y_arr, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_arr, prob_arr)) if len(np.unique(y_arr)) > 1 else 0.0,
        "recall": recall_val,
        "fpr": fpr_val,
        "recall_at_fpr5": recall_at_fpr5,
        "actual_fpr_at_fpr5": actual_fpr5,
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def compute_subgroup_fairness(
    X_eval: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Desglosa desempeño y tasas de error por edad (>=50 vs <50), empleo y cuantiles de ingreso."""
    y_arr = np.asarray(y_true, dtype=np.int8)
    prob_arr = np.asarray(y_prob, dtype=np.float64)
    y_pred = (prob_arr >= threshold).astype(np.int8)

    subgroup_definitions: dict[str, pd.Series] = {
        "age_group": pd.Series(
            np.where(X_eval["customer_age"].to_numpy() >= 50, "age_>=50", "age_<50"),
            index=X_eval.index,
        ),
        "employment_status": X_eval["employment_status"].astype(str),
        "income_band": pd.cut(
            X_eval["income"].astype(float),
            bins=[-np.inf, 0.35, 0.65, np.inf],
            labels=["low_income", "mid_income", "high_income"],
        ).astype(str),
    }

    report: dict[str, Any] = {}
    for attr_name, group_series in subgroup_definitions.items():
        groups_summary: dict[str, dict[str, float | int]] = {}
        fpr_values: list[float] = []
        recall_values: list[float] = []

        for group_val in sorted(group_series.unique()):
            mask = (group_series == group_val).to_numpy()
            if mask.sum() < 20:
                continue
            yt_g = y_arr[mask]
            yp_g = y_pred[mask]
            pr_g = prob_arr[mask]

            tn, fp, fn, tp = confusion_matrix(yt_g, yp_g, labels=[0, 1]).ravel()
            fpr_g = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            rec_g = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            ap_g = float(average_precision_score(yt_g, pr_g)) if yt_g.sum() > 0 else 0.0

            fpr_values.append(fpr_g)
            if yt_g.sum() > 0:
                recall_values.append(rec_g)

            groups_summary[str(group_val)] = {
                "count": int(mask.sum()),
                "positives": int(yt_g.sum()),
                "ap": round(ap_g, 4),
                "recall": round(rec_g, 4),
                "fpr": round(fpr_g, 4),
            }

        report[attr_name] = {
            "groups": groups_summary,
            "max_fpr_gap": round(max(fpr_values) - min(fpr_values), 4) if len(fpr_values) >= 2 else 0.0,
            "max_recall_gap": (
                round(max(recall_values) - min(recall_values), 4) if len(recall_values) >= 2 else 0.0
            ),
        }

    return report
