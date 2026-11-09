"""Cálculo de métricas globales (AUC-PR, F1, ROC-AUC, Recall@FPR) y equidad por subgrupos.

Implementa las métricas predeclaradas en el Hito 1 y la auditoría demográfica del Criterio 7
sobre `customer_age`, `employment_status` e `income` siguiendo Jesus et al. (NeurIPS 2022),
incluyendo mitigación post-procesamiento de sesgo etario mediante calibración de umbrales por
subgrupo exclusivamente sobre el conjunto de validación (`Val`).
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
    calibrated_on_val: bool = False,
) -> dict[str, Any]:
    """Calcula AUC-PR (ap), F1-score, ROC-AUC, Recall@FPR<=5% y matriz de confusión.

    Si `calibrated_on_val=True`, `recall_at_fpr5` y `actual_fpr_at_fpr5` reportan el desempeño
    estrictamente libre de fuga usando el umbral `threshold` calibrado en validación (`Val`),
    conservando aparte `roc_curve_recall_at_fpr5` solo como referencia descriptiva de la curva ROC.
    """
    y_arr = np.asarray(y_true, dtype=np.int8)
    prob_arr = np.asarray(y_prob, dtype=np.float64)
    y_pred = (prob_arr >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(y_arr, y_pred, labels=[0, 1]).ravel()
    fpr_val = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    recall_val = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision_val = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    _, roc_recall_at_fpr5, roc_actual_fpr5 = threshold_at_max_fpr(
        y_arr, prob_arr, OPERATIONAL_MAX_FPR
    )
    effective_recall_fpr5 = recall_val if calibrated_on_val else roc_recall_at_fpr5
    effective_actual_fpr5 = fpr_val if calibrated_on_val else roc_actual_fpr5

    return {
        "ap": float(average_precision_score(y_arr, prob_arr)),
        "f1": float(f1_score(y_arr, y_pred, zero_division=0)),
        "precision": precision_val,
        "roc_auc": float(roc_auc_score(y_arr, prob_arr)) if len(np.unique(y_arr)) > 1 else 0.0,
        "recall": recall_val,
        "fpr": fpr_val,
        "recall_at_fpr5": effective_recall_fpr5,
        "actual_fpr_at_fpr5": effective_actual_fpr5,
        "roc_curve_recall_at_fpr5": roc_recall_at_fpr5,
        "roc_curve_fpr_at_fpr5": roc_actual_fpr5,
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def calibrate_age_group_thresholds_val(
    X_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray,
    val_probs: np.ndarray,
    max_fpr: float = OPERATIONAL_MAX_FPR,
    max_val_fpr_gap: float = 0.03,
) -> dict[str, Any]:
    """Calibra umbrales diferenciados por grupo etario (<50 vs >=50) EXCLUSIVAMENTE en Val.

    Resuelve en validación el problema de optimización con restricciones (Hardt et al., 2016):
    maximizar Recall_val global sujeto a FPR_val_global <= max_fpr y |FPR_val(<50) - FPR_val(>=50)| <= max_val_fpr_gap,
    sin observar jamás las etiquetas ni predicciones de Test.
    """
    y_arr = np.asarray(y_val, dtype=np.int8)
    prob_arr = np.asarray(val_probs, dtype=np.float64)
    is_ge50 = X_val["customer_age"].to_numpy() >= 50
    mask_lt50 = ~is_ge50
    mask_ge50 = is_ge50

    n_neg_total = int((y_arr == 0).sum())
    n_pos_total = int((y_arr == 1).sum())

    # Precalcular candidatos en Val para cada subgrupo
    grid_lt50 = np.linspace(max(0.01, max_fpr * 0.70), max_fpr * 1.02, 17)
    grid_ge50 = np.linspace(max(0.01, max_fpr * 0.70), min(0.09, max_fpr * 1.50), 21)

    cand_lt50 = [threshold_at_max_fpr(y_arr[mask_lt50], prob_arr[mask_lt50], float(b)) for b in grid_lt50]
    cand_ge50 = [threshold_at_max_fpr(y_arr[mask_ge50], prob_arr[mask_ge50], float(b)) for b in grid_ge50]

    best_score: tuple[float, float, float] | None = None
    best_pair: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None

    y_lt, p_lt = y_arr[mask_lt50], prob_arr[mask_lt50]
    y_ge, p_ge = y_arr[mask_ge50], prob_arr[mask_ge50]
    neg_lt = int((y_lt == 0).sum())
    pos_lt = int((y_lt == 1).sum())
    neg_ge = int((y_ge == 0).sum())
    pos_ge = int((y_ge == 1).sum())

    for c_lt in cand_lt50:
        tau_lt, rec_lt, fpr_lt = c_lt
        fp_lt = fpr_lt * neg_lt
        tp_lt = rec_lt * pos_lt
        for c_ge in cand_ge50:
            tau_ge, rec_ge, fpr_ge = c_ge
            fp_ge = fpr_ge * neg_ge
            tp_ge = rec_ge * pos_ge

            global_val_fpr = (fp_lt + fp_ge) / max(1, n_neg_total)
            val_gap = abs(fpr_lt - fpr_ge)
            if global_val_fpr <= max_fpr and val_gap <= max_val_fpr_gap:
                global_val_rec = (tp_lt + tp_ge) / max(1, n_pos_total)
                score = (round(global_val_rec, 6), -round(val_gap, 6), -round(global_val_fpr, 6))
                if best_score is None or score > best_score:
                    best_score = score
                    best_pair = (c_lt, c_ge)

    if best_pair is None:
        c_lt = threshold_at_max_fpr(y_lt, p_lt, max_fpr=max_fpr)
        c_ge = threshold_at_max_fpr(y_ge, p_ge, max_fpr=max_fpr)
    else:
        c_lt, c_ge = best_pair

    thresholds: dict[str, float] = {
        "age_<50": float(c_lt[0]),
        "age_>=50": float(c_ge[0]),
    }
    val_groups: dict[str, dict[str, float | int]] = {
        "age_<50": {
            "count": int(mask_lt50.sum()),
            "positives": pos_lt,
            "calibrated_threshold_tau": round(float(c_lt[0]), 6),
            "val_recall": round(float(c_lt[1]), 4),
            "val_fpr": round(float(c_lt[2]), 4),
        },
        "age_>=50": {
            "count": int(mask_ge50.sum()),
            "positives": pos_ge,
            "calibrated_threshold_tau": round(float(c_ge[0]), 6),
            "val_recall": round(float(c_ge[1]), 4),
            "val_fpr": round(float(c_ge[2]), 4),
        },
    }

    return {
        "max_fpr_constraint": float(max_fpr),
        "max_val_fpr_gap_constraint": float(max_val_fpr_gap),
        "thresholds": thresholds,
        "val_groups": val_groups,
    }


def predict_with_age_group_thresholds(
    X_eval: pd.DataFrame,
    y_prob: np.ndarray,
    group_thresholds: dict[str, float],
) -> np.ndarray:
    """Genera predicciones binarias aplicando el umbral calibrado en Val según grupo etario."""
    prob_arr = np.asarray(y_prob, dtype=np.float64)
    is_ge50 = X_eval["customer_age"].to_numpy() >= 50
    tau_vec = np.where(
        is_ge50,
        float(group_thresholds["age_>=50"]),
        float(group_thresholds["age_<50"]),
    )
    return (prob_arr >= tau_vec).astype(np.int8)


def compute_metrics_with_custom_predictions(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    thresholds_metadata: dict[str, float],
) -> dict[str, Any]:
    """Calcula métricas globales en Test para predicciones con umbrales por subgrupo calibrados en Val."""
    y_arr = np.asarray(y_true, dtype=np.int8)
    prob_arr = np.asarray(y_prob, dtype=np.float64)
    pred_arr = np.asarray(y_pred, dtype=np.int8)

    tn, fp, fn, tp = confusion_matrix(y_arr, pred_arr, labels=[0, 1]).ravel()
    fpr_val = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    recall_val = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision_val = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    _, roc_recall_at_fpr5, roc_actual_fpr5 = threshold_at_max_fpr(
        y_arr, prob_arr, OPERATIONAL_MAX_FPR
    )
    return {
        "ap": float(average_precision_score(y_arr, prob_arr)),
        "f1": float(f1_score(y_arr, pred_arr, zero_division=0)),
        "precision": precision_val,
        "roc_auc": float(roc_auc_score(y_arr, prob_arr)) if len(np.unique(y_arr)) > 1 else 0.0,
        "fpr": fpr_val,
        "recall": recall_val,
        "recall_at_fpr5": recall_val,
        "actual_fpr_at_fpr5": fpr_val,
        "roc_curve_recall_at_fpr5": roc_recall_at_fpr5,
        "roc_curve_fpr_at_fpr5": roc_actual_fpr5,
        "group_thresholds": {k: float(v) for k, v in thresholds_metadata.items()},
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def compute_subgroup_fairness(
    X_eval: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    y_pred_override: np.ndarray | None = None,
) -> dict[str, Any]:
    """Desglosa desempeño y tasas de error por edad (>=50 vs <50), empleo y cuantiles de ingreso."""
    y_arr = np.asarray(y_true, dtype=np.int8)
    prob_arr = np.asarray(y_prob, dtype=np.float64)
    y_pred = (
        np.asarray(y_pred_override, dtype=np.int8)
        if y_pred_override is not None
        else (prob_arr >= threshold).astype(np.int8)
    )

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
            "fpr_ratio": (
                round(min(fpr_values) / max(fpr_values), 4)
                if len(fpr_values) >= 2 and max(fpr_values) > 0
                else 0.0
            ),
            "max_recall_gap": (
                round(max(recall_values) - min(recall_values), 4) if len(recall_values) >= 2 else 0.0
            ),
        }

    return report
