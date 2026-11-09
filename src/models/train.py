"""Orquestador modular de entrenamiento, validación cruzada estratificada y registro en MLflow.

Uso desde la raíz del repositorio:
    python -m src.models.train --sample-size 100000 --mode baseline
    python -m src.models.train --sample-size 100000 --mode full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import (
    CV_SPLITS,
    DEFAULT_SAMPLE_SIZE,
    EXPERIMENTS_DIR,
    FLAML_ESTIMATORS,
    FLAML_MAX_ITER,
    MLFLOW_TRACKING_DIR,
    OPERATIONAL_MAX_FPR,
    PREDECLARED_ABSOLUTE_AP_GAIN,
    PREDECLARED_MAX_SUBGROUP_FPR_GAP,
    PREDECLARED_MIN_RECALL_AT_FPR5,
    PREDECLARED_RELATIVE_AP_GAIN,
    RAW_BAF_PATH,
    SEED,
    set_global_seed,
)
from src.data.loader import create_stratified_splits, load_baf_dataframe
from src.evaluation.metrics import (
    calibrate_age_group_thresholds_val,
    compute_classification_metrics,
    compute_metrics_with_custom_predictions,
    compute_subgroup_fairness,
    predict_with_age_group_thresholds,
    threshold_at_max_fpr,
)
from src.features.preprocessing import replace_negative_sentinels_df
from src.models.pipelines import build_baseline_pipeline, build_smote_pipeline


def evaluate_cv_pipeline(pipeline: Any, X_train: Any, y_train: Any, seed: int = SEED) -> dict[str, float]:
    """Ejecuta validación cruzada estratificada reportando media, desviación y varianza."""
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed)
    cv_scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=skf,
        scoring={"ap": "average_precision", "f1": "f1", "roc_auc": "roc_auc"},
        n_jobs=-1,
        return_train_score=False,
    )
    summary: dict[str, float] = {}
    for metric_key in ("ap", "f1", "roc_auc"):
        vals = cv_scores[f"test_{metric_key}"]
        summary[f"cv_{metric_key}_mean"] = float(np.mean(vals))
        summary[f"cv_{metric_key}_std"] = float(np.std(vals, ddof=1))
        summary[f"cv_{metric_key}_var"] = float(np.var(vals, ddof=1))
    return summary


def evaluate_flaml_best_cv_folds(
    automl: Any,
    X_train_clean: pd.DataFrame,
    y_train: pd.Series,
    seed: int = SEED,
) -> dict[str, Any]:
    """Calcula media, desviación estándar y varianza en 5-fold Stratified CV para el mejor modelo FLAML."""
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed)
    y_arr = y_train.to_numpy(dtype=np.int8)

    ap_folds: list[float] = []
    f1_folds: list[float] = []
    roc_folds: list[float] = []

    base_model = automl.model
    for tr_idx, va_idx in skf.split(X_train_clean, y_arr):
        X_tr = X_train_clean.iloc[tr_idx].copy()
        y_tr = y_arr[tr_idx]
        X_va = X_train_clean.iloc[va_idx].copy()
        y_va = y_arr[va_idx]

        if getattr(automl, "_transformer", None) is not None:
            transformer = type(automl._transformer)()
            X_tr_t, _ = transformer.fit_transform(X_tr, y_tr, task="classification")
            X_va_t = transformer.transform(X_va)
        else:
            X_tr_t, X_va_t = X_tr, X_va

        import copy

        fold_model = copy.deepcopy(base_model.estimator)
        fold_model.fit(X_tr_t, y_tr)
        probs = fold_model.predict_proba(X_va_t)[:, 1]
        preds = (probs >= 0.5).astype(np.int8)

        ap_folds.append(float(average_precision_score(y_va, probs)))
        f1_folds.append(float(f1_score(y_va, preds, zero_division=0)))
        roc_folds.append(float(roc_auc_score(y_va, probs)))

    return {
        "cv_ap_mean": float(np.mean(ap_folds)),
        "cv_ap_std": float(np.std(ap_folds, ddof=1)),
        "cv_ap_var": float(np.var(ap_folds, ddof=1)),
        "cv_ap_folds": [round(v, 6) for v in ap_folds],
        "cv_f1_mean": float(np.mean(f1_folds)),
        "cv_f1_std": float(np.std(f1_folds, ddof=1)),
        "cv_f1_var": float(np.var(f1_folds, ddof=1)),
        "cv_roc_auc_mean": float(np.mean(roc_folds)),
        "cv_roc_auc_std": float(np.std(roc_folds, ddof=1)),
        "cv_roc_auc_var": float(np.var(roc_folds, ddof=1)),
    }


def extract_flaml_feature_importances(automl: Any, default_cols: list[str]) -> list[dict[str, float | str]]:
    """Extrae e interpreta las importancias de características del mejor estimador de FLAML."""
    est = automl.model.estimator
    importances = getattr(est, "feature_importances_", None)
    if importances is None and hasattr(est, "coef_"):
        importances = np.abs(np.ravel(est.coef_))
    if importances is None:
        return []

    feat_names = getattr(est, "feature_name_", None) or getattr(est, "feature_names_in_", None)
    if feat_names is None or len(feat_names) != len(importances):
        feat_names = default_cols[: len(importances)]

    imp_arr = np.asarray(importances, dtype=np.float64)
    total = float(imp_arr.sum()) if imp_arr.sum() > 0 else 1.0
    pairs = [
        {
            "feature": str(name),
            "importance_raw": float(val),
            "importance_pct": round(float(100.0 * val / total), 2),
        }
        for name, val in zip(feat_names, imp_arr)
    ]
    pairs.sort(key=lambda d: float(d["importance_raw"]), reverse=True)
    return pairs


def build_error_analysis_report(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_probs: np.ndarray,
    y_pred: np.ndarray,
    group_thresholds: dict[str, float] | None = None,
    scalar_threshold: float | None = None,
    n_cases_per_type: int = 6,
) -> dict[str, Any]:
    """Extrae >= 12 casos reales de error (6 FP + 6 FN) en Test y estadísticas comparativas TP/FN/FP/TN."""
    y_arr = y_test.to_numpy(dtype=np.int8)
    prob_arr = np.asarray(test_probs, dtype=np.float64)
    pred_arr = np.asarray(y_pred, dtype=np.int8)

    fp_indices = np.where((y_arr == 0) & (pred_arr == 1))[0]
    fn_indices = np.where((y_arr == 1) & (pred_arr == 0))[0]
    tp_indices = np.where((y_arr == 1) & (pred_arr == 1))[0]
    tn_indices = np.where((y_arr == 0) & (pred_arr == 0))[0]

    # Ordenar FP más extremos (mayor probabilidad asignada a un cliente legítimo)
    fp_sorted = fp_indices[np.argsort(-prob_arr[fp_indices])][:n_cases_per_type]
    # Ordenar FN más críticos (menor probabilidad asignada a un fraude real que evadió el filtro)
    fn_sorted = fn_indices[np.argsort(prob_arr[fn_indices])][:n_cases_per_type]

    key_cols = [
        "customer_age",
        "income",
        "credit_risk_score",
        "proposed_credit_limit",
        "name_email_similarity",
        "employment_status",
        "housing_status",
        "device_os",
        "keep_alive_session",
        "phone_home_valid",
        "email_is_free",
        "prev_address_months_count",
        "current_address_months_count",
        "intended_balcon_amount",
        "velocity_6h",
        "device_distinct_emails_8w",
        "month",
    ]

    def _serialize_case(idx: int, error_type: str, rank: int) -> dict[str, Any]:
        row = X_test.iloc[idx]
        age_val = int(row["customer_age"])
        age_grp = "age_>=50" if age_val >= 50 else "age_<50"
        active_tau = (
            float(group_thresholds[age_grp])
            if group_thresholds is not None
            else float(scalar_threshold or 0.5)
        )
        profile: dict[str, Any] = {}
        for c in key_cols:
            val = row[c]
            if isinstance(val, (np.floating, float)):
                profile[c] = round(float(val), 4)
            elif isinstance(val, (np.integer, int)):
                profile[c] = int(val)
            else:
                profile[c] = str(val)

        # Diagnóstico causal automático basado en las señales dominantes del caso
        drivers: list[str] = []
        if error_type == "FP":
            if float(row["proposed_credit_limit"]) >= 1000:
                drivers.append(f"cupo solicitado alto ({float(row['proposed_credit_limit']):.0f})")
            if str(row["device_os"]) == "windows":
                drivers.append("device_os=windows (mayor riesgo basal)")
            if int(row["keep_alive_session"]) == 0:
                drivers.append("keep_alive_session=0 (sesión sin persistencia)")
            if int(row["phone_home_valid"]) == 0:
                drivers.append("phone_home_valid=0")
            if float(row["name_email_similarity"]) < 0.35:
                drivers.append(f"baja similitud nombre-correo ({float(row['name_email_similarity']):.2f})")
            if float(row["prev_address_months_count"]) < 0:
                drivers.append("ausencia de historial de domicilio previo (-1)")
            if float(row["income"]) >= 0.7:
                drivers.append(f"cuantil de ingreso alto ({float(row['income']):.1f})")
        else:
            if int(row["keep_alive_session"]) == 1:
                drivers.append("keep_alive_session=1 (emula sesión legítima)")
            if str(row["device_os"]) in ("other", "linux", "macintosh"):
                drivers.append(f"device_os={row['device_os']} (baja tasa basal)")
            if float(row["proposed_credit_limit"]) <= 500:
                drivers.append(f"cupo solicitado conservador ({float(row['proposed_credit_limit']):.0f})")
            if float(row["name_email_similarity"]) >= 0.60:
                drivers.append(f"alta similitud nombre-correo ({float(row['name_email_similarity']):.2f})")
            if int(row["phone_home_valid"]) == 1:
                drivers.append("phone_home_valid=1")
            if float(row["current_address_months_count"]) >= 48:
                drivers.append(
                    f"antigüedad declarada alta en domicilio ({float(row['current_address_months_count']):.0f}m)"
                )

        return {
            "case_id": f"{error_type}_{rank:02d}_test_idx_{int(idx)}",
            "test_row_index": int(idx),
            "error_type": error_type,
            "y_true": int(y_arr[idx]),
            "y_pred": int(pred_arr[idx]),
            "predicted_prob": round(float(prob_arr[idx]), 6),
            "active_threshold_tau": round(active_tau, 6),
            "age_group": age_grp,
            "features": profile,
            "causal_drivers": drivers,
        }

    cases: list[dict[str, Any]] = []
    for r, idx in enumerate(fp_sorted, start=1):
        cases.append(_serialize_case(int(idx), "FP", r))
    for r, idx in enumerate(fn_sorted, start=1):
        cases.append(_serialize_case(int(idx), "FN", r))

    def _quadrant_summary(indices: np.ndarray) -> dict[str, Any]:
        if len(indices) == 0:
            return {"count": 0}
        sub = X_test.iloc[indices]
        return {
            "count": int(len(indices)),
            "mean_prob": round(float(prob_arr[indices].mean()), 4),
            "mean_customer_age": round(float(sub["customer_age"].mean()), 2),
            "pct_age_ge50": round(float(100.0 * (sub["customer_age"] >= 50).mean()), 2),
            "mean_income": round(float(sub["income"].mean()), 3),
            "mean_credit_risk_score": round(float(sub["credit_risk_score"].mean()), 2),
            "mean_proposed_credit_limit": round(float(sub["proposed_credit_limit"].mean()), 2),
            "mean_name_email_similarity": round(float(sub["name_email_similarity"].mean()), 3),
            "pct_device_os_windows": round(float(100.0 * (sub["device_os"].astype(str) == "windows").mean()), 2),
            "pct_keep_alive_0": round(float(100.0 * (sub["keep_alive_session"] == 0).mean()), 2),
            "pct_prev_address_missing": round(
                float(100.0 * (sub["prev_address_months_count"] < 0).mean()), 2
            ),
        }

    return {
        "total_errors_audited": len(cases),
        "fp_cases_count": len(fp_sorted),
        "fn_cases_count": len(fn_sorted),
        "quadrant_profiles": {
            "TP": _quadrant_summary(tp_indices),
            "FN": _quadrant_summary(fn_indices),
            "FP": _quadrant_summary(fp_indices),
            "TN": _quadrant_summary(tn_indices),
        },
        "cases": cases,
    }


def run_flaml_cash_with_mlflow(
    X_train: Any,
    y_train: Any,
    max_iter: int = FLAML_MAX_ITER,
    seed: int = SEED,
) -> tuple[Any, dict[str, dict[str, Any]]]:
    """Ejecuta búsqueda CASH con FLAML sobre DataFrame sin leakage previo y registra child runs."""
    from flaml import AutoML

    # Solo conversión fila a fila sin estado (<0 -> NaN); FLAML gestiona tipos dentro de su flujo
    X_train_clean = replace_negative_sentinels_df(X_train)

    automl = AutoML()
    automl.fit(
        X_train=X_train_clean,
        y_train=y_train.to_numpy(),
        task="classification",
        metric="ap",
        estimator_list=list(FLAML_ESTIMATORS),
        eval_method="cv",
        n_splits=CV_SPLITS,
        split_type="stratified",
        max_iter=max_iter,
        seed=seed,
        mlflow_logging=False,
        verbose=1,
    )

    leaderboard: dict[str, dict[str, Any]] = {}
    for est_name in FLAML_ESTIMATORS:
        best_cfg = automl.best_config_per_estimator.get(est_name)
        best_loss = automl.best_loss_per_estimator.get(est_name)
        if best_cfg is not None and best_loss is not None:
            cv_ap_est = float(1.0 - float(best_loss))
            clean_cfg = {
                str(k): (
                    int(v)
                    if isinstance(v, (np.integer, int)) and not isinstance(v, bool)
                    else float(v)
                    if isinstance(v, (np.floating, float))
                    else str(v)
                    if not isinstance(v, bool)
                    else bool(v)
                )
                for k, v in best_cfg.items()
            }
            leaderboard[est_name] = {
                "cv_ap_best": cv_ap_est,
                "best_config": clean_cfg,
            }
            with mlflow.start_run(run_name=f"flaml_child_{est_name}", nested=True):
                mlflow.log_param("estimator_family", est_name)
                mlflow.log_param("seed", seed)
                for k, v in clean_cfg.items():
                    mlflow.log_param(f"hp_{k}", v)
                mlflow.log_metric("cv_ap_best", cv_ap_est)

    return automl, leaderboard


def train_and_track(
    csv_path: Path = RAW_BAF_PATH,
    sample_size: int | None = DEFAULT_SAMPLE_SIZE,
    mode: str = "baseline",
    flaml_max_iter: int = FLAML_MAX_ITER,
    seed: int = SEED,
) -> dict[str, Any]:
    """Ejecuta el flujo de entrenamiento reproducible registrando artefactos en experiments/."""
    set_global_seed(seed)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_DIR.as_uri())
    mlflow.set_experiment("fraud-detection-baf-crisp-ml")

    df, sha256_hex = load_baf_dataframe(csv_path=csv_path, sample_size=sample_size, seed=seed)
    splits = create_stratified_splits(df=df, dataset_sha256=sha256_hex, seed=seed)

    results: dict[str, Any] = {
        "seed": seed,
        "dataset_sha256": splits.dataset_sha256,
        "total_rows_loaded": splits.total_rows_loaded,
        "fraud_rate": round(splits.fraud_rate, 6),
        "memory_usage_mb": splits.memory_usage_mb,
        "splits_summary": {
            "train": {
                "rows": int(len(splits.y_train)),
                "positives": int(splits.y_train.sum()),
                "fraud_rate": round(float(splits.y_train.mean()), 6),
            },
            "val": {
                "rows": int(len(splits.y_val)),
                "positives": int(splits.y_val.sum()),
                "fraud_rate": round(float(splits.y_val.mean()), 6),
            },
            "test": {
                "rows": int(len(splits.y_test)),
                "positives": int(splits.y_test.sum()),
                "fraud_rate": round(float(splits.y_test.mean()), 6),
            },
        },
    }

    # 1. Línea Base: LogisticRegression(class_weight="balanced") en Pipeline
    with mlflow.start_run(run_name="baseline_logreg_balanced"):
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "imbalance_strategy": "class_weight=balanced",
                "seed": seed,
                "sample_size": splits.total_rows_loaded,
                "dataset_sha256": splits.dataset_sha256,
                "cv_splits": CV_SPLITS,
                "operational_max_fpr": OPERATIONAL_MAX_FPR,
            }
        )
        baseline_pipe = build_baseline_pipeline(seed=seed)
        cv_summary = evaluate_cv_pipeline(baseline_pipe, splits.X_train, splits.y_train, seed=seed)
        mlflow.log_metrics(cv_summary)

        baseline_pipe.fit(splits.X_train, splits.y_train)
        val_probs = baseline_pipe.predict_proba(splits.X_val)[:, 1]
        opt_thr, val_rec5, val_fpr5 = threshold_at_max_fpr(
            splits.y_val.to_numpy(), val_probs, OPERATIONAL_MAX_FPR
        )

        test_probs = baseline_pipe.predict_proba(splits.X_test)[:, 1]
        test_naive_metrics = compute_classification_metrics(
            splits.y_test, test_probs, threshold=0.50, calibrated_on_val=False
        )
        test_metrics = compute_classification_metrics(
            splits.y_test, test_probs, threshold=opt_thr, calibrated_on_val=True
        )
        subgroup_metrics = compute_subgroup_fairness(
            splits.X_test, splits.y_test, test_probs, threshold=opt_thr
        )

        # Calibración por grupo etario en Val para la Línea Base
        base_age_cal = calibrate_age_group_thresholds_val(
            splits.X_val, splits.y_val, val_probs, max_fpr=OPERATIONAL_MAX_FPR
        )
        base_mitigated_preds = predict_with_age_group_thresholds(
            splits.X_test, test_probs, base_age_cal["thresholds"]
        )
        base_mitigated_test = compute_metrics_with_custom_predictions(
            splits.y_test, test_probs, base_mitigated_preds, base_age_cal["thresholds"]
        )
        base_mitigated_fairness = compute_subgroup_fairness(
            splits.X_test,
            splits.y_test,
            test_probs,
            threshold=opt_thr,
            y_pred_override=base_mitigated_preds,
        )

        mlflow.log_metrics(
            {
                "val_calibrated_threshold": opt_thr,
                "val_recall_at_fpr5": val_rec5,
                "val_fpr_at_fpr5": val_fpr5,
                "test_naive_thr05_f1": test_naive_metrics["f1"],
                "test_naive_thr05_precision": test_naive_metrics["precision"],
                "test_naive_thr05_recall": test_naive_metrics["recall"],
                "test_naive_thr05_fpr": test_naive_metrics["fpr"],
                "test_ap": test_metrics["ap"],
                "test_f1": test_metrics["f1"],
                "test_precision": test_metrics["precision"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_recall_at_fpr5": test_metrics["recall_at_fpr5"],
                "test_fpr_at_val_tau": test_metrics["fpr"],
                "test_age_max_fpr_gap": subgroup_metrics["age_group"]["max_fpr_gap"],
                "test_age_mitigated_fpr_gap": base_mitigated_fairness["age_group"]["max_fpr_gap"],
                "test_employment_max_fpr_gap": subgroup_metrics["employment_status"]["max_fpr_gap"],
                "test_income_max_fpr_gap": subgroup_metrics["income_band"]["max_fpr_gap"],
            }
        )

        artifacts_dir = EXPERIMENTS_DIR / "models"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(baseline_pipe, artifacts_dir / "baseline_logreg.joblib")

        results["baseline"] = {
            "cv": cv_summary,
            "val_calibration": {
                "operational_max_fpr": OPERATIONAL_MAX_FPR,
                "calibrated_threshold_tau": opt_thr,
                "val_recall_at_tau": val_rec5,
                "val_fpr_at_tau": val_fpr5,
            },
            "test_naive_threshold_050": test_naive_metrics,
            "test": test_metrics,
            "fairness": subgroup_metrics,
            "age_fairness_mitigation": {
                "val_group_calibration": base_age_cal,
                "test_mitigated": base_mitigated_test,
                "fairness_mitigated": base_mitigated_fairness,
            },
        }

    # 2. Modo completo (Fase 3): Comparativa SMOTE in-fold + búsqueda CASH con FLAML + Equidad + Errores
    if mode == "full":
        # 2.a Regresión Logística con SMOTE in-fold (sampling_strategy=0.1)
        with mlflow.start_run(run_name="logreg_smote_infold"):
            mlflow.log_params(
                {
                    "model_type": "LogisticRegression",
                    "imbalance_strategy": "SMOTE_in_fold_0.1",
                    "seed": seed,
                    "dataset_sha256": splits.dataset_sha256,
                }
            )
            smote_pipe = build_smote_pipeline(seed=seed, sampling_strategy=0.1)
            smote_cv = evaluate_cv_pipeline(smote_pipe, splits.X_train, splits.y_train, seed=seed)
            smote_pipe.fit(splits.X_train, splits.y_train)
            val_probs_smote = smote_pipe.predict_proba(splits.X_val)[:, 1]
            opt_thr_smote, val_rec_smote, val_fpr_smote = threshold_at_max_fpr(
                splits.y_val.to_numpy(), val_probs_smote, OPERATIONAL_MAX_FPR
            )
            test_probs_smote = smote_pipe.predict_proba(splits.X_test)[:, 1]
            smote_test_naive = compute_classification_metrics(
                splits.y_test, test_probs_smote, threshold=0.50, calibrated_on_val=False
            )
            smote_test_cal = compute_classification_metrics(
                splits.y_test, test_probs_smote, threshold=opt_thr_smote, calibrated_on_val=True
            )
            smote_fairness = compute_subgroup_fairness(
                splits.X_test, splits.y_test, test_probs_smote, threshold=opt_thr_smote
            )
            mlflow.log_metrics(
                {
                    **smote_cv,
                    "val_calibrated_threshold": opt_thr_smote,
                    "test_ap": smote_test_cal["ap"],
                    "test_f1": smote_test_cal["f1"],
                    "test_roc_auc": smote_test_cal["roc_auc"],
                    "test_recall_at_fpr5": smote_test_cal["recall_at_fpr5"],
                    "test_fpr_at_val_tau": smote_test_cal["fpr"],
                }
            )
            results["logreg_smote_cv"] = smote_cv
            results["logreg_smote_infold"] = {
                "cv": smote_cv,
                "val_calibration": {
                    "calibrated_threshold_tau": opt_thr_smote,
                    "val_recall_at_tau": val_rec_smote,
                    "val_fpr_at_tau": val_fpr_smote,
                },
                "test_naive_threshold_050": smote_test_naive,
                "test": smote_test_cal,
                "fairness": smote_fairness,
            }

        # 2.b GBDT (LightGBM) con SMOTE in-fold para contrastar SMOTE en árboles vs sin SMOTE
        with mlflow.start_run(run_name="lgbm_smote_infold"):
            lgbm_est = LGBMClassifier(
                n_estimators=150,
                learning_rate=0.05,
                num_leaves=31,
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            )
            lgbm_smote_pipe = build_smote_pipeline(
                estimator=lgbm_est, seed=seed, sampling_strategy=0.1
            )
            lgbm_smote_cv = evaluate_cv_pipeline(
                lgbm_smote_pipe, splits.X_train, splits.y_train, seed=seed
            )
            lgbm_smote_pipe.fit(splits.X_train, splits.y_train)
            val_probs_lgbm_sm = lgbm_smote_pipe.predict_proba(splits.X_val)[:, 1]
            opt_thr_lgbm_sm, val_rec_lgbm_sm, val_fpr_lgbm_sm = threshold_at_max_fpr(
                splits.y_val.to_numpy(), val_probs_lgbm_sm, OPERATIONAL_MAX_FPR
            )
            test_probs_lgbm_sm = lgbm_smote_pipe.predict_proba(splits.X_test)[:, 1]
            lgbm_smote_test = compute_classification_metrics(
                splits.y_test, test_probs_lgbm_sm, threshold=opt_thr_lgbm_sm, calibrated_on_val=True
            )
            lgbm_smote_fairness = compute_subgroup_fairness(
                splits.X_test, splits.y_test, test_probs_lgbm_sm, threshold=opt_thr_lgbm_sm
            )
            mlflow.log_params(
                {
                    "model_type": "LGBMClassifier",
                    "imbalance_strategy": "SMOTE_in_fold_0.1",
                    "seed": seed,
                    "dataset_sha256": splits.dataset_sha256,
                }
            )
            mlflow.log_metrics(
                {
                    **lgbm_smote_cv,
                    "val_calibrated_threshold": opt_thr_lgbm_sm,
                    "test_ap": lgbm_smote_test["ap"],
                    "test_f1": lgbm_smote_test["f1"],
                    "test_roc_auc": lgbm_smote_test["roc_auc"],
                    "test_recall_at_fpr5": lgbm_smote_test["recall_at_fpr5"],
                    "test_fpr_at_val_tau": lgbm_smote_test["fpr"],
                }
            )
            results["lgbm_smote_infold"] = {
                "cv": lgbm_smote_cv,
                "val_calibration": {
                    "calibrated_threshold_tau": opt_thr_lgbm_sm,
                    "val_recall_at_tau": val_rec_lgbm_sm,
                    "val_fpr_at_tau": val_fpr_lgbm_sm,
                },
                "test": lgbm_smote_test,
                "fairness": lgbm_smote_fairness,
            }

        # 2.c Búsqueda CASH en FLAML + Mitigación de Sesgo Etario + Auditoría de Errores
        with mlflow.start_run(run_name="flaml_cash_parent"):
            mlflow.log_params(
                {
                    "automl_library": "FLAML",
                    "primary_metric": "ap",
                    "estimators": ",".join(FLAML_ESTIMATORS),
                    "max_iter": flaml_max_iter,
                    "cv_splits": CV_SPLITS,
                    "seed": seed,
                    "dataset_sha256": splits.dataset_sha256,
                }
            )
            automl, flaml_leaderboard = run_flaml_cash_with_mlflow(
                splits.X_train, splits.y_train, max_iter=flaml_max_iter, seed=seed
            )
            X_train_clean = replace_negative_sentinels_df(splits.X_train)
            flaml_cv_summary = evaluate_flaml_best_cv_folds(
                automl, X_train_clean, splits.y_train, seed=seed
            )

            X_val_clean = replace_negative_sentinels_df(splits.X_val)
            X_test_clean = replace_negative_sentinels_df(splits.X_test)

            val_probs_flaml = automl.predict_proba(X_val_clean)[:, 1]
            opt_thr_f, val_rec5_f, val_fpr5_f = threshold_at_max_fpr(
                splits.y_val.to_numpy(), val_probs_flaml, OPERATIONAL_MAX_FPR
            )
            test_probs_flaml = automl.predict_proba(X_test_clean)[:, 1]

            flaml_test_naive = compute_classification_metrics(
                splits.y_test, test_probs_flaml, threshold=0.50, calibrated_on_val=False
            )
            flaml_test = compute_classification_metrics(
                splits.y_test, test_probs_flaml, threshold=opt_thr_f, calibrated_on_val=True
            )
            flaml_fairness = compute_subgroup_fairness(
                splits.X_test, splits.y_test, test_probs_flaml, threshold=opt_thr_f
            )

            # Mitigación de sesgo por subgrupo etario calibrando umbrales en Val sin fuga a Test
            flaml_age_cal = calibrate_age_group_thresholds_val(
                splits.X_val, splits.y_val, val_probs_flaml, max_fpr=OPERATIONAL_MAX_FPR
            )
            flaml_mitigated_preds = predict_with_age_group_thresholds(
                splits.X_test, test_probs_flaml, flaml_age_cal["thresholds"]
            )
            flaml_mitigated_test = compute_metrics_with_custom_predictions(
                splits.y_test,
                test_probs_flaml,
                flaml_mitigated_preds,
                flaml_age_cal["thresholds"],
            )
            flaml_mitigated_fairness = compute_subgroup_fairness(
                splits.X_test,
                splits.y_test,
                test_probs_flaml,
                threshold=opt_thr_f,
                y_pred_override=flaml_mitigated_preds,
            )

            feature_importances = extract_flaml_feature_importances(
                automl, list(splits.X_train.columns)
            )
            error_report = build_error_analysis_report(
                splits.X_test,
                splits.y_test,
                test_probs_flaml,
                flaml_mitigated_preds,
                group_thresholds=flaml_age_cal["thresholds"],
                scalar_threshold=opt_thr_f,
                n_cases_per_type=6,
            )
            error_json_path = EXPERIMENTS_DIR / "error_analysis_hito3.json"
            error_json_path.write_text(
                json.dumps(error_report, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            best_cv_ap = float(1.0 - float(automl.best_loss))
            base_cv_ap = results["baseline"]["cv"]["cv_ap_mean"]
            rel_ap_gain = float((best_cv_ap - base_cv_ap) / base_cv_ap) if base_cv_ap > 0 else 0.0
            abs_ap_gain = float(best_cv_ap - base_cv_ap)

            mlflow.log_metrics(
                {
                    "best_cv_ap": best_cv_ap,
                    "cv_ap_mean": flaml_cv_summary["cv_ap_mean"],
                    "cv_ap_std": flaml_cv_summary["cv_ap_std"],
                    "cv_ap_var": flaml_cv_summary["cv_ap_var"],
                    "cv_f1_mean": flaml_cv_summary["cv_f1_mean"],
                    "cv_roc_auc_mean": flaml_cv_summary["cv_roc_auc_mean"],
                    "val_calibrated_threshold": opt_thr_f,
                    "val_recall_at_fpr5": val_rec5_f,
                    "val_fpr_at_fpr5": val_fpr5_f,
                    "test_ap": flaml_test["ap"],
                    "test_f1": flaml_test["f1"],
                    "test_precision": flaml_test["precision"],
                    "test_roc_auc": flaml_test["roc_auc"],
                    "test_recall_at_fpr5": flaml_test["recall_at_fpr5"],
                    "test_fpr_at_val_tau": flaml_test["fpr"],
                    "test_age_unmitigated_fpr_gap": flaml_fairness["age_group"]["max_fpr_gap"],
                    "test_age_mitigated_fpr_gap": flaml_mitigated_fairness["age_group"]["max_fpr_gap"],
                    "test_mitigated_recall_at_fpr5": flaml_mitigated_test["recall_at_fpr5"],
                    "test_mitigated_fpr": flaml_mitigated_test["fpr"],
                    "test_mitigated_f1": flaml_mitigated_test["f1"],
                }
            )

            joblib.dump(
                {
                    "automl": automl,
                    "threshold": opt_thr_f,
                    "age_group_thresholds": flaml_age_cal["thresholds"],
                },
                EXPERIMENTS_DIR / "models" / "flaml_best.joblib",
            )
            results["flaml_best"] = {
                "best_estimator": automl.best_estimator,
                "best_config": flaml_leaderboard.get(automl.best_estimator, {}).get(
                    "best_config", {}
                ),
                "best_cv_ap": best_cv_ap,
                "cv": flaml_cv_summary,
                "estimators_leaderboard": flaml_leaderboard,
                "val_calibration": {
                    "operational_max_fpr": OPERATIONAL_MAX_FPR,
                    "calibrated_threshold_tau": opt_thr_f,
                    "val_recall_at_tau": val_rec5_f,
                    "val_fpr_at_tau": val_fpr5_f,
                },
                "test_naive_threshold_050": flaml_test_naive,
                "test": flaml_test,
                "fairness": flaml_fairness,
                "age_fairness_mitigation": {
                    "val_group_calibration": flaml_age_cal,
                    "test_mitigated": flaml_mitigated_test,
                    "fairness_mitigated": flaml_mitigated_fairness,
                },
                "top_feature_importances": feature_importances[:15],
                "hito1_success_verification": {
                    "baseline_cv_ap": round(base_cv_ap, 6),
                    "best_cv_ap": round(best_cv_ap, 6),
                    "relative_ap_gain": round(rel_ap_gain, 4),
                    "absolute_ap_gain": round(abs_ap_gain, 6),
                    "meets_primary_ap_criterion": bool(
                        rel_ap_gain >= PREDECLARED_RELATIVE_AP_GAIN
                        or abs_ap_gain >= PREDECLARED_ABSOLUTE_AP_GAIN
                    ),
                    "test_recall_at_fpr5_global_tau": round(flaml_test["recall_at_fpr5"], 4),
                    "test_recall_at_fpr5_mitigated_tau": round(
                        flaml_mitigated_test["recall_at_fpr5"], 4
                    ),
                    "meets_recall_at_fpr5_criterion": bool(
                        flaml_mitigated_test["recall_at_fpr5"] >= PREDECLARED_MIN_RECALL_AT_FPR5
                        and flaml_mitigated_test["fpr"] <= OPERATIONAL_MAX_FPR + 0.005
                    ),
                    "age_fpr_gap_unmitigated": flaml_fairness["age_group"]["max_fpr_gap"],
                    "age_fpr_gap_mitigated": flaml_mitigated_fairness["age_group"]["max_fpr_gap"],
                    "meets_fairness_age_gap_criterion": bool(
                        flaml_mitigated_fairness["age_group"]["max_fpr_gap"]
                        <= PREDECLARED_MAX_SUBGROUP_FPR_GAP
                    ),
                },
            }
            results["error_analysis_summary"] = {
                "artifact_path": "experiments/error_analysis_hito3.json",
                "total_errors_audited": error_report["total_errors_audited"],
                "quadrant_profiles": error_report["quadrant_profiles"],
            }

    out_json = EXPERIMENTS_DIR / f"latest_{mode}_results.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Resultados verificables exportados en: {out_json}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento reproducible CRISP-ML(Q).")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Tamaño de muestra estratificada (0 para usar los 1.000.000 de registros).",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "full"],
        default="baseline",
        help="'baseline' ejecuta Regresión Logística balanceada; 'full' añade SMOTE + FLAML CASH.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=FLAML_MAX_ITER,
        help="Número determinista de iteraciones max_iter para FLAML.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    effective_sample = None if args.sample_size <= 0 else args.sample_size
    train_and_track(
        sample_size=effective_sample,
        mode=args.mode,
        flaml_max_iter=args.max_iter,
    )
