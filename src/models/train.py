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
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import (
    CV_SPLITS,
    DEFAULT_SAMPLE_SIZE,
    EXPERIMENTS_DIR,
    FLAML_ESTIMATORS,
    FLAML_MAX_ITER,
    MLFLOW_TRACKING_DIR,
    OPERATIONAL_MAX_FPR,
    RAW_BAF_PATH,
    SEED,
    set_global_seed,
)
from src.data.loader import create_stratified_splits, load_baf_dataframe
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_subgroup_fairness,
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


def run_flaml_cash_with_mlflow(
    X_train: Any,
    y_train: Any,
    max_iter: int = FLAML_MAX_ITER,
    seed: int = SEED,
) -> Any:
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
        verbose=1,
    )

    for est_name in FLAML_ESTIMATORS:
        best_cfg = automl.best_config_per_estimator.get(est_name)
        best_loss = automl.best_loss_per_estimator.get(est_name)
        if best_cfg is not None and best_loss is not None:
            with mlflow.start_run(run_name=f"flaml_child_{est_name}", nested=True):
                mlflow.log_param("estimator_family", est_name)
                mlflow.log_param("seed", seed)
                for k, v in best_cfg.items():
                    mlflow.log_param(f"hp_{k}", v)
                mlflow.log_metric("cv_ap_best", float(1.0 - float(best_loss)))

    return automl


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
        }

    # 2. Modo completo (Fase 3): Comparativa SMOTE in-fold + búsqueda CASH con FLAML
    if mode == "full":
        with mlflow.start_run(run_name="logreg_smote_infold"):
            mlflow.log_params(
                {
                    "model_type": "LogisticRegression",
                    "imbalance_strategy": "SMOTE_in_fold_0.1",
                    "seed": seed,
                    "dataset_sha256": splits.dataset_sha256,
                }
            )
            smote_pipe = build_smote_pipeline(seed=seed)
            smote_cv = evaluate_cv_pipeline(smote_pipe, splits.X_train, splits.y_train, seed=seed)
            mlflow.log_metrics(smote_cv)
            results["logreg_smote_cv"] = smote_cv

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
            automl = run_flaml_cash_with_mlflow(
                splits.X_train, splits.y_train, max_iter=flaml_max_iter, seed=seed
            )
            val_probs_flaml = automl.predict_proba(
                replace_negative_sentinels_df(splits.X_val)
            )[:, 1]
            opt_thr_f, _, _ = threshold_at_max_fpr(
                splits.y_val.to_numpy(), val_probs_flaml, OPERATIONAL_MAX_FPR
            )
            test_probs_flaml = automl.predict_proba(
                replace_negative_sentinels_df(splits.X_test)
            )[:, 1]
            flaml_test = compute_classification_metrics(
                splits.y_test, test_probs_flaml, threshold=opt_thr_f, calibrated_on_val=True
            )
            flaml_fairness = compute_subgroup_fairness(
                splits.X_test, splits.y_test, test_probs_flaml, threshold=opt_thr_f
            )
            mlflow.log_metric("best_cv_ap", float(1.0 - float(automl.best_loss)))
            mlflow.log_metric("test_ap", flaml_test["ap"])
            mlflow.log_metric("test_f1", flaml_test["f1"])
            mlflow.log_metric("test_roc_auc", flaml_test["roc_auc"])

            joblib.dump(
                {"automl": automl, "threshold": opt_thr_f},
                EXPERIMENTS_DIR / "models" / "flaml_best.joblib",
            )
            results["flaml_best"] = {
                "best_estimator": automl.best_estimator,
                "best_cv_ap": float(1.0 - float(automl.best_loss)),
                "test": flaml_test,
                "fairness": flaml_fairness,
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
