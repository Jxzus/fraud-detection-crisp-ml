"""Constructores de pipelines para Línea Base (class_weight) y remuestreo SMOTE sin data leakage.

Regla de oro de ingeniería (Fase 3 CRISP-ML(Q)):
- La Línea Base usa `sklearn.pipeline.Pipeline` con `LogisticRegression(class_weight="balanced")`.
- Cualquier experimento con `SMOTE` DEBE encapsularse en `imblearn.pipeline.Pipeline` para que
  los puntos sintéticos se generen exclusivamente dentro de la partición de entrenamiento de
  cada fold en validación cruzada, jamás antes del split.
"""

from __future__ import annotations

from typing import Any

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline

from src.config import SEED
from src.features.preprocessing import build_preprocessor


def build_baseline_pipeline(seed: int = SEED, c_reg: float = 1.0) -> SkPipeline:
    """Construye la línea base mínima obligatoria: Regresión Logística balanceada en Pipeline."""
    return SkPipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    C=c_reg,
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def build_smote_pipeline(
    estimator: Any | None = None,
    seed: int = SEED,
    sampling_strategy: float | str = 0.1,
) -> ImbPipeline:
    """Construye un pipeline de imbalanced-learn que ejecuta SMOTE solo dentro de cada fold."""
    if estimator is None:
        estimator = LogisticRegression(
            class_weight=None,
            max_iter=1000,
            solver="lbfgs",
            random_state=seed,
        )

    return ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("smote", SMOTE(sampling_strategy=sampling_strategy, random_state=seed)),
            ("classifier", estimator),
        ]
    )
