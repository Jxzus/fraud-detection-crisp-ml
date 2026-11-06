"""Preprocesamiento tabular anti-leakage para el dataset BAF (NeurIPS 2022).

Todo estadístico (medianas de imputación, medias/desviaciones de escalado y categorías de
OneHotEncoder) se ajusta EXCLUSIVAMENTE sobre el subconjunto de entrenamiento de cada fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ALL_FEATURE_COLS,
    CATEGORICAL_COLS,
    MISSING_NEGATIVE_COLS,
    NUMERIC_COLS,
    TEMPORAL_COL,
)


class NegativeSentinelToNaN(BaseEstimator, TransformerMixin):
    """Reemplaza valores centinela negativos (< 0) de BAF por NaN preservando float32."""

    def __init__(self, sentinel_cols: tuple[str, ...] = MISSING_NEGATIVE_COLS) -> None:
        self.sentinel_cols = sentinel_cols

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> NegativeSentinelToNaN:  # noqa: ARG002
        self.feature_names_in_ = list(X.columns) if hasattr(X, "columns") else None
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        for col in df.columns:
            df[col] = df[col].astype(np.float32)
            if col in self.sentinel_cols:
                mask = df[col] < 0
                if mask.any():
                    df.loc[mask, col] = np.nan
        return df.to_numpy(dtype=np.float32)


def build_preprocessor(include_temporal_col: bool = True) -> ColumnTransformer:
    """Construye el ColumnTransformer compatible con sklearn e imblearn en precisión float32."""
    numeric_features = list(NUMERIC_COLS) + ([TEMPORAL_COL] if include_temporal_col else [])
    categorical_features = list(CATEGORICAL_COLS)

    numeric_pipeline = Pipeline(
        steps=[
            ("sentinel_to_nan", NegativeSentinelToNaN()),
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def replace_negative_sentinels_df(
    X: pd.DataFrame,
    sentinel_cols: tuple[str, ...] = MISSING_NEGATIVE_COLS,
) -> pd.DataFrame:
    """Transformación fila a fila SIN ESTADO (cero data leakage): convierte centinelas < 0 a NaN."""
    df = X.copy()
    for col in sentinel_cols:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)
            mask = df[col] < 0
            if mask.any():
                df.loc[mask, col] = np.nan
    return df


def validate_feature_columns(X: pd.DataFrame) -> None:
    """Verifica que el DataFrame de entrada contenga las características esperadas."""
    missing = set(ALL_FEATURE_COLS) - set(X.columns)
    if missing:
        raise ValueError(f"Faltan columnas de entrada requeridas: {sorted(missing)}")
