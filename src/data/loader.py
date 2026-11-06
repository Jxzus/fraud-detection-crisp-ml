"""Carga eficiente en memoria (float32/int32), muestreo estratificado y partición sin leakage.

Cumple con la restricción de ejecución rápida en recursos gratuitos (PC local / Colab / Kaggle):
1. Convierte tipos de datos durante e inmediatamente después de la lectura (~65% menos RAM).
2. Permite trabajar con una muestra estratificada reproducible (ej. 50.000 a 100.000 filas con
   seed=42) preservando intacta la tasa real de fraude (~1.10%) y la distribución temporal.
3. Separa estrictamente train/val/test ANTES de ajustar cualquier transformador o SMOTE.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import (
    BINARY_INT8_COLS,
    CATEGORICAL_COLS,
    DEFAULT_SAMPLE_SIZE,
    EXPECTED_BAF_COLUMNS,
    FLOAT32_COLS,
    INTEGER_INT32_COLS,
    RAW_BAF_PATH,
    SEED,
    TARGET_COL,
    TEMPORAL_COL,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VAL_FRACTION,
    set_global_seed,
)
from src.data.download import compute_file_sha256

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class DatasetSplits:
    """Contenedor tipado para las particiones libres de fuga de datos y sus metadatos."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    dataset_sha256: str
    total_rows_loaded: int
    fraud_rate: float
    memory_usage_mb: float


def _optimize_baf_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Compacta tipos numéricos a float32/int32/int8 y categóricos a category sin pérdida útil."""
    import numpy as np

    df[TARGET_COL] = df[TARGET_COL].astype(np.int8)
    df[TEMPORAL_COL] = df[TEMPORAL_COL].astype(np.int8)

    for col in BINARY_INT8_COLS:
        df[col] = df[col].astype(np.int8)

    for col in INTEGER_INT32_COLS:
        df[col] = df[col].astype(np.int32)

    for col in FLOAT32_COLS:
        df[col] = df[col].astype(np.float32)

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    return df


def load_baf_dataframe(
    csv_path: Path = RAW_BAF_PATH,
    sample_size: int | None = DEFAULT_SAMPLE_SIZE,
    seed: int = SEED,
) -> tuple[pd.DataFrame, str]:
    """Carga Base.csv con tipos compactos y aplica muestreo estratificado determinista opcional.

    Args:
        csv_path: Ruta hacia `data/raw/Base.csv`.
        sample_size: Número de registros a muestrear estratificadamente (ej. 50_000 o 100_000).
            Si es `None` o mayor/igual al tamaño total, conserva los 1.000.000 de registros.
        seed: Semilla fija para reproducibilidad (`SEED = 42`).

    Returns:
        Tupla `(df_optimizado, sha256_hex)`.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos en '{csv_path}'. "
            "Ejecute primero: python -m src.data.download"
        )

    set_global_seed(seed)
    sha256_hex = compute_file_sha256(csv_path)

    # Lectura inicial con float32 para numéricos (evita picos de float64 y tolera '1500.0')
    initial_dtypes: dict[str, str] = {
        **{col: "category" for col in CATEGORICAL_COLS},
        **{
            col: "float32"
            for col in (
                (TARGET_COL, TEMPORAL_COL)
                + BINARY_INT8_COLS
                + INTEGER_INT32_COLS
                + FLOAT32_COLS
            )
        },
    }

    df = pd.read_csv(csv_path, dtype=initial_dtypes, low_memory=False)

    missing_cols = set(EXPECTED_BAF_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"El CSV no cumple el esquema esperado de BAF. Faltan: {sorted(missing_cols)}")

    df = _optimize_baf_dtypes(df)

    if sample_size is not None and 0 < sample_size < len(df):
        # Estratificación conjunta por (fraud_bool, month) para preservar prevalencia (~1.10%) y tiempo
        stratify_key = df[TARGET_COL].astype(str) + "_" + df[TEMPORAL_COL].astype(str)
        if stratify_key.value_counts().min() < 2:
            stratify_key = df[TARGET_COL]

        df_sampled, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=stratify_key,
            random_state=seed,
        )
        df = df_sampled.sort_index().reset_index(drop=True)

    return df, sha256_hex


def create_stratified_splits(
    df: pd.DataFrame,
    dataset_sha256: str,
    seed: int = SEED,
) -> DatasetSplits:
    """Genera particiones Train (70%), Val (15%) y Test (15%) estratificadas sin data leakage."""
    from sklearn.model_selection import train_test_split

    if abs((TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION) - 1.0) > 1e-6:
        raise ValueError("Las fracciones de train, val y test deben sumar exactamente 1.0.")

    set_global_seed(seed)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].copy()

    holdout_fraction = VAL_FRACTION + TEST_FRACTION
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=holdout_fraction,
        stratify=y,
        random_state=seed,
    )

    relative_test_fraction = TEST_FRACTION / holdout_fraction
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_fraction,
        stratify=y_temp,
        random_state=seed,
    )

    mem_mb = float(df.memory_usage(deep=True).sum()) / (1024 * 1024)
    return DatasetSplits(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        dataset_sha256=dataset_sha256,
        total_rows_loaded=len(df),
        fraud_rate=float(y.mean()),
        memory_usage_mb=round(mem_mb, 2),
    )
