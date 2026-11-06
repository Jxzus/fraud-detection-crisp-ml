"""Configuración centralizada, constantes de reproducibilidad y esquema del dataset BAF.

Todas las fases de CRISP-ML(Q) consumen las constantes definidas en este módulo para
garantizar determinismo (SEED = 42), trazabilidad y ausencia de fuga de información.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# ==============================================================================
# 1. Semilla maestra y determinismo global
# ==============================================================================
SEED: Final[int] = 42


def set_global_seed(seed: int = SEED) -> None:
    """Fija las semillas pseudoaleatorias de Python, variables de entorno y NumPy."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


# ==============================================================================
# 2. Rutas normativas del repositorio
# ==============================================================================
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
DATA_RAW_DIR: Final[Path] = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
EXPERIMENTS_DIR: Final[Path] = PROJECT_ROOT / "experiments"
MLFLOW_TRACKING_DIR: Final[Path] = EXPERIMENTS_DIR / "mlruns"
DOCS_DIR: Final[Path] = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR: Final[Path] = PROJECT_ROOT / "notebooks"

RAW_BAF_FILENAME: Final[str] = "Base.csv"
RAW_BAF_PATH: Final[Path] = DATA_RAW_DIR / RAW_BAF_FILENAME
DATASET_MANIFEST_PATH: Final[Path] = EXPERIMENTS_DIR / "dataset_manifest.json"

# ==============================================================================
# 3. Esquema oficial del Bank Account Fraud (BAF) Dataset Suite - NeurIPS 2022
#    Fuente: Jesus et al. (2022), Feedzai | Licencia: CC BY-NC-SA 4.0
# ==============================================================================
KAGGLE_DATASET_HANDLE: Final[str] = "sgpjesus/bank-account-fraud-dataset-neurips-2022"
HF_FALLBACK_BASE_CSV_URL: Final[str] = (
    "https://huggingface.co/datasets/jAEhEEkIM/operationbench-baf-raw/resolve/main/Base.csv"
)
DATASET_LICENSE: Final[str] = "CC BY-NC-SA 4.0"

TARGET_COL: Final[str] = "fraud_bool"
TEMPORAL_COL: Final[str] = "month"

CATEGORICAL_COLS: Final[tuple[str, ...]] = (
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
)

BINARY_INT8_COLS: Final[tuple[str, ...]] = (
    "email_is_free",
    "phone_home_valid",
    "phone_mobile_valid",
    "has_other_cards",
    "foreign_request",
    "keep_alive_session",
)

INTEGER_INT32_COLS: Final[tuple[str, ...]] = (
    "customer_age",
    "zip_count_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "credit_risk_score",
    "device_fraud_count",
)

# Columnas numéricas continuas o con centinelas negativos (-1 -> NaN en float32 sin upcast a float64)
FLOAT32_COLS: Final[tuple[str, ...]] = (
    "income",
    "name_email_similarity",
    "prev_address_months_count",
    "current_address_months_count",
    "days_since_request",
    "intended_balcon_amount",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_months_count",
    "proposed_credit_limit",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
)

NUMERIC_COLS: Final[tuple[str, ...]] = BINARY_INT8_COLS + INTEGER_INT32_COLS + FLOAT32_COLS
ALL_FEATURE_COLS: Final[tuple[str, ...]] = NUMERIC_COLS + CATEGORICAL_COLS + (TEMPORAL_COL,)
EXPECTED_BAF_COLUMNS: Final[tuple[str, ...]] = (TARGET_COL,) + ALL_FEATURE_COLS

# Convenciones de valores faltantes codificados como negativos en BAF (Datasheet NeurIPS 2022)
MISSING_NEGATIVE_COLS: Final[tuple[str, ...]] = (
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "intended_balcon_amount",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
)

# Atributos auditados para métricas de equidad por subgrupo (IA Responsable - Criterio 7)
PROTECTED_SUBGROUP_COLS: Final[tuple[str, ...]] = (
    "customer_age",
    "employment_status",
    "income",
)

# Mapa explícito de dtypes eficientes en memoria (~65% ahorro de RAM frente a float64/int64/object)
BAF_OPTIMIZED_DTYPES: Final[dict[str, str]] = {
    TARGET_COL: "int8",
    TEMPORAL_COL: "int8",
    **{col: "category" for col in CATEGORICAL_COLS},
    **{col: "int8" for col in BINARY_INT8_COLS},
    **{col: "int32" for col in INTEGER_INT32_COLS},
    **{col: "float32" for col in FLOAT32_COLS},
}

# ==============================================================================
# 4. Parámetros de muestreo eficiente y evaluación predeclarada (Hito 1)
# ==============================================================================
DEFAULT_SAMPLE_SIZE: Final[int] = 100_000  # Configurable entre 50_000 y 100_000 (o None para 1M)
TRAIN_FRACTION: Final[float] = 0.70
VAL_FRACTION: Final[float] = 0.15
TEST_FRACTION: Final[float] = 0.15

PRIMARY_METRIC: Final[str] = "ap"  # Average Precision (AUC-PR)
CV_SPLITS: Final[int] = 5
FLAML_ESTIMATORS: Final[list[str]] = ["lgbm", "xgboost", "rf", "extra_tree", "lrl1"]
FLAML_MAX_ITER: Final[int] = 30  # Se usa max_iter en lugar de time_budget por reproducibilidad

# Umbrales de éxito congelados en el Hito 1 antes de experimentar
PREDECLARED_RELATIVE_AP_GAIN: Final[float] = 0.15  # >= +15% mejora relativa sobre baseline en AUC-PR
PREDECLARED_ABSOLUTE_AP_GAIN: Final[float] = 0.05  # o >= +0.05 puntos absolutos en AUC-PR
OPERATIONAL_MAX_FPR: Final[float] = 0.05  # Tope operativo de falsos positivos (5% sugerido por BAF)
PREDECLARED_MIN_RECALL_AT_FPR5: Final[float] = 0.50  # Recall objetivo a FPR <= 5%
PREDECLARED_MAX_SUBGROUP_FPR_GAP: Final[float] = 0.05  # Brecha máxima de FPR entre subgrupos <= 5%


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuración inmutable de una corrida experimental registrable en MLflow."""

    sample_size: int | None = DEFAULT_SAMPLE_SIZE
    seed: int = SEED
    cv_splits: int = CV_SPLITS
    primary_metric: str = PRIMARY_METRIC
    flaml_max_iter: int = FLAML_MAX_ITER
    flaml_estimators: list[str] = field(default_factory=lambda: list(FLAML_ESTIMATORS))
    experiment_name: str = "fraud-detection-baf-crisp-ml"
