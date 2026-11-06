"""Pruebas automatizadas de integridad arquitectónica y reproducibilidad (Paso 1 / Hito 1)."""

from __future__ import annotations

import json

from src.config import (
    ALL_FEATURE_COLS,
    BAF_OPTIMIZED_DTYPES,
    EXPECTED_BAF_COLUMNS,
    FLAML_ESTIMATORS,
    PROJECT_ROOT,
    PROTECTED_SUBGROUP_COLS,
    SEED,
    TARGET_COL,
)


def test_mandatory_repository_structure() -> None:
    """Verifica que existan todos los archivos y carpetas obligatorios del lineamiento."""
    required_paths = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "data" / "README.md",
        PROJECT_ROOT / "src" / "config.py",
        PROJECT_ROOT / "src" / "data" / "download.py",
        PROJECT_ROOT / "src" / "data" / "loader.py",
        PROJECT_ROOT / "src" / "features" / "preprocessing.py",
        PROJECT_ROOT / "src" / "models" / "pipelines.py",
        PROJECT_ROOT / "src" / "models" / "train.py",
        PROJECT_ROOT / "src" / "evaluation" / "metrics.py",
        PROJECT_ROOT / "src" / "app" / "streamlit_app.py",
        PROJECT_ROOT / "notebooks" / "README.md",
        PROJECT_ROOT / "experiments" / "seed_config.json",
        PROJECT_ROOT / "docs" / "bitacora_ia.md",
        PROJECT_ROOT / "docs" / "hito1_propuesta.md",
        PROJECT_ROOT / "docs" / "hito1_propuesta.tex",
    ]
    for path in required_paths:
        assert path.exists(), f"Falta archivo o directorio obligatorio: {path}"


def test_requirements_strictly_pinned() -> None:
    """Garantiza que todas las dependencias en requirements.txt tengan versión exacta (==)."""
    req_lines = [
        line.strip()
        for line in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(req_lines) >= 12
    for line in req_lines:
        assert "==" in line, f"La dependencia '{line}' no tiene versión exacta fijada con '=='."


def test_baf_schema_and_memory_dtypes() -> None:
    """Valida las 32 columnas exactas de BAF Base.csv, sin duplicados y sin float64/int64."""
    assert SEED == 42
    assert len(EXPECTED_BAF_COLUMNS) == 32
    assert len(set(EXPECTED_BAF_COLUMNS)) == 32
    assert TARGET_COL not in ALL_FEATURE_COLS
    assert set(EXPECTED_BAF_COLUMNS) == set(BAF_OPTIMIZED_DTYPES.keys())

    allowed_dtypes = {"int8", "int32", "float32", "category"}
    for col, dtype_name in BAF_OPTIMIZED_DTYPES.items():
        assert dtype_name in allowed_dtypes, f"Columna {col} usa tipo no optimizado: {dtype_name}"

    for subgroup_col in ("customer_age", "employment_status", "income"):
        assert subgroup_col in PROTECTED_SUBGROUP_COLS
        assert subgroup_col in ALL_FEATURE_COLS


def test_flaml_and_seed_manifest_consistency() -> None:
    """Verifica estimadores obligatorios de FLAML y consistencia de seed_config.json."""
    assert FLAML_ESTIMATORS == ["lgbm", "xgboost", "rf", "extra_tree", "lrl1"]
    seed_cfg = json.loads(
        (PROJECT_ROOT / "experiments" / "seed_config.json").read_text(encoding="utf-8")
    )
    assert seed_cfg["seed"] == 42
    assert seed_cfg["predeclared_evaluation_hito1"]["primary_metric"] == "ap"
