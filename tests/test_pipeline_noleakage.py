"""Pruebas matemáticas anti-leakage entre Train/Val/Test y dentro de folds de validación cruzada.

Demuestra formalmente (Fase 3 CRISP-ML(Q)):
1. Disyunción exacta y estratificación determinista entre particiones `train`, `val` y `test`.
2. Invarianza estricta de los estadísticos del preprocesador (`SimpleImputer`, `VarianceThreshold`,
   `StandardScaler`, `OneHotEncoder`) ante perturbaciones extremas en `val` y `test`.
3. Preservación de negativos legítimos (`credit_risk_score < 0`) frente a centinelas (`< 0 -> NaN`
   solo en `MISSING_NEGATIVE_COLS`) y eliminación limpia in-fold de `device_fraud_count == 0`.
4. Aislamiento fold a fold en `StratifiedKFold` tanto para `build_baseline_pipeline` como para
   `build_smote_pipeline` (`imblearn.pipeline.Pipeline`).
5. Calibración del umbral operativo `tau` exclusivamente sobre `Val` sin fuga hacia `Test`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from src.config import (
    ALL_FEATURE_COLS,
    EXPECTED_BAF_COLUMNS,
    MISSING_NEGATIVE_COLS,
    SEED,
    TARGET_COL,
)
from src.data.loader import _optimize_baf_dtypes, create_stratified_splits
from src.evaluation.metrics import compute_classification_metrics, threshold_at_max_fpr
from src.features.preprocessing import build_preprocessor
from src.models.pipelines import build_baseline_pipeline, build_smote_pipeline


def _make_synthetic_baf_df(n_rows: int = 1000, seed: int = SEED) -> pd.DataFrame:
    """Genera un DataFrame sintético determinista con las 32 columnas exactas de BAF Base.csv."""
    rng = np.random.default_rng(seed)

    # Prevalencia controlada (~4% para garantizar suficientes positivos en cada fold de prueba unitaria)
    y = np.zeros(n_rows, dtype=np.int8)
    n_pos = max(40, int(round(0.04 * n_rows)))
    y[:n_pos] = 1
    rng.shuffle(y)

    df = pd.DataFrame(
        {
            "fraud_bool": y,
            "income": rng.choice([0.1, 0.3, 0.5, 0.7, 0.9], size=n_rows).astype(np.float32),
            "name_email_similarity": (
                rng.uniform(0.05, 0.95, size=n_rows) - 0.20 * y
            ).clip(0.0, 1.0).astype(np.float32),
            "prev_address_months_count": rng.choice(
                [-1.0, 12.0, 24.0, 60.0], p=[0.70, 0.10, 0.10, 0.10], size=n_rows
            ).astype(np.float32),
            "current_address_months_count": rng.choice(
                [-1.0, 6.0, 36.0, 120.0], p=[0.05, 0.35, 0.40, 0.20], size=n_rows
            ).astype(np.float32),
            "customer_age": rng.choice([20, 30, 40, 50, 60, 70], size=n_rows).astype(np.int32),
            "days_since_request": rng.exponential(scale=1.2, size=n_rows).astype(np.float32),
            "intended_balcon_amount": rng.choice(
                [-1.0, 15.5, 50.0, 100.0], p=[0.74, 0.10, 0.10, 0.06], size=n_rows
            ).astype(np.float32),
            "payment_type": rng.choice(["AA", "AB", "AC", "AD"], size=n_rows),
            "zip_count_4w": rng.integers(50, 4000, size=n_rows, dtype=np.int32),
            "velocity_6h": rng.normal(5000.0, 1500.0, size=n_rows).astype(np.float32),
            "velocity_24h": rng.normal(4800.0, 1200.0, size=n_rows).astype(np.float32),
            "velocity_4w": rng.normal(4900.0, 900.0, size=n_rows).astype(np.float32),
            "bank_branch_count_8w": rng.integers(0, 500, size=n_rows, dtype=np.int32),
            "date_of_birth_distinct_emails_4w": rng.integers(1, 25, size=n_rows, dtype=np.int32),
            "employment_status": rng.choice(["CA", "CB", "CC", "CD", "CE"], size=n_rows),
            # Incluye valores negativos reales de negocio (score crediticio < 0 NO es centinela)
            "credit_risk_score": (
                rng.integers(-120, 320, size=n_rows) + 60 * y
            ).astype(np.int32),
            "email_is_free": rng.integers(0, 2, size=n_rows, dtype=np.int8),
            "housing_status": rng.choice(["BA", "BB", "BC", "BD", "BE"], size=n_rows),
            "phone_home_valid": rng.integers(0, 2, size=n_rows, dtype=np.int8),
            "phone_mobile_valid": rng.integers(0, 2, size=n_rows, dtype=np.int8),
            "bank_months_count": rng.choice(
                [-1.0, 1.0, 15.0, 30.0], p=[0.25, 0.25, 0.25, 0.25], size=n_rows
            ).astype(np.float32),
            "has_other_cards": rng.integers(0, 2, size=n_rows, dtype=np.int8),
            "proposed_credit_limit": (
                rng.choice([200.0, 500.0, 1000.0, 1500.0], size=n_rows) + 500.0 * y
            ).astype(np.float32),
            "foreign_request": rng.integers(0, 2, size=n_rows, dtype=np.int8),
            "source": rng.choice(["INTERNET", "TELEAPP"], p=[0.99, 0.01], size=n_rows),
            "session_length_in_minutes": rng.choice(
                [-1.0, 2.5, 7.0, 18.0], p=[0.02, 0.48, 0.35, 0.15], size=n_rows
            ).astype(np.float32),
            "device_os": rng.choice(["windows", "other", "linux", "macintosh"], size=n_rows),
            "keep_alive_session": rng.integers(0, 2, size=n_rows, dtype=np.int8),
            "device_distinct_emails_8w": rng.choice(
                [-1.0, 1.0, 2.0], p=[0.01, 0.90, 0.09], size=n_rows
            ).astype(np.float32),
            # Columna degenerada constante igual a 0 documentada en BAF Base.csv
            "device_fraud_count": np.zeros(n_rows, dtype=np.int32),
            "month": rng.integers(0, 8, size=n_rows, dtype=np.int8),
        }
    )
    assert set(df.columns) == set(EXPECTED_BAF_COLUMNS)
    return _optimize_baf_dtypes(df)


def test_splits_disjoint_stratified_and_deterministic() -> None:
    """Demuestra partición 70/15/15 exacta, disyunción total de filas y determinismo seed=42."""
    df = _make_synthetic_baf_df(n_rows=1000, seed=SEED)
    # Codificar huella única en velocity_6h para rastrear disyunción matemática exacta de filas
    df["velocity_6h"] = np.arange(len(df), dtype=np.float32)

    splits_a = create_stratified_splits(df, dataset_sha256="dummy_sha256", seed=SEED)
    splits_b = create_stratified_splits(df, dataset_sha256="dummy_sha256", seed=SEED)

    assert len(splits_a.X_train) == 700
    assert len(splits_a.X_val) == 150
    assert len(splits_a.X_test) == 150
    assert TARGET_COL not in splits_a.X_train.columns
    assert set(splits_a.X_train.columns) == set(ALL_FEATURE_COLS)

    ids_train = set(splits_a.X_train["velocity_6h"].to_numpy().tolist())
    ids_val = set(splits_a.X_val["velocity_6h"].to_numpy().tolist())
    ids_test = set(splits_a.X_test["velocity_6h"].to_numpy().tolist())

    assert ids_train.isdisjoint(ids_val), "Fuga detectada: intersección no vacía entre Train y Val."
    assert ids_train.isdisjoint(ids_test), "Fuga detectada: intersección no vacía entre Train y Test."
    assert ids_val.isdisjoint(ids_test), "Fuga detectada: intersección no vacía entre Val y Test."
    assert len(ids_train | ids_val | ids_test) == len(df)

    # Estratificación preservada dentro de tolerancia de redondeo entero (<= 1 positivo)
    expected_rate = float(df[TARGET_COL].mean())
    for y_part in (splits_a.y_train, splits_a.y_val, splits_a.y_test):
        assert abs(float(y_part.mean()) - expected_rate) <= (1.0 / len(y_part) + 1e-6)

    # Reproducibilidad exacta bit a bit con seed=42
    pd.testing.assert_frame_equal(splits_a.X_train, splits_b.X_train)
    pd.testing.assert_series_equal(splits_a.y_test, splits_b.y_test)


def test_preprocessor_invariance_to_val_test_contamination() -> None:
    """Demuestra que alterar radicalmente Val y Test no altera ni un bit del ajuste en Train."""
    df = _make_synthetic_baf_df(n_rows=800, seed=SEED)
    splits = create_stratified_splits(df, dataset_sha256="sha", seed=SEED)

    prep_clean = build_preprocessor()
    prep_clean.fit(splits.X_train, splits.y_train)

    # Crear copia contaminada en Val y Test con outliers masivos y categorías nunca vistas
    X_val_contaminated = splits.X_val.copy()
    X_test_contaminated = splits.X_test.copy()
    X_val_contaminated["velocity_6h"] = np.float32(1e8)
    X_test_contaminated["proposed_credit_limit"] = np.float32(9e8)
    X_test_contaminated["device_os"] = pd.Categorical(["UNSEEN_QUANTUM_OS"] * len(X_test_contaminated))
    for col in MISSING_NEGATIVE_COLS:
        X_test_contaminated[col] = np.float32(-1.0)

    # Verificar que si se cometiera el error de ajustar sobre Train+Test, los estadísticos cambiarían
    prep_leaky = build_preprocessor()
    prep_leaky.fit(pd.concat([splits.X_train.astype(object), X_test_contaminated.astype(object)], ignore_index=True))
    clean_means = prep_clean.named_transformers_["num"].named_steps["scaler"].mean_
    leaky_means = prep_leaky.named_transformers_["num"].named_steps["scaler"].mean_
    assert not np.allclose(clean_means, leaky_means), "El test de control con fuga debería alterar la media."

    # El preprocesador limpio mantiene intacta la transformación sobre Train y maneja categorías nuevas
    Z_train = prep_clean.transform(splits.X_train)
    Z_test_contam = prep_clean.transform(X_test_contaminated)
    Z_single_row = prep_clean.transform(splits.X_test.iloc[[0]])

    assert not np.isnan(Z_train).any()
    assert not np.isnan(Z_test_contam).any()
    assert Z_single_row.shape == (1, Z_train.shape[1])

    # Verificar eliminación limpia de columna constante device_fraud_count == 0
    feature_names_out = list(prep_clean.get_feature_names_out())
    assert "num__device_fraud_count" not in feature_names_out

    # Verificar que negativos legítimos en credit_risk_score NO se borraron como NaN
    num_features_in = list(prep_clean.transformers_[0][2])
    crs_idx = num_features_in.index("credit_risk_score")
    sentinel_out = prep_clean.named_transformers_["num"].named_steps["sentinel_to_nan"].transform(
        splits.X_train[num_features_in]
    )
    assert (splits.X_train["credit_risk_score"] < 0).any()
    assert not np.isnan(sentinel_out[:, crs_idx]).any()


def test_cv_infold_isolation_baseline_and_smote() -> None:
    """Demuestra matemáticamente cero fuga entre folds de CV en Baseline y SMOTE in-fold."""
    df = _make_synthetic_baf_df(n_rows=600, seed=SEED)
    splits = create_stratified_splits(df, dataset_sha256="sha", seed=SEED)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(iter(skf.split(splits.X_train, splits.y_train)))

    X_fold_train = splits.X_train.iloc[train_idx].reset_index(drop=True)
    y_fold_train = splits.y_train.iloc[train_idx].reset_index(drop=True)
    X_fold_val = splits.X_train.iloc[val_idx].reset_index(drop=True)

    # Perturbar masivamente el fold de validación en un dataset completo alternativo
    X_perturbed_all = splits.X_train.copy()
    X_perturbed_all.iloc[val_idx, X_perturbed_all.columns.get_loc("velocity_24h")] = np.float32(5e7)
    X_fold_train_from_perturbed = X_perturbed_all.iloc[train_idx].reset_index(drop=True)

    for pipe_builder in (
        lambda: build_baseline_pipeline(seed=SEED),
        lambda: build_smote_pipeline(seed=SEED, sampling_strategy=0.2),
    ):
        p_orig = clone(pipe_builder())
        p_pert = clone(pipe_builder())

        p_orig.fit(X_fold_train, y_fold_train)
        p_pert.fit(X_fold_train_from_perturbed, y_fold_train)

        # Coeficientes, medianas y predicciones sobre X_fold_val original deben ser idénticos bit a bit
        np.testing.assert_array_equal(
            p_orig.named_steps["classifier"].coef_,
            p_pert.named_steps["classifier"].coef_,
        )
        probs_orig = p_orig.predict_proba(X_fold_val)
        probs_pert = p_pert.predict_proba(X_fold_val)
        np.testing.assert_array_equal(probs_orig, probs_pert)
        # SMOTE jamás sobremuestrea el fold de validación durante inferencia
        assert probs_orig.shape == (len(X_fold_val), 2)


def test_threshold_calibrated_exclusively_on_val_without_test_snooping() -> None:
    """Verifica que el umbral operativo tau depende solo de Val y no espía etiquetas de Test."""
    rng = np.random.default_rng(SEED)
    y_val = np.array([0] * 190 + [1] * 10, dtype=np.int8)
    val_probs = np.concatenate(
        [rng.uniform(0.01, 0.70, size=190), rng.uniform(0.55, 0.98, size=10)]
    )

    tau_a, val_rec, val_fpr = threshold_at_max_fpr(y_val, val_probs, max_fpr=0.05)
    assert val_fpr <= 0.05

    y_test_1 = np.array([0] * 190 + [1] * 10, dtype=np.int8)
    y_test_2 = 1 - y_test_1  # Inversión radical de etiquetas de Test
    test_probs = rng.uniform(0.01, 0.99, size=200)

    # El umbral aplicado en Test permanece fijo e invariante a y_test
    m1 = compute_classification_metrics(y_test_1, test_probs, threshold=tau_a, calibrated_on_val=True)
    m2 = compute_classification_metrics(y_test_2, test_probs, threshold=tau_a, calibrated_on_val=True)
    assert m1["threshold"] == tau_a
    assert m2["threshold"] == tau_a
    assert m1["recall_at_fpr5"] == m1["recall"]
    assert m1["actual_fpr_at_fpr5"] == m1["fpr"]


def test_age_group_threshold_mitigation_on_val_without_leakage() -> None:
    """Verifica que la calibración por grupo etario en Val iguala FPR sin espiar Test."""
    from src.evaluation.metrics import (
        calibrate_age_group_thresholds_val,
        compute_subgroup_fairness,
        predict_with_age_group_thresholds,
    )

    rng = np.random.default_rng(SEED)
    n_val, n_test = 600, 600

    df_val = _make_synthetic_baf_df(n_rows=n_val, seed=SEED)
    df_test = _make_synthetic_baf_df(n_rows=n_test, seed=SEED + 1)

    y_val = df_val["fraud_bool"].to_numpy(dtype=np.int8)
    y_test = df_test["fraud_bool"].to_numpy(dtype=np.int8)

    # Simular inflación sistemática de probabilidades en age >= 50
    is_ge50_val = (df_val["customer_age"].to_numpy() >= 50).astype(np.float64)
    is_ge50_test = (df_test["customer_age"].to_numpy() >= 50).astype(np.float64)

    val_probs = np.clip(rng.uniform(0.05, 0.65, size=n_val) + 0.22 * is_ge50_val + 0.25 * y_val, 0.0, 0.999)
    test_probs = np.clip(rng.uniform(0.05, 0.65, size=n_test) + 0.22 * is_ge50_test + 0.25 * y_test, 0.0, 0.999)

    cal_info = calibrate_age_group_thresholds_val(df_val, y_val, val_probs, max_fpr=0.05)
    assert cal_info["thresholds"]["age_>=50"] > cal_info["thresholds"]["age_<50"]

    mitigated_preds = predict_with_age_group_thresholds(df_test, test_probs, cal_info["thresholds"])
    fairness_mit = compute_subgroup_fairness(
        df_test, y_test, test_probs, y_pred_override=mitigated_preds
    )
    assert fairness_mit["age_group"]["max_fpr_gap"] <= 0.05
