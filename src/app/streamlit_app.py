"""Prototipo interactivo en Streamlit para evaluación en vivo (Criterio 9 / Hito 4).

Permite evaluar solicitudes de apertura de cuenta en tiempo real, ajustar dinámicamente el
umbral de decisión (tau) y auditar el impacto sobre métricas globales y subgrupos.

Ejecución:
    streamlit run src/app/streamlit_app.py
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from src.config import EXPERIMENTS_DIR, OPERATIONAL_MAX_FPR, SEED

st.set_page_config(
    page_title="Detección de Fraude en Onboarding Bancario | CRISP-ML(Q)",
    layout="wide",
)

st.title("Sistema de Detección de Fraude en Apertura de Cuentas Digitales")
st.caption(
    f"Metodología CRISP-ML(Q) | Ruta C (Aprendizaje Automático) | Dataset BAF (NeurIPS 2022) | seed={SEED}"
)

st.sidebar.header("Control Operativo en Vivo")
decision_threshold = st.sidebar.slider(
    "Umbral de decisión de fraude (tau)",
    min_value=0.01,
    max_value=0.99,
    value=0.50,
    step=0.01,
    help="Ajuste interactivo para evaluar el compromiso entre Recall de fraude y fricción (FPR).",
)
st.sidebar.info(f"Política objetivo: FPR <= {OPERATIONAL_MAX_FPR:.0%} | Brecha subgrupos <= 5%")

flaml_artifact = EXPERIMENTS_DIR / "models" / "flaml_best.joblib"
baseline_artifact = EXPERIMENTS_DIR / "models" / "baseline_logreg.joblib"

col_info, col_live = st.columns([1, 1])

with col_info:
    st.subheader("Estado de Artefactos Experimentales")
    for summary_name in ("latest_full_results.json", "latest_baseline_results.json"):
        summary_path = EXPERIMENTS_DIR / summary_name
        if summary_path.is_file():
            st.success(f"Resultados cargados desde `{summary_path.name}`")
            st.json(json.loads(summary_path.read_text(encoding="utf-8")))
            break
    else:
        st.warning(
            "Aún no se han generado artefactos entrenados. Ejecute primero:\n"
            "`python -m src.data.download` y `python -m src.models.train --mode baseline`"
        )

with col_live:
    st.subheader("Simulador de Solicitud de Apertura (Evaluación en Vivo)")
    with st.form("live_application_form"):
        c1, c2 = st.columns(2)
        customer_age = c1.number_input("Edad del solicitante (customer_age)", 18, 90, 34, step=1)
        income = c2.slider("Cuantil de ingresos (income)", 0.1, 0.9, 0.5, 0.1)
        credit_risk_score = c1.number_input("Puntaje de riesgo crediticio", -170, 390, 120)
        proposed_credit_limit = c2.number_input("Cupo solicitado (proposed_credit_limit)", 200.0, 2000.0, 500.0)
        name_email_similarity = c1.slider("Similitud nombre-correo", 0.0, 1.0, 0.65, 0.01)
        employment_status = c2.selectbox("Estado laboral", ["CA", "CB", "CC", "CD", "CE", "CF", "CG"])
        device_os = c1.selectbox("Sistema operativo (device_os)", ["windows", "other", "linux", "macintosh", "x11"])
        device_distinct_emails_8w = c2.number_input("Correos distintos en dispositivo (8 sem.)", -1, 5, 1)
        submitted = st.form_submit_button("Evaluar Riesgo de Fraude")

    if submitted:
        sample_record = pd.DataFrame(
            [
                {
                    "income": income,
                    "name_email_similarity": name_email_similarity,
                    "prev_address_months_count": -1.0,
                    "current_address_months_count": 36.0,
                    "customer_age": customer_age,
                    "days_since_request": 0.02,
                    "intended_balcon_amount": -1.0,
                    "payment_type": "AA",
                    "zip_count_4w": 1200,
                    "velocity_6h": 4500.0,
                    "velocity_24h": 4200.0,
                    "velocity_4w": 4800.0,
                    "bank_branch_count_8w": 10,
                    "date_of_birth_distinct_emails_4w": 5,
                    "employment_status": employment_status,
                    "credit_risk_score": credit_risk_score,
                    "email_is_free": 1,
                    "housing_status": "BC",
                    "phone_home_valid": 0,
                    "phone_mobile_valid": 1,
                    "bank_months_count": 24.0,
                    "has_other_cards": 0,
                    "proposed_credit_limit": proposed_credit_limit,
                    "foreign_request": 0,
                    "source": "INTERNET",
                    "session_length_in_minutes": 5.5,
                    "device_os": device_os,
                    "keep_alive_session": 1,
                    "device_distinct_emails_8w": float(device_distinct_emails_8w),
                    "device_fraud_count": 0,
                    "month": 3,
                }
            ]
        )
        if flaml_artifact.is_file():
            from src.features.preprocessing import replace_negative_sentinels_df

            bundle = joblib.load(flaml_artifact)
            prob = float(
                bundle["automl"].predict_proba(replace_negative_sentinels_df(sample_record))[0, 1]
            )
            model_label = f"FLAML ({bundle['automl'].best_estimator})"
        elif baseline_artifact.is_file():
            pipe = joblib.load(baseline_artifact)
            prob = float(pipe.predict_proba(sample_record)[0, 1])
            model_label = "Baseline (LogisticRegression balanced)"
        else:
            prob = None
            model_label = None

        if prob is not None:
            is_fraud = prob >= decision_threshold
            st.metric(f"Probabilidad estimada ({model_label})", f"{prob:.4f}")
            if is_fraud:
                st.error(f"ALERTA DE FRAUDE ACTIVADA (p = {prob:.4f} >= tau = {decision_threshold:.2f})")
            else:
                st.success(f"SOLICITUD APROBADA (p = {prob:.4f} < tau = {decision_threshold:.2f})")
        else:
            st.info("Entrene primero la línea base para habilitar la inferencia en tiempo real.")
