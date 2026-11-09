# `fraud-detection-crisp-ml` — Detección de Fraude en Apertura Digital de Cuentas Bancarias

Sistema modular de Aprendizaje Automático (**Ruta C**, capítulos 19–21 de Russell & Norvig) desarrollado bajo la metodología **CRISP-ML(Q)** (Studer et al., 2021) para detectar fraude por suplantación e identidades sintéticas en solicitudes digitales de apertura de cuentas bancarias en el contexto financiero colombiano (**Superintendencia Financiera de Colombia — SFC** y **Asobancaria**).

---

## Instalación y Ejecución Completa (4 Comandos)

Requisitos previos: **Python 3.10–3.12** en recursos gratuitos (PC local, Google Colab o Kaggle).

```bash
# 1. Instalar dependencias exactas fijadas (reproducibilidad garantizada)
pip install -r requirements.txt

# 2. Descargar Base.csv (BAF NeurIPS 2022, CC BY-NC-SA 4.0) y verificar hash SHA-256
python -m src.data.download

# 3. Ejecutar entrenamiento sin data leakage (Baseline + SMOTE in-fold + FLAML CASH + MLflow)
python -m src.models.train --sample-size 100000 --mode full

# 4. Iniciar prototipo interactivo para inferencia y ajuste de umbral en vivo
streamlit run src/app/streamlit_app.py
```
> *Nota para ejecución rápida solo de Línea Base:* reemplazar `--mode full` por `--mode baseline` en el comando 3.

---

## Resumen de Resultados Verificados (Hitos 1, 2 y 3 — `seed = 42`)

| Modelo / Configuración (`n = 100.000`, `Test = 15.000`) | `5-Fold CV AUC-PR` | `Test AUC-PR` | `Test Recall@FPR<=5%` | `Test FPR` | `Test ROC-AUC` | Equidad Etaria (`Delta FPR`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Línea Base `LogisticRegression (balanced)`** ($\hat{\tau}_{\text{val}}=0,8173$) | `0,1494 ± 0,0360` | `0,1413` | `0,4970` (`82/165`) | `0,0495` | `0,8805` | `0,0927` *(incumple `<= 0,05`)* |
| **`LogisticRegression + SMOTE` in-fold (`0.1`)** ($\hat{\tau}_{\text{val}}=0,2956$) | `0,1575 ± 0,0346` | `0,1421` | `0,5394` (`89/165`) | `0,0502` | `0,8847` | `0,0811` *(incumple `<= 0,05`)* |
| **`LightGBM + SMOTE` in-fold (`0.1`)** ($\hat{\tau}_{\text{val}}=0,0506$) | `0,1562 ± 0,0315` | `0,1537` | `0,5273` (`87/165`) | `0,0457` | `0,8902` | `0,0718` *(incumple `<= 0,05`)* |
| **`FLAML Best (lgbm)` — Umbral Único** ($\hat{\tau}_{\text{val}}=0,0355$) | **`0,1797`** (`+20,25%`) | **`0,1556`** | `0,5273` (`87/165`) | `0,0496` | **`0,8887`** | `0,0786` *(incumple `<= 0,05`)* |
| **`FLAML Best (lgbm)` + Umbrales Equitativos en Val** | **`0,1797` (Cumple)** | **`0,1556`** | **`0,5030` (Cumple)** | **`0,0477`** | **`0,8887`** | **`0,0070` (Cumple `<= 0,05`)** |

---

## Arquitectura del Repositorio

```text
fraud-detection-crisp-ml/
├── README.md                     # Guía general, resultados verificados y ejecución en 4 comandos
├── requirements.txt              # Versiones exactas fijadas (pinned versions)
├── pytest.ini                    # Configuración de pruebas automatizadas (9/9 en verde)
├── data/                         # Datos (BAF NeurIPS 2022, CC BY-NC-SA 4.0, cero PII)
│   └── README.md                 # Datasheet resumido, esquema y gobernanza
├── src/                          # Código modular de producción (cero lógica en notebooks)
│   ├── config.py                 # Semilla global (SEED=42), rutas y umbrales Hito 1
│   ├── data/
│   │   ├── download.py           # Descarga automática con reanudación HTTP Range y SHA-256
│   │   └── loader.py             # Carga float32/int32, muestreo estratificado y splits
│   ├── features/
│   │   └── preprocessing.py      # Transformador de centinelas negativos y ColumnTransformer
│   ├── models/
│   │   ├── pipelines.py          # Baseline (class_weight) e imblearn Pipeline (SMOTE in-fold)
│   │   └── train.py              # Validación cruzada 5-fold, FLAML CASH, equidad y errores
│   ├── evaluation/
│   │   └── metrics.py            # AUC-PR, Recall@FPR<=5% y calibración equitativa en Val
│   └── app/
│       └── streamlit_app.py      # Prototipo interactivo con umbrales equitativos por subgrupo
├── notebooks/                    # Exclusivo para análisis exploratorio (EDA)
│   └── 01_eda_baf.ipynb          # EDA ejecutado (desbalance, centinelas, deriva y sesgo)
├── tests/                        # Pruebas automatizadas de esquema y cero data leakage
│   ├── test_scaffolding.py       # Integridad estructural, dtypes y configuración seed=42
│   └── test_pipeline_noleakage.py# Demostración matemática de aislamiento Train/Val/Test y CV
├── experiments/                  # Artefactos JSON verificables, modelos y figuras
│   ├── seed_config.json          # Criterios predeclarados congelados en el Hito 1
│   ├── dataset_manifest.json     # Hash SHA-256 verificado de Base.csv
│   ├── data_quality_audit.json   # Auditoría de memoria (-81,49% RAM) y centinelas
│   ├── latest_baseline_results.json # Resultados verificados de la Línea Base (Hito 2)
│   ├── latest_full_results.json  # Resultados completos SMOTE + FLAML + Equidad (Hito 3)
│   ├── error_analysis_hito3.json # Auditoría causal de 12 casos reales (6 FP + 6 FN) en Test
│   ├── models/                   # Modelos entrenados serializados (baseline y flaml_best)
│   └── figures/                  # 9 figuras PNG de EDA (01-05) y evaluación Hito 3 (06-09)
└── docs/
    ├── hito1_propuesta.md        # Propuesta formal de 1 página (Hito 1)
    ├── hito1_propuesta.tex       # Fuente LaTeX de 1 página del Hito 1
    ├── hito2_avance.md           # Avance de 1 página con métricas reales de Línea Base (Hito 2)
    ├── hito2_avance.tex          # Fuente LaTeX de 1 página del Hito 2
    ├── fraud_detection_crisp_ml_hito2.pdf # Entregable compilado de 1 página (Hito 2)
    ├── hito3_modelado_evaluacion.md  # Informe de 1 página de Modelado, Equidad y Errores (Hito 3)
    ├── hito3_modelado_evaluacion.tex # Fuente LaTeX de 1 página del Hito 3
    ├── fraud_detection_crisp_ml_hito3.pdf # Entregable compilado de 1 página (Hito 3)
    └── bitacora_ia.md            # Bitácora auditable humano-IA y reflexión crítica (media página)
```

---

## Reglas de Ingeniería y Calidad (CRISP-ML(Q))

1. **Reproducibilidad estricta:** Semilla fija `SEED = 42` en particiones, validación cruzada estratificada (`n_splits=5`), `SMOTE` y `FLAML` controlado por `max_iter` (sin `time_budget` dependiente del hardware).
2. **Eficiencia computacional:** Conversión explícita a `float32`/`int32`/`int8`/`category` (`-81,49%` RAM) y muestreo estratificado configurable (`50.000`–`100.000` registros preservando el `~1,103%` de fraude).
3. **Cero fuga de información (*Zero Data Leakage*):** Imputación, filtro de varianza nula, escalado, codificación y sobremuestreo `SMOTE` (`imblearn.pipeline.Pipeline`) ejecutados estrictamente dentro de cada fold de entrenamiento; umbrales operativos y de equidad calibrados exclusivamente sobre `Val`.
4. **IA Responsable verificada:** Optimización primaria por `AUC-PR` (`ap`), control operativo `Recall@FPR<=5% >= 0,50` y mitigación post-procesamiento de sesgo etario logrando `Delta FPR = 0,0070 <= 0,05` en `Test`.
