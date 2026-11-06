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
> *Nota para Hito 2 / ejecución rápida solo de Línea Base:* reemplazar `--mode full` por `--mode baseline` en el comando 3.

---

## Arquitectura del Repositorio

```text
fraud-detection-crisp-ml/
├── README.md                     # Guía general y ejecución en < 5 comandos
├── requirements.txt              # Versiones exactas fijadas (pinned versions)
├── data/                         # Datos (BAF NeurIPS 2022, CC BY-NC-SA 4.0, cero PII)
│   └── README.md                 # Datasheet resumido, esquema y gobernanza
├── src/                          # Código modular de producción (cero lógica en notebooks)
│   ├── config.py                 # Semilla global (SEED=42), rutas y umbrales Hito 1
│   ├── data/
│   │   ├── download.py           # Descarga automática y manifiesto SHA-256
│   │   └── loader.py             # Carga float32/int32, muestreo estratificado y splits
│   ├── features/
│   │   └── preprocessing.py      # Transformador de centinelas negativos y ColumnTransformer
│   ├── models/
│   │   ├── pipelines.py          # Baseline (class_weight) e imblearn Pipeline (SMOTE in-fold)
│   │   └── train.py              # Validación cruzada 5-fold, FLAML (max_iter) y MLflow
│   ├── evaluation/
│   │   └── metrics.py            # AUC-PR, F1, ROC-AUC, Recall@FPR<=5% y equidad por subgrupo
│   └── app/
│       └── streamlit_app.py      # Prototipo funcional interactivo para sustentación
├── notebooks/                    # Exclusivo para análisis exploratorio (EDA)
├── experiments/                  # Configuraciones seed=42, JSON verificables y MLflow runs
└── docs/
    ├── hito1_propuesta.md        # Propuesta formal de 1 página (Hito 1)
    ├── hito1_propuesta.tex       # Fuente LaTeX de 1 página para exportar PDF del Hito 1
    └── bitacora_ia.md            # Bitácora auditable de interacción humano-IA
```

---

## Reglas de Ingeniería y Calidad (CRISP-ML(Q))

1. **Reproducibilidad estricta:** Semilla fija `SEED = 42` en particiones, validación cruzada estratificada (`n_splits=5`), `SMOTE` y `FLAML` controlado por `max_iter` (sin `time_budget` dependiente del hardware).
2. **Eficiencia computacional:** Conversión explícita a `float32`/`int32`/`int8`/`category` (~65% menos RAM) y muestreo estratificado configurable (`50.000`–`100.000` registros preservando el ~1.10% de fraude).
3. **Cero fuga de información (*Zero Data Leakage*):** Imputación, escalado, codificación y sobremuestreo `SMOTE` (`imblearn.pipeline.Pipeline`) ejecutados estrictamente dentro de cada fold de entrenamiento.
4. **IA Responsable predeclarada:** Optimización primaria por `AUC-PR` (`ap`), control operativo `Recall@FPR<=5%` y auditoría de sesgos por edad (`customer_age`), estado laboral (`employment_status`) e ingresos (`income`).
