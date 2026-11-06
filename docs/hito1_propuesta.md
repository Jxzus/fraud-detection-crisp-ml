# Hito 1 — Propuesta de Proyecto (1 Página)
**Título:** Detección de Fraude por Suplantación en Apertura Digital de Cuentas Bancarias mediante Aprendizaje Automático Auditable bajo CRISP-ML(Q)  
**Repositorio:** `fraud-detection-crisp-ml` | **Ruta:** Ruta C — Aprendizaje Automático (Russell & Norvig, caps. 19–21) | **Metodología:** CRISP-ML(Q) | **Semilla global:** `seed = 42`

---

### 1. Problema Real, Contexto Colombiano y Usuario Objetivo
La digitalización financiera en Colombia ha trasladado el riesgo de fraude hacia los procesos de vinculación no presencial (*onboarding* digital). Según los *Informes de Operaciones y Quejas* de la **Superintendencia Financiera de Colombia (SFC, 2024–2025)**, los canales digitales (Internet y aplicaciones móviles) concentran entre el **82,3% y el 82,8% del volumen total de operaciones** del sistema financiero y **más del 85% de las reclamaciones por fraude**, mientras que **Asobancaria** reporta un promedio cercano a **94 intentos de ciberataque por segundo** contra el sector, destacándose la apertura fraudulenta de cuentas de ahorro y productos de crédito mediante suplantación e identidades sintéticas ("cuentas mula").  
- **Usuario objetivo:** Analistas de prevención de fraude y motores de riesgo en tiempo real de bancos y *fintechs* en Colombia, que requieren bloquear solicitudes apócrifas en milisegundos sin rechazar injustificadamente (*falsos positivos*) a ciudadanos legítimos.

### 2. Formulación de IA (PEAS) y Definición Formal de la Tarea
- **PEAS del Agente Clasificador:** **P (Rendimiento):** Maximizar captura de fraude (`AUC-PR`, `Recall` sujeto a `FPR <= 5%`) y minimizar disparidad de falsos positivos entre grupos demográficos (`Delta FPR`). **E (Entorno):** Plataforma digital de apertura de cuentas; *parcialmente observable*, *estocástico*, *secuencial/dinámico* (sujeto a *concept drift* y atacantes adaptativos) y de *costos fuertemente asimétricos*. **A (Actuadores):** Aprobar apertura automática, derivar a validación reforzada (biometría/analista) o rechazar solicitud. **S (Sensores):** Vector tabular $\mathbf{x} \in \mathcal{X}$ de 31 atributos de identidad, historial financiero, sesión y dispositivo.
- **Tarea de aprendizaje:** Estimación supervisada inductiva de la probabilidad posterior $\hat{p}_\theta(y=1 \mid \mathbf{x}) \in [0,1]$ sobre la etiqueta binaria `fraud_bool` ($y \in \{0,1\}$), aplicando una regla de decisión sensible al costo $\hat{y} = \mathbb{I}[\hat{p}_\theta(y=1 \mid \mathbf{x}) \ge \tau]$ cuyo umbral operativo $\tau$ se calibra exclusivamente en validación.

### 3. Fuente de Datos, Licencia y Gobernanza sin *Data Leakage*
- **Origen y tamaño:** *Bank Account Fraud (BAF) Dataset Suite — `Base.csv`* (Jesus et al., **NeurIPS 2022**, Feedzai), con `1.000.000` de solicitudes a lo largo de 8 meses (`month` `0`–`7`), `32` columnas e incidencia real de fraude de **`~1,10%`** (desbalance cercano a `1:90`).
- **Licencia y privacidad:** **CC BY-NC-SA 4.0**. Generado con técnicas de síntesis preservadora de privacidad sobre registros bancarios reales (**cero PII sin anonimizar**). Sus variables son semánticamente interpretables (habilitando el diagnóstico causal de `>= 10` casos reales de error) e incluyen atributos demográficos explícitos (`customer_age`, `employment_status`, `income`) indispensables para evaluar IA Responsable.
- **Viabilidad en cómputo gratuito y partición:** Carga optimizada en `float32`/`int32`/`int8`/`category` (~65% menos RAM) y muestreo estratificado determinista configurable (`50.000`–`100.000` filas con `seed=42`) preservando la prevalencia exacta de fraude. Partición estratificada estricta **`70% Train / 15% Val / 15% Test`** antes de cualquier transformación.

### 4. Línea Base y Estrategia de Ingeniería de Modelos (Fase 3 CRISP-ML(Q))
1. **Línea Base Mínima:** `sklearn.pipeline.Pipeline` integrando conversión de centinelas negativos (`< 0 -> NaN`), imputación por mediana con indicador, escalado estándar, `OneHotEncoder` y **`LogisticRegression(class_weight="balanced", random_state=42)`**.
2. **Búsqueda CASH y Desbalance:** Evaluación sistemática con **FLAML** (`estimator_list=["lgbm", "xgboost", "rf", "extra_tree", "lrl1"]`, `metric="ap"`, `eval_method="cv"`, `n_splits=5`, `seed=42`) controlada por `max_iter` (sin `time_budget` para reproducibilidad entre máquinas). Se contrastará ponderación `class_weight` contra **SMOTE** ejecutado **exclusivamente dentro de cada fold** vía `imblearn.pipeline.Pipeline`, registrando cada corrida, hash `SHA-256` y *child runs* en **MLflow**.

### 5. Evaluación Predeclarada y Umbral de Éxito (Congelados antes de Experimentar)
| Dimensión | Métricas Predeclaradas | Umbral Cuantitativo de Éxito del Proyecto |
| :--- | :--- | :--- |
| **Desempeño Primario (Desbalance 1,1%)** | **`AUC-PR` (`ap` — *Average Precision*)** (media ± std en 5-fold CV y Test) | Superar el `AUC-PR` medio CV de la Regresión Logística balanceada en **`>= 15%` relativo** (o **`>= +0,05` puntos absolutos**). |
| **Operación Bancaria Secundaria** | `Recall@FPR<=5%`, `F1-score`, `ROC-AUC` y Matriz de Confusión | Alcanzar **`Recall >= 0,50`** sobre el conjunto de prueba manteniendo una tasa de falsos positivos **`FPR <= 5%`**. |
| **IA Responsable (Equidad por Subgrupo)** | `FPR`, `Recall` y `AUC-PR` por `customer_age` (`>=50` vs. `<50`), `employment_status` e `income` | Mantener brecha máxima de falsos positivos entre grupos etarios **`Delta FPR <= 0,05`** (o mitigar mediante calibración de umbrales por grupo). |

### 6. Referencias Base Iniciales
1. S. Russell & P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed., Pearson, 2020 (Caps. 19–21).
2. S. Studer et al., "Towards CRISP-ML(Q): A Machine Learning Process Model with Quality Assurance Methodology," *Mach. Learn. Knowl. Extr.*, vol. 3, no. 2, pp. 392–413, 2021.
3. S. Jesus et al., "Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation," in *Proc. NeurIPS*, 2022.
4. T. Saito & M. Rehmsmeier, "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets," *PLOS ONE*, vol. 10, no. 3, 2015.
5. C. Wang et al., "FLAML: A Fast and Lightweight AutoML Library," in *Proc. MLSys*, 2021. / Superintendencia Financiera de Colombia (SFC), *Informe de Operaciones y Quejas*, 2025.
