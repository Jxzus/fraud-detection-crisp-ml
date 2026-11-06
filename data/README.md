# Gobernanza de Datos — Bank Account Fraud (BAF) Dataset Suite (`Base.csv`)

## 1. Fuente Oficial y Referencia Científica
- **Dataset:** *Bank Account Fraud (BAF) Dataset Suite* — variante `Base.csv` (1.000.000 de solicitudes de apertura de cuenta bancaria, 32 columnas, prevalencia real de fraude ≈ `1.10%`).
- **Autores y Publicación:** S. Jesus, J. Pombal, D. Alves, A. Cruz, P. Saleiro, R. P. Ribeiro, J. Gama, P. Bizarro, *"Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation"*, **Advances in Neural Information Processing Systems (NeurIPS 2022)**.
- **Repositorios oficiales:**
  - Kaggle: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
  - GitHub (Datasheet y documentación de Feedzai Research): https://github.com/feedzai/bank-account-fraud

## 2. Licencia y Privacidad (Cero PII sin Anonimizar)
- **Licencia:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (**CC BY-NC-SA 4.0**), apta para investigación y uso académico citando a los autores originales.
- **Protección de datos personales:** El conjunto fue generado por Feedzai aplicando técnicas de síntesis generativa con preservación de privacidad (CTGAN + ruido diferencial y codificación de categorías) sobre registros reales de *onboarding* bancario. **No contiene información personal identificable (PII) real de ningún ciudadano.**

## 3. Atributos Clave y Auditoría de Subgrupos (IA Responsable)
- **Variable objetivo (`int8`):** `fraud_bool` (`1` = solicitud fraudulenta por suplantación/identidad sintética; `0` = solicitud legítima).
- **Variable temporal (`int8`):** `month` (`0` a `7`, ocho meses consecutivos).
- **Atributos auditados por subgrupo (Criterio 7):**
  - `customer_age`: Edad del solicitante en décadas (`>= 50` vs. `< 50` años, siguiendo el protocolo de equidad de Jesus et al., NeurIPS 2022).
  - `employment_status`: Situación laboral codificada (`CA`–`CG`).
  - `income`: Cuantil de ingreso anual normalizado en `[0.1, 0.9]`.
- **Valores faltantes codificados como centinelas negativos (`< 0`):** `prev_address_months_count`, `current_address_months_count`, `bank_months_count`, `intended_balcon_amount`, `session_length_in_minutes` y `device_distinct_emails_8w`. El transformador `NegativeSentinelToNaN` en `src/features/preprocessing.py` los convierte a `NaN` para imputarlos exclusivamente dentro de cada fold de entrenamiento.

## 4. Descarga Reproducible y Eficiencia en Memoria
Ejecute desde la raíz del proyecto:
```bash
python -m src.data.download
```
El cargador `src/data/loader.py` emplea tipos compactos (`float32`, `int32`, `int8`, `category`) reduciendo el consumo de RAM en ~65% y soporta muestreo estratificado reproducible (`sample_size=100000`, `seed=42`) manteniendo intacta la proporción de fraude (~1.10%).
