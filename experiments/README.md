# Trazabilidad Experimental y Aseguramiento de Calidad (`experiments/`)

Este directorio almacena las evidencias verificables que respaldan cada tabla y figura del informe final en formato IEEE:
- `seed_config.json`: Registro predeclarado de la semilla maestra (`seed=42`), particiones, estimadores de FLAML y umbrales de éxito fijados en el Hito 1.
- `dataset_manifest.json`: Hash criptográfico `SHA-256` del archivo `data/raw/Base.csv` verificado antes de cada corrida.
- `latest_baseline_results.json` / `latest_full_results.json`: Métricas de validación cruzada estratificada (`media`, `std`, `var`), prueba reservada (`AUC-PR`, `F1`, `ROC-AUC`, `Recall@FPR<=5%`, matriz de confusión) y desglose de equidad por subgrupos (`customer_age`, `employment_status`, `income`).
- `mlruns/`: Almacén local de corridas de **MLflow**, incluyendo parámetros, hash del dataset y un *child run* anidado por cada familia de estimador evaluada en FLAML (`lgbm`, `xgboost`, `rf`, `extra_tree`, `lrl1`).
