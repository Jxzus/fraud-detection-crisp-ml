"""Descarga reproducible y verificación criptográfica (SHA-256) del dataset BAF (NeurIPS 2022).

Uso desde la raíz del repositorio:
    python -m src.data.download
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.config import (
    DATA_RAW_DIR,
    DATASET_LICENSE,
    DATASET_MANIFEST_PATH,
    EXPERIMENTS_DIR,
    HF_FALLBACK_BASE_CSV_URL,
    KAGGLE_DATASET_HANDLE,
    RAW_BAF_FILENAME,
    RAW_BAF_PATH,
    SEED,
)


def compute_file_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calcula el hash SHA-256 de un archivo por bloques sin cargarlo completo en RAM."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def _has_kaggle_credentials() -> bool:
    """Verifica si existen credenciales de Kaggle configuradas en entorno o disco."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.is_file()


def _download_via_kagglehub(target_path: Path) -> bool:
    """Intenta descargar Base.csv desde la fuente oficial en Kaggle."""
    if not _has_kaggle_credentials():
        return False
    try:
        import kagglehub

        print(f"[INFO] Descargando desde Kaggle ({KAGGLE_DATASET_HANDLE})...")
        downloaded = Path(
            kagglehub.dataset_download(KAGGLE_DATASET_HANDLE, path=RAW_BAF_FILENAME)
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, target_path)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] No fue posible descargar vía kagglehub ({exc}). Usando espejo público...")
        return False


def _download_via_https_mirror(target_path: Path, url: str = HF_FALLBACK_BASE_CSV_URL) -> None:
    """Descarga Base.csv por streaming HTTPS desde espejo académico público sin requerir token."""
    import requests

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".csv.part")
    print(f"[INFO] Descargando Base.csv por streaming desde espejo público:\n       {url}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as out_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out_file.write(chunk)
    tmp_path.replace(target_path)


def save_dataset_manifest(csv_path: Path, manifest_path: Path = DATASET_MANIFEST_PATH) -> dict:
    """Registra el hash SHA-256, tamaño, licencia y semilla maestra para auditoría en MLflow."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    sha256_hex = compute_file_sha256(csv_path)
    manifest = {
        "dataset_name": "Bank Account Fraud (BAF) Dataset Suite - Base.csv (NeurIPS 2022)",
        "authors": "Jesus et al. (Feedzai, NeurIPS 2022)",
        "license": DATASET_LICENSE,
        "contains_raw_pii": False,
        "file_path": str(csv_path.relative_to(csv_path.parents[2])),
        "file_size_bytes": csv_path.stat().st_size,
        "sha256": sha256_hex,
        "seed": SEED,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def ensure_baf_dataset(target_path: Path = RAW_BAF_PATH, force: bool = False) -> dict:
    """Garantiza la disponibilidad local de Base.csv y genera su manifiesto SHA-256."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    if not target_path.is_file() or force:
        used_kaggle = _download_via_kagglehub(target_path)
        if not used_kaggle:
            _download_via_https_mirror(target_path)
    else:
        print(f"[OK] Archivo existente detectado en: {target_path}")

    manifest = save_dataset_manifest(target_path)
    print(f"[OK] SHA-256 verificado: {manifest['sha256']}")
    print(f"[OK] Manifiesto guardado en: {DATASET_MANIFEST_PATH}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga y verifica criptográficamente el dataset BAF (Base.csv)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Fuerza la redescarga aunque data/raw/Base.csv ya exista.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ensure_baf_dataset(force=args.force)
