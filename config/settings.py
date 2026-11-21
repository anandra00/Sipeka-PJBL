"""Konfigurasi path, HDFS, dan koneksi PostgreSQL untuk SiPeka."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATASET_DIR = BASE_DIR / "Dataset"
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_QUARANTINE_DIR = DATA_DIR / "quarantine"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"
FIGURES_DIR = DOCS_DIR / "figures"


@dataclass(frozen=True)
class Settings:
    """Konfigurasi terpusat yang dapat di-override lewat environment variable."""

    base_dir: Path = BASE_DIR
    dataset_dir: Path = DATASET_DIR
    data_raw_dir: Path = DATA_RAW_DIR
    data_processed_dir: Path = DATA_PROCESSED_DIR
    data_quarantine_dir: Path = DATA_QUARANTINE_DIR
    models_dir: Path = MODELS_DIR
    figures_dir: Path = FIGURES_DIR

    hdfs_namenode: str = os.environ.get("HDFS_NAMENODE", "hdfs://namenode:9000")
    postgres_host: str = os.environ.get("POSTGRES_HOST", "postgres")
    postgres_port: int = int(os.environ.get("POSTGRES_PORT", "5432"))
    postgres_db: str = os.environ.get("POSTGRES_DB", "sipeka")
    postgres_user: str = os.environ.get("POSTGRES_USER", "sipeka")
    postgres_password: str = os.environ.get("POSTGRES_PASSWORD", "sipeka")

    @property
    def jdbc_url(self) -> str:
        return (
            f"jdbc:postgresql://{self.postgres_host}:{self.postgres_port}/"
            f"{self.postgres_db}"
        )

    @property
    def jdbc_properties(self) -> dict[str, str]:
        return {
            "user": self.postgres_user,
            "password": self.postgres_password,
            "driver": "org.postgresql.Driver",
        }

    def pastikan_folder_lokal(self) -> None:
        """Pastikan seluruh direktori output lokal tersedia."""
        for folder in (
            self.data_raw_dir,
            self.data_processed_dir,
            self.data_quarantine_dir,
            self.models_dir,
            self.figures_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Kembalikan instance Settings yang di-cache."""
    return Settings()
