# Sipeka-PJBL

**SiPeka — Sistem Prediksi Volume Pengangkutan Sampah DKI Jakarta**

Proyek Big Data untuk memprediksi volume pengangkutan sampah harian (`m³`) di kali/sungai/situ/waduk DKI Jakarta menggunakan PySpark, Spark MLlib, dan Plotly.

## Struktur Direktori

- `config/` — konfigurasi path, HDFS, dan PostgreSQL (`config/settings.py`)
- `docker/` — `Dockerfile.jupyter`, `spark-defaults.conf`, dan skrip init PostgreSQL (`docker/initdb/01_init.sql`)
- `scripts/` — skrip tes koneksi (`scripts/test_connections.py`)
- `src/utils/` — factory `SparkSession` (`src/utils/spark_session.py`)
- `src/ingestion/` — ingestion batch Step 3 (`src/ingestion/ingest.py`)
- `src/processing/` — cleaning & feature engineering Step 4 (`src/processing/clean_features.py`)
- `notebooks/` — Step 5 EDA (`02_eda.ipynb`) dan Step 6 Modeling & Evaluasi (`03_modeling.ipynb`)
- `models/` — artefak model terbaik, metadata JSON, dan tabel metrik CSV
- `docs/figures/` — grafik interaktif Plotly (`.html` dan `.json`)
- `PRD/` — dokumen panduan proyek

## Menjalankan Pipeline (Step 3–6)

1. Tempatkan dataset `.xls` (2018, 2020, 2021) di folder `Dataset/`.
2. Salin `.env.example` menjadi `.env`.
3. Jalankan ingestion & feature engineering:
   ```bash
   python src/ingestion/ingest.py
   python src/processing/clean_features.py
   ```
4. Buka dan jalankan `notebooks/02_eda.ipynb` serta `notebooks/03_modeling.ipynb`.
