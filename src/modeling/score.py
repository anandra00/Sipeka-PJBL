"""Batch scoring & forecast 30 hari ke depan memakai model aktif di models/active_model.json.

Menghasilkan file siap pakai di `data/serving/` (CSV + JSON) untuk Dashboard
Streamlit / Laravel, serta opsi penulisan ke PostgreSQL via `--postgres`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from config.settings import DATA_DIR, DATA_PROCESSED_DIR, DOCS_DIR, get_settings
from src.modeling.train import (
    FITUR_HISTORIS,
    KOLOM_TANGGAL,
    TARGET,
    baca_active_model,
    baca_parquet_fitur,
    muat_pipeline_model,
)
from src.utils.spark_session import get_spark

SERVING_DIR = DATA_DIR / "serving"
BULAN_MUSIM_HUJAN = [10, 11, 12, 1, 2, 3, 4]


def tambah_fitur_kalender(df: DataFrame, kolom_tanggal: str = KOLOM_TANGGAL) -> DataFrame:
    pi = math.pi
    bulan = F.month(kolom_tanggal)
    dow = F.dayofweek(kolom_tanggal)
    return df.withColumns(
        {
            "tahun": F.year(kolom_tanggal).cast("double"),
            "bulan": bulan.cast("double"),
            "hari_dalam_bulan": F.dayofmonth(kolom_tanggal).cast("double"),
            "hari_dalam_minggu": dow.cast("double"),
            "is_weekend": F.when(dow.isin(1, 7), 1.0).otherwise(0.0),
            "minggu_ke": F.ceil(F.dayofmonth(kolom_tanggal) / 7).cast("double"),
            "kuartal": F.quarter(kolom_tanggal).cast("double"),
            "hari_ke_dalam_tahun": F.dayofyear(kolom_tanggal).cast("double"),
            "bulan_sin": F.sin(2 * pi * (bulan - 1) / 12),
            "bulan_cos": F.cos(2 * pi * (bulan - 1) / 12),
            "hari_sin": F.sin(2 * pi * (dow - 1) / 7),
            "hari_cos": F.cos(2 * pi * (dow - 1) / 7),
            "is_musim_hujan": F.when(bulan.isin(BULAN_MUSIM_HUJAN), 1.0).otherwise(0.0),
            "is_hari_libur": F.lit(0.0),
        }
    )


def jalankan_scoring(horizon_hari: int = 30, tulis_postgres: bool = False) -> dict:
    settings = get_settings()
    settings.pastikan_folder_lokal()
    SERVING_DIR.mkdir(parents=True, exist_ok=True)

    active = baca_active_model()
    if active is None:
        raise FileNotFoundError("Belum ada model aktif di models/. Jalankan train.py dulu.")

    versi_model = active["versi_model"]
    versi_fitur = active["versi_fitur"]
    fitur_path = DATA_PROCESSED_DIR / f"features_{versi_fitur}"

    spark = get_spark("SiPeka-Batch-Scoring")
    try:
        model = muat_pipeline_model(spark, active["model_path"])
        df = baca_parquet_fitur(spark, fitur_path, versi_fitur)

        # 1. Prediksi pada partisi test terbaru
        tgl_valid_akhir = active.get("skema_split", {}).get("tanggal_valid_akhir")
        if tgl_valid_akhir:
            df_test = df.filter(F.col(KOLOM_TANGGAL) > F.lit(tgl_valid_akhir).cast("date"))
        else:
            df_test = df.filter(F.col("tahun") == df.agg(F.max("tahun")).first()[0])

        pred_test = (
            model.transform(df_test)
            .withColumn("y_pred", F.round(F.greatest(F.lit(0.0), F.col("prediction")), 4))
            .select(
                F.col(KOLOM_TANGGAL).alias("tanggal"),
                "titik_lokasi",
                "kecamatan",
                "wilayah",
                F.col(TARGET).alias("y_true"),
                "y_pred",
                F.lit(versi_model).alias("model_version"),
            )
        )

        # 2. Forecast `horizon_hari` hari ke depan menggunakan profil historis normal per titik
        df_ref = df.filter(F.col("is_outlier") == 0) if "is_outlier" in df.columns else df
        max_tgl = df.agg(F.max(KOLOM_TANGGAL)).first()[0]

        jendela = Window.partitionBy("titik_lokasi").orderBy(
            F.desc("jumlah"), F.asc("kecamatan"), F.asc("wilayah")
        )
        lookup_lokasi = (
            df_ref.groupBy("titik_lokasi", "kecamatan", "wilayah")
            .agg(F.count("*").alias("jumlah"))
            .withColumn("rk", F.row_number().over(jendela))
            .filter(F.col("rk") == 1)
            .select("titik_lokasi", "kecamatan", "wilayah")
        )

        # Bangun daftar tanggal masa depan murni di JVM (tanpa Python worker)
        df_tanggal = spark.range(1, horizon_hari + 1).select(
            F.date_add(F.lit(max_tgl), F.col("id").cast("int")).alias(KOLOM_TANGGAL)
        )
        df_future = lookup_lokasi.crossJoin(df_tanggal)
        df_future = tambah_fitur_kalender(df_future)

        if versi_fitur == "v2":
            hist_per_titik = df_ref.groupBy("titik_lokasi").agg(
                *[F.mean(c).alias(c) for c in FITUR_HISTORIS]
            )
            rata_global = (
                df_ref.agg(*[F.mean(c).alias(c) for c in FITUR_HISTORIS]).first().asDict()
            )
            rata_global = {k: (v if v is not None else 0.0) for k, v in rata_global.items()}
            hist_per_titik = hist_per_titik.fillna(rata_global)
            df_future = df_future.join(hist_per_titik, on="titik_lokasi", how="left")

        forecast_df = (
            model.transform(df_future)
            .withColumn("y_pred", F.round(F.greatest(F.lit(0.0), F.col("prediction")), 4))
            .select(
                F.col(KOLOM_TANGGAL).alias("tanggal"),
                "titik_lokasi",
                "kecamatan",
                "wilayah",
                "y_pred",
                F.lit(versi_model).alias("model_version"),
            )
        )

        # Agregat ringan untuk konsumsi langsung Laravel / Streamlit
        pred_wilayah_pd = (
            pred_test.groupBy("tanggal", "wilayah")
            .agg(
                F.round(F.sum("y_true"), 2).alias("total_aktual_m3"),
                F.round(F.sum("y_pred"), 2).alias("total_prediksi_m3"),
                F.count("*").alias("jumlah_titik"),
            )
            .orderBy("tanggal", "wilayah")
            .toPandas()
        )
        forecast_wilayah_pd = (
            forecast_df.groupBy("tanggal", "wilayah")
            .agg(
                F.round(F.sum("y_pred"), 2).alias("total_prediksi_m3"),
                F.count("*").alias("jumlah_titik"),
            )
            .orderBy("tanggal", "wilayah")
            .toPandas()
        )
        forecast_titik_pd = (
            forecast_df.groupBy("titik_lokasi", "kecamatan", "wilayah")
            .agg(
                F.round(F.mean("y_pred"), 3).alias("rata_prediksi_harian_m3"),
                F.round(F.sum("y_pred"), 2).alias(f"total_prediksi_{horizon_hari}hari_m3"),
            )
            .orderBy(F.desc("rata_prediksi_harian_m3"))
            .toPandas()
        )

        # Gabungkan metadata GIS dari dataset_titik_phb-v3.json (menggantikan titik_phb_silika.json lama)
        from src.modeling.export_model import ekspor_model_aktif, muat_dataset_titik_v3

        _, daftar_v3, _ = muat_dataset_titik_v3()
        if daftar_v3:
            df_geo = pd.DataFrame(daftar_v3)[
                [
                    "titik_lokasi",
                    "id",
                    "lps",
                    "lat",
                    "lng",
                    "kel",
                    "status_lahan",
                    "luas_lahan",
                    "truk",
                    "alat_berat",
                    "nama_koordinator",
                    "kat",
                    "foto_url",
                    "marker_icon",
                ]
            ].rename(columns={"id": "id_phb", "lat": "latitude", "lng": "longitude", "kel": "kelurahan"})
            df_geo = df_geo.drop_duplicates(subset=["titik_lokasi"])
            forecast_titik_pd = forecast_titik_pd.merge(df_geo, on="titik_lokasi", how="left")

        pred_wilayah_pd.to_csv(SERVING_DIR / "predictions_harian_wilayah.csv", index=False)
        forecast_wilayah_pd.to_csv(SERVING_DIR / "forecast_harian_wilayah.csv", index=False)
        forecast_titik_pd.to_csv(SERVING_DIR / "forecast_per_titik.csv", index=False)

        # Simpan juga full forecast_future Parquet untuk query detail per tanggal & titik
        forecast_df.write.mode("overwrite").parquet(
            (SERVING_DIR / "forecast_future_parquet").resolve().as_uri()
        )

        if tulis_postgres:
            pred_test.write.jdbc(
                url=settings.jdbc_url,
                table="predictions",
                mode="overwrite",
                properties=settings.jdbc_properties,
            )
            forecast_df.write.jdbc(
                url=settings.jdbc_url,
                table="forecast_future",
                mode="overwrite",
                properties=settings.jdbc_properties,
            )
            print("[Scoring] Tabel PostgreSQL predictions & forecast_future diperbarui.")

        # Ekspor/sinkronkan juga active_model.pkl, active_model.h5, dan titik_phb_v3_forecast.json
        hasil_ekspor = ekspor_model_aktif(spark_existing=spark)

        ringkasan = {
            "model_version": versi_model,
            "model": active["model"],
            "versi_fitur": versi_fitur,
            "scored_at": datetime.now().isoformat(timespec="seconds"),
            "horizon_hari": horizon_hari,
            "forecast_mulai": (max_tgl + timedelta(days=1)).isoformat(),
            "forecast_akhir": (max_tgl + timedelta(days=horizon_hari)).isoformat(),
            "jumlah_baris_test": int(pred_test.count()),
            "jumlah_titik_forecast": int(len(forecast_titik_pd)),
            "jumlah_titik_berkoordinat": int(forecast_titik_pd["latitude"].notna().sum())
            if "latitude" in forecast_titik_pd.columns
            else 0,
            "jumlah_entri_phb_v3": hasil_ekspor["jumlah_entri_v3"],
            "pkl_path": hasil_ekspor["pkl_active"],
            "h5_path": hasil_ekspor["h5_active"],
            "titik_v3_forecast_path": hasil_ekspor["titik_v3_forecast"],
            "metrik_validation": active.get("metrik_validation", {}),
            "metrik_test": active.get("metrik_test", {}),
        }
        (SERVING_DIR / "active_model_summary.json").write_text(
            json.dumps(ringkasan, indent=2), encoding="utf-8"
        )
        print(f"[Scoring] Selesai -> {SERVING_DIR}")
        print(json.dumps(ringkasan, indent=2))
        return ringkasan
    finally:
        spark.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch scoring & forecast SiPeka")
    parser.add_argument("--horizon", type=int, default=30, help="Jumlah hari forecast ke depan")
    parser.add_argument("--postgres", action="store_true", help="Tulis juga ke PostgreSQL")
    args = parser.parse_args()
    jalankan_scoring(horizon_hari=args.horizon, tulis_postgres=args.postgres)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
