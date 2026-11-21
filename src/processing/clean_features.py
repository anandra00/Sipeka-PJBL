"""Step 4 — Cleaning & Feature Engineering SiPeka (PySpark).

Membaca Parquet raw tervalidasi hasil Step 3, membersihkan teks & duplikat
(titik_lokasi, tanggal), menandai outlier tanpa menghapus diam-diam, membangun
fitur kalender (v1) dan fitur historis bebas-leakage (v2), serta menyimpan hasil
ke Parquet lokal (dan opsional HDFS / PostgreSQL).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("PYTHONPATH", str(Path(__file__).resolve().parent.parent.parent)))

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from config.settings import get_settings
from src.utils.spark_session import get_spark

BULAN_MUSIM_HUJAN = [10, 11, 12, 1, 2, 3, 4]

# Hari libur nasional resmi Indonesia (SKB 3 Menteri) untuk 2018, 2020, 2021.
HARI_LIBUR_NASIONAL = sorted(
    {
        # 2018
        "2018-01-01", "2018-02-16", "2018-03-17", "2018-03-30", "2018-04-14",
        "2018-05-01", "2018-05-10", "2018-05-29", "2018-06-01", "2018-06-15",
        "2018-06-16", "2018-08-17", "2018-08-22", "2018-09-11", "2018-11-20",
        "2018-12-25",
        # 2020
        "2020-01-01", "2020-01-25", "2020-03-22", "2020-03-25", "2020-04-10",
        "2020-05-01", "2020-05-07", "2020-05-21", "2020-05-24", "2020-05-25",
        "2020-06-01", "2020-07-31", "2020-08-17", "2020-08-20", "2020-10-29",
        "2020-12-25",
        # 2021
        "2021-01-01", "2021-02-12", "2021-03-11", "2021-03-14", "2021-04-02",
        "2021-05-01", "2021-05-13", "2021-05-14", "2021-05-26", "2021-06-01",
        "2021-07-20", "2021-08-11", "2021-08-17", "2021-10-20", "2021-12-25",
    }
)


def normalisasi_teks(
    kolom: str, fallback: str = "TIDAK DIKETAHUI", rapikan_tanda_baca: bool = False
) -> F.Column:
    """Trim, rapikan spasi ganda, huruf kapital seragam, ganti '-' atau kosong."""
    teks = F.upper(F.trim(F.col(kolom)))
    if rapikan_tanda_baca:
        # Seragamkan variasi ',' -> '.' (mis. 'JLN,' vs 'JLN.') dan hapus titik awal.
        teks = F.regexp_replace(teks, r",", ".")
        teks = F.regexp_replace(teks, r"^[\s.,;:-]+", "")
    bersih = F.trim(F.regexp_replace(teks, r"\s+", " "))
    return (
        F.when(bersih.isNull() | (bersih == "") | (bersih == "-"), F.lit(fallback))
        .otherwise(bersih)
    )


def bersihkan_data(df_raw: DataFrame) -> tuple[DataFrame, dict]:
    """Bersihkan data raw dan hasilkan statistik kualitas data sebelum/sesudah."""
    n_sebelum = df_raw.count()
    n_vol_null = df_raw.filter(F.col("volume_sampah_perhari").isNull()).count()
    n_vol_negatif = df_raw.filter(F.col("volume_sampah_perhari") < 0).count()

    df_norm = df_raw.withColumns(
        {
            "titik_lokasi": normalisasi_teks("titik_lokasi", fallback="", rapikan_tanda_baca=True),
            "kecamatan": normalisasi_teks("kecamatan"),
            "wilayah": normalisasi_teks("wilayah"),
        }
    ).filter(
        (F.col("titik_lokasi") != "")
        & F.col("tanggal").isNotNull()
        & F.col("volume_sampah_perhari").isNotNull()
        & (F.col("volume_sampah_perhari") >= 0)
    )

    n_setelah_filter = df_norm.count()

    # Tangani duplikat (titik_lokasi, tanggal): jumlahkan volume harian per titik
    # karena satu titik bisa memiliki lebih dari satu entri pengangkutan per hari.
    df_dedup = (
        df_norm.groupBy("titik_lokasi", "tanggal")
        .agg(
            F.first("kecamatan", ignorenulls=True).alias("kecamatan"),
            F.first("wilayah", ignorenulls=True).alias("wilayah"),
            F.sum("volume_sampah_perhari").alias("volume_sampah_perhari"),
        )
    )
    n_setelah_dedup = df_dedup.count()

    # Tandai outlier ekstrem (persentil 99.9) tanpa menghapus baris.
    p999 = float(df_dedup.approxQuantile("volume_sampah_perhari", [0.999], 1e-5)[0])
    df_bersih = df_dedup.withColumn(
        "is_outlier",
        F.when(F.col("volume_sampah_perhari") > F.lit(p999), F.lit(1)).otherwise(F.lit(0)),
    ).withColumn("ambang_p999", F.lit(p999))
    n_outlier = df_bersih.filter(F.col("is_outlier") == 1).count()

    laporan = {
        "baris_raw_valid": n_sebelum,
        "volume_null": n_vol_null,
        "volume_negatif": n_vol_negatif,
        "baris_setelah_filter_target": n_setelah_filter,
        "baris_setelah_dedup_harian": n_setelah_dedup,
        "ambang_outlier_p999": round(p999, 4),
        "jumlah_outlier": n_outlier,
        "aturan_duplikat": "sum(volume_sampah_perhari) per (titik_lokasi, tanggal)",
    }
    return df_bersih.drop("ambang_p999"), laporan


def tambah_fitur_kalender(df: DataFrame) -> DataFrame:
    """Bangun 14 fitur kalender sesuai kontrak Step 6 (03_modeling.ipynb)."""
    pi = math.pi
    bulan = F.month("tanggal")
    dow = F.dayofweek("tanggal")  # 1 = Minggu, 7 = Sabtu

    return df.withColumns(
        {
            "tahun": F.year("tanggal"),
            "bulan": bulan,
            "hari_dalam_bulan": F.dayofmonth("tanggal"),
            "hari_dalam_minggu": dow,
            "is_weekend": F.when(dow.isin(1, 7), 1).otherwise(0),
            "minggu_ke": F.ceil(F.dayofmonth("tanggal") / 7).cast("int"),
            "kuartal": F.quarter("tanggal"),
            "hari_ke_dalam_tahun": F.dayofyear("tanggal"),
            "bulan_sin": F.sin(2 * pi * (bulan - 1) / 12),
            "bulan_cos": F.cos(2 * pi * (bulan - 1) / 12),
            "hari_sin": F.sin(2 * pi * (dow - 1) / 7),
            "hari_cos": F.cos(2 * pi * (dow - 1) / 7),
            "is_musim_hujan": F.when(bulan.isin(BULAN_MUSIM_HUJAN), 1).otherwise(0),
            "is_hari_libur": F.when(
                F.date_format("tanggal", "yyyy-MM-dd").isin(*HARI_LIBUR_NASIONAL), 1
            ).otherwise(0),
        }
    )


def tambah_fitur_historis(df_v1: DataFrame, ambang_p999: float) -> DataFrame:
    """Bangun fitur lag, rolling, & statistik historis titik TANPA leakage.

    Seluruh fitur dihitung dari nilai yang di-winsorize pada P99.9 dan hanya
    memakai observasi sebelum tanggal target (upper bound = -1).
    """
    w_urut = Window.partitionBy("titik_lokasi").orderBy("tanggal")
    vol_wins = F.least(F.col("volume_sampah_perhari"), F.lit(ambang_p999))
    is_nol = F.when(F.col("volume_sampah_perhari") == 0, F.lit(1.0)).otherwise(F.lit(0.0))

    df_tmp = df_v1.withColumns({"_vol_w": vol_wins, "_is_nol": is_nol})
    df_fitur = df_tmp.withColumns(
        {
            "lag_1": F.lag("_vol_w", 1).over(w_urut),
            "lag_7": F.lag("_vol_w", 7).over(w_urut),
            "lag_14": F.lag("_vol_w", 14).over(w_urut),
            "rolling_mean_7": F.avg("_vol_w").over(w_urut.rowsBetween(-7, -1)),
            "rolling_mean_14": F.avg("_vol_w").over(w_urut.rowsBetween(-14, -1)),
            "rolling_mean_30": F.avg("_vol_w").over(w_urut.rowsBetween(-30, -1)),
            "rolling_std_7": F.coalesce(
                F.stddev_samp("_vol_w").over(w_urut.rowsBetween(-7, -1)), F.lit(0.0)
            ),
            "rolling_max_7": F.max("_vol_w").over(w_urut.rowsBetween(-7, -1)),
            "titik_hist_mean": F.avg("_vol_w").over(
                w_urut.rowsBetween(Window.unboundedPreceding, -1)
            ),
            "titik_hist_zero_rate": F.avg("_is_nol").over(
                w_urut.rowsBetween(Window.unboundedPreceding, -1)
            ),
        }
    )
    return df_fitur.drop("_vol_w", "_is_nol")


def main() -> int:
    parser = argparse.ArgumentParser(description="Cleaning & feature engineering SiPeka")
    parser.add_argument("--hdfs", action="store_true", help="Baca/tulis Parquet lewat HDFS")
    parser.add_argument(
        "--postgres",
        action="store_true",
        help="Simpan juga tabel agregat ke PostgreSQL via JDBC",
    )
    args = parser.parse_args()

    settings = get_settings()
    settings.pastikan_folder_lokal()

    spark = get_spark("SiPeka-Step4-Features")
    raw_uri = (
        f"{settings.hdfs_namenode}/sipeka/raw/parquet"
        if args.hdfs
        else (settings.data_raw_dir / "parquet").resolve().as_uri()
    )
    v1_uri = (
        f"{settings.hdfs_namenode}/sipeka/processed/features_v1"
        if args.hdfs
        else (settings.data_processed_dir / "features_v1").resolve().as_uri()
    )
    v2_uri = (
        f"{settings.hdfs_namenode}/sipeka/processed/features_v2"
        if args.hdfs
        else (settings.data_processed_dir / "features_v2").resolve().as_uri()
    )

    df_raw = spark.read.parquet(raw_uri)
    df_bersih, laporan = bersihkan_data(df_raw)
    print("Data quality report:", json.dumps(laporan, indent=2))

    df_v1 = tambah_fitur_kalender(df_bersih).cache()
    df_v2 = tambah_fitur_historis(df_v1, float(laporan["ambang_outlier_p999"]))

    df_v1.write.mode("overwrite").partitionBy("tahun").parquet(v1_uri)
    df_v2.write.mode("overwrite").partitionBy("tahun").parquet(v2_uri)
    print(f"Tersimpan features_v1 -> {v1_uri}")
    print(f"Tersimpan features_v2 -> {v2_uri}")

    if args.postgres:
        agg_harian_wilayah = df_v1.groupBy("tanggal", "tahun", "bulan", "wilayah").agg(
            F.sum("volume_sampah_perhari").alias("total_volume"),
            F.mean("volume_sampah_perhari").alias("rata_volume"),
            F.count("*").alias("jumlah_titik"),
        )
        agg_bulanan_kecamatan = df_v1.groupBy("tahun", "bulan", "wilayah", "kecamatan").agg(
            F.sum("volume_sampah_perhari").alias("total_volume"),
            F.mean("volume_sampah_perhari").alias("rata_volume"),
            F.countDistinct("titik_lokasi").alias("jumlah_titik"),
        )
        agg_titik = df_v1.groupBy("titik_lokasi", "kecamatan", "wilayah").agg(
            F.sum("volume_sampah_perhari").alias("total_volume"),
            F.mean("volume_sampah_perhari").alias("rata_volume"),
            F.count("*").alias("jumlah_hari"),
        )
        for nama_tabel, df_agg in (
            ("agg_harian_wilayah", agg_harian_wilayah),
            ("agg_bulanan_kecamatan", agg_bulanan_kecamatan),
            ("agg_titik", agg_titik),
        ):
            df_agg.write.jdbc(
                url=settings.jdbc_url,
                table=nama_tabel,
                mode="overwrite",
                properties=settings.jdbc_properties,
            )
            print(f"Tersimpan tabel PostgreSQL -> {nama_tabel}")

    df_v1.unpersist()

    report_path = settings.data_processed_dir / "data_quality_report.json"
    report_path.write_text(json.dumps(laporan, indent=2), encoding="utf-8")
    print(f"Laporan kualitas data disimpan ke {report_path}")
    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
