"""Skrip verifikasi 4 koneksi SiPeka: Spark, HDFS, PostgreSQL, dan JDBC."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("PYTHONPATH", str(Path(__file__).resolve().parent.parent)))

import psycopg2
from config.settings import get_settings
from src.utils.spark_session import get_spark, tampilkan_info


def main() -> int:
    settings = get_settings()
    lulus = 0

    # 1. SparkSession
    spark = get_spark("SiPeka-Test-Connections")
    tampilkan_info(spark)
    print("[1/4] SparkSession OK")
    lulus += 1

    # 2. HDFS tulis & baca
    hdfs_test_path = f"{settings.hdfs_namenode}/sipeka/_test_koneksi"
    df_uji = spark.createDataFrame([(1, "ok")], ["id", "status"])
    df_uji.write.mode("overwrite").parquet(hdfs_test_path)
    jumlah = spark.read.parquet(hdfs_test_path).count()
    assert jumlah == 1, f"Jumlah baris HDFS tidak sesuai: {jumlah}"
    print(f"[2/4] HDFS OK ({hdfs_test_path})")
    lulus += 1

    # 3. PostgreSQL (psycopg2)
    with psycopg2.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database(), now();")
            db_info = cur.fetchone()
    print(f"[3/4] PostgreSQL OK ({db_info})")
    lulus += 1

    # 4. PySpark -> PostgreSQL via JDBC
    df_jdbc = spark.createDataFrame(
        [("pyspark", "Tes koneksi Step 1 berhasil")],
        ["sumber", "pesan"],
    )
    df_jdbc.write.jdbc(
        url=settings.jdbc_url,
        table="staging.koneksi_test",
        mode="append",
        properties=settings.jdbc_properties,
    )
    df_baca = spark.read.jdbc(
        url=settings.jdbc_url,
        table="staging.koneksi_test",
        properties=settings.jdbc_properties,
    )
    print(f"[4/4] JDBC PostgreSQL OK (total baris staging.koneksi_test = {df_baca.count()})")
    lulus += 1

    spark.stop()
    print(f"\nSemua tes koneksi lulus ({lulus}/4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
