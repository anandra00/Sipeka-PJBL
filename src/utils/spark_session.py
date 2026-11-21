"""Factory SparkSession untuk SiPeka (Docker & lokal Windows/Linux)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession


def _siapkan_hadoop_windows() -> None:
    """Set HADOOP_HOME dan PYSPARK_PYTHON otomatis di Windows."""
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    if os.name != "nt" or os.environ.get("HADOOP_HOME"):
        return
    kandidat = Path(__file__).resolve().parent.parent.parent / ".hadoop"
    if (kandidat / "bin" / "winutils.exe").exists():
        os.environ["HADOOP_HOME"] = str(kandidat)
        os.environ["hadoop.home.dir"] = str(kandidat)
        bin_dir = str(kandidat / "bin")
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"


def get_spark(app_name: str = "SiPeka") -> SparkSession:
    """Bangun atau ambil SparkSession dengan konfigurasi aman."""
    _siapkan_hadoop_windows()
    default_mem = "2g" if os.name == "nt" else "1g"
    driver_memory = os.environ.get("SPARK_DRIVER_MEMORY", default_mem)
    default_master = os.environ.get("SPARK_MASTER")
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.driver.memory", driver_memory)
        .config("spark.pyspark.python", os.environ["PYSPARK_PYTHON"])
        .config("spark.pyspark.driver.python", os.environ["PYSPARK_DRIVER_PYTHON"])
        .config(
            "spark.sql.shuffle.partitions",
            os.environ.get("SPARK_SHUFFLE_PARTITIONS", "8"),
        )
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.session.timeZone", "Asia/Jakarta")
    )
    if default_master:
        builder = builder.master(default_master)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(os.environ.get("SPARK_LOG_LEVEL", "WARN"))
    return spark


def tampilkan_info(spark: SparkSession) -> None:
    """Cetak ringkasan SparkSession aktif."""
    conf = spark.sparkContext.getConf()
    print(f"App Name       : {spark.sparkContext.appName}")
    print(f"Spark Version  : {spark.version}")
    print(f"Master         : {spark.sparkContext.master}")
    print(f"Driver Memory  : {conf.get('spark.driver.memory', 'default')}")
    print(f"Default FS     : {conf.get('spark.hadoop.fs.defaultFS', 'file:///')}")
