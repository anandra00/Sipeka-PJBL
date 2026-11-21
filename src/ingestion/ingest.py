"""Step 3 — Ingestion Batch SiPeka (HTML .xls -> CSV staging -> Parquet raw + quarantine).

Membaca file .xls ekspor Jakarta Open Data (format tabel HTML), menyeragamkan nama
kolom antar tahun, memvalidasi tanggal & volume di PySpark dengan schema eksplisit,
memisahkan baris bermasalah ke folder quarantine, dan menulis Parquet terpartisi
per tahun secara idempoten.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from html import unescape
from pathlib import Path

sys.path.insert(0, os.environ.get("PYTHONPATH", str(Path(__file__).resolve().parent.parent.parent)))

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from config.settings import get_settings
from src.utils.spark_session import get_spark

TD_PATTERN = re.compile(r"<td>(.*?)</td>", re.IGNORECASE)
TH_PATTERN = re.compile(r"<th>(.*?)</th>", re.IGNORECASE)
TAG_PATTERN = re.compile(r"<[^>]+>")

KOLOM_STANDAR = [
    "periode_data",
    "bulan",
    "titik_lokasi",
    "kecamatan",
    "wilayah",
    "panjang_luas",
    "satuan_panjang_luas",
    "tanggal",
    "volume_sampah_perhari",
]

ALIAS_KOLOM = {
    "panjang_atau_luas": "panjang_luas",
}

SCHEMA_RAW = StructType([StructField(col, StringType(), True) for col in KOLOM_STANDAR])

PLACEHOLDER_KOSONG = {"", "-", "?", "·", ".", "_", "`"}


def bersihkan_sel_html(teks: str) -> str:
    """Hapus tag HTML sisa, unescape entity, dan rapikan spasi."""
    tanpa_tag = TAG_PATTERN.sub("", teks)
    return " ".join(unescape(tanpa_tag).split())


def ekstrak_html_xls_ke_csv(sumber: Path, tujuan_csv: Path) -> dict[str, int]:
    """Konversi streaming file .xls (tabel HTML) ke CSV ber-header standar."""
    tujuan_csv.parent.mkdir(parents=True, exist_ok=True)
    header_kolom: list[str] | None = None
    jumlah_baris = 0
    baris_kolom_tidak_cocok = 0

    with (
        sumber.open("r", encoding="utf-8", errors="replace") as fin,
        tujuan_csv.open("w", encoding="utf-8", newline="") as fout,
    ):
        writer = csv.writer(fout)
        writer.writerow(KOLOM_STANDAR)

        for line in fin:
            if header_kolom is None and "<th" in line.lower():
                raw_headers = [bersihkan_sel_html(h).lower() for h in TH_PATTERN.findall(line)]
                mapped = [ALIAS_KOLOM.get(h, h) for h in raw_headers]
                if mapped == KOLOM_STANDAR:
                    header_kolom = mapped
                continue

            if "<td>" not in line and "<TD>" not in line:
                continue

            sel = [bersihkan_sel_html(v) for v in TD_PATTERN.findall(line)]
            if len(sel) != len(KOLOM_STANDAR):
                baris_kolom_tidak_cocok += 1
                continue

            writer.writerow(sel)
            jumlah_baris += 1

    return {
        "jumlah_baris_csv": jumlah_baris,
        "baris_kolom_tidak_cocok": baris_kolom_tidak_cocok,
    }


def proses_ingestion_tahun(
    spark: SparkSession,
    csv_path: Path,
    output_parquet: str,
    quarantine_parquet: str,
    tahun: int,
    source_file: str,
) -> dict:
    """Validasi tipe tanggal & volume di PySpark lalu tulis valid + quarantine."""
    df_str = spark.read.csv(
        csv_path.resolve().as_uri(),
        header=True,
        schema=SCHEMA_RAW,
    )

    # Trim spasi di seluruh kolom string.
    df_str = df_str.withColumns({c: F.trim(F.col(c)) for c in KOLOM_STANDAR})

    # Tanggal di dataset berupa hari (1..31) dan periode_data berupa YYYYMM.
    # Dengan ansi.enabled=false (default Spark 3.5), to_date mengembalikan NULL jika gagal.
    periode_str = F.col("periode_data")
    tgl_str = F.col("tanggal")
    tgl_gabung = F.concat_ws(
        "-",
        F.substring(periode_str, 1, 4),
        F.substring(periode_str, 5, 2),
        F.lpad(tgl_str, 2, "0"),
    )
    kandidat_tanggal = F.when(
        tgl_str.rlike(r"^\d{4}-\d{2}-\d{2}$"),
        F.to_date(tgl_str, "yyyy-MM-dd"),
    ).when(
        periode_str.rlike(r"^\d{6}$") & tgl_str.rlike(r"^\d{1,2}$"),
        F.to_date(tgl_gabung, "yyyy-MM-dd"),
    )

    vol_raw = F.col("volume_sampah_perhari")
    vol_norm = F.regexp_replace(vol_raw, ",", ".")
    is_vol_kosong = vol_raw.isNull() | vol_raw.isin(*sorted(PLACEHOLDER_KOSONG))
    vol_cast = F.when(is_vol_kosong, F.lit(None).cast("double")).otherwise(
        vol_norm.cast("double")
    )

    alasan_quarantine = (
        F.when(kandidat_tanggal.isNull(), F.lit("tanggal_tidak_valid"))
        .when((~is_vol_kosong) & vol_cast.isNull(), F.lit("volume_tidak_valid"))
        .otherwise(F.lit(None).cast("string"))
    )

    waktu_ingest = datetime.now(timezone.utc).isoformat(timespec="seconds")
    df_tagged = df_str.withColumns(
        {
            "tanggal_parsed": kandidat_tanggal,
            "volume_parsed": vol_cast,
            "alasan_quarantine": alasan_quarantine,
            "source_file": F.lit(source_file),
            "tahun_sumber": F.lit(int(tahun)),
            "ingested_at": F.lit(waktu_ingest),
        }
    ).cache()

    df_valid = (
        df_tagged.filter(F.col("alasan_quarantine").isNull())
        .drop("tanggal", "volume_sampah_perhari", "alasan_quarantine")
        .withColumnRenamed("tanggal_parsed", "tanggal")
        .withColumnRenamed("volume_parsed", "volume_sampah_perhari")
    )
    df_quarantine = df_tagged.filter(F.col("alasan_quarantine").isNotNull())

    n_masuk = df_tagged.count()
    n_valid = df_valid.count()
    n_quarantine = df_quarantine.count()
    assert n_masuk == n_valid + n_quarantine, (
        f"Ketidakcocokan baris {tahun}: masuk={n_masuk}, valid={n_valid}, quarantine={n_quarantine}"
    )

    # Idempotent per tahun_sumber dengan dynamic partition overwrite.
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    df_valid.write.mode("overwrite").partitionBy("tahun_sumber").parquet(output_parquet)
    if n_quarantine > 0:
        df_quarantine.write.mode("overwrite").partitionBy("tahun_sumber").parquet(
            quarantine_parquet
        )

    df_tagged.unpersist()

    return {
        "tahun": tahun,
        "source_file": source_file,
        "baris_masuk": n_masuk,
        "baris_valid": n_valid,
        "baris_quarantine": n_quarantine,
        "ingested_at": waktu_ingest,
    }


def deteksi_tahun_dari_nama(path: Path) -> int:
    cocok = re.search(r"(20\d{2})", path.name)
    if not cocok:
        raise ValueError(f"Tidak dapat mendeteksi tahun dari nama file: {path.name}")
    return int(cocok.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingestion batch dataset SiPeka")
    parser.add_argument("--hdfs", action="store_true", help="Tulis output Parquet juga ke HDFS")
    args = parser.parse_args()

    settings = get_settings()
    settings.pastikan_folder_lokal()

    files = sorted(settings.dataset_dir.glob("*.xls"))
    if not files:
        raise FileNotFoundError(f"Tidak ada file .xls di {settings.dataset_dir}")

    spark = get_spark("SiPeka-Step3-Ingestion")
    output_parquet = (
        f"{settings.hdfs_namenode}/sipeka/raw/parquet"
        if args.hdfs
        else (settings.data_raw_dir / "parquet").resolve().as_uri()
    )
    quarantine_parquet = (
        f"{settings.hdfs_namenode}/sipeka/quarantine"
        if args.hdfs
        else settings.data_quarantine_dir.resolve().as_uri()
    )

    laporan: list[dict] = []
    for xls_path in files:
        tahun = deteksi_tahun_dari_nama(xls_path)
        csv_staging = settings.data_raw_dir / f"sipeka_{tahun}.csv"
        print(f"[Step 3] Ekstrak HTML -> CSV: {xls_path.name} -> {csv_staging.name}")
        info_csv = ekstrak_html_xls_ke_csv(xls_path, csv_staging)
        print(f"         Baris CSV: {info_csv['jumlah_baris_csv']:,} (anomali kolom: {info_csv['baris_kolom_tidak_cocok']})")

        info_spark = proses_ingestion_tahun(
            spark=spark,
            csv_path=csv_staging,
            output_parquet=output_parquet,
            quarantine_parquet=quarantine_parquet,
            tahun=tahun,
            source_file=xls_path.name,
        )
        info_spark.update(info_csv)
        laporan.append(info_spark)
        print(
            f"         Valid: {info_spark['baris_valid']:,} | "
            f"Quarantine: {info_spark['baris_quarantine']:,}"
        )

    report_path = settings.data_raw_dir / "ingestion_report.json"
    report_path.write_text(json.dumps(laporan, indent=2), encoding="utf-8")
    print(f"\nLaporan ingestion disimpan ke {report_path}")
    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
