"""Auto-retraining pipeline SiPeka dengan Champion vs Challenger gate.

Melatih kandidat model pada data terbaru (split kronologis 70/15/15),
membandingkan RMSE validation Challenger terhadap model aktif (Champion) pada
data validasi yang sama, lalu mempromosikan model baru ke
`models/active_model.json` hanya jika performanya lebih baik (atau belum ada Champion).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd
from pyspark import StorageLevel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import (
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.regression import (
    GBTRegressor,
    LinearRegression,
    RandomForestRegressor,
)
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from config.settings import DATA_PROCESSED_DIR, MODELS_DIR, get_settings
from src.utils.spark_session import get_spark

TARGET = "volume_sampah_perhari"
KOLOM_TANGGAL = "tanggal"
KOLOM_KATEGORIK = ["titik_lokasi", "kecamatan", "wilayah"]
FITUR_KALENDER = [
    "tahun",
    "bulan",
    "hari_dalam_bulan",
    "hari_dalam_minggu",
    "is_weekend",
    "minggu_ke",
    "kuartal",
    "hari_ke_dalam_tahun",
    "bulan_sin",
    "bulan_cos",
    "hari_sin",
    "hari_cos",
    "is_musim_hujan",
    "is_hari_libur",
]
FITUR_HISTORIS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
    "rolling_std_7",
    "rolling_max_7",
    "titik_hist_mean",
    "titik_hist_zero_rate",
]

SEED = 42
RASIO_TRAIN = 0.70
RASIO_VALID = 0.15
MAX_BINS_MINIMUM = 32
LATIH_TANPA_OUTLIER = True

ACTIVE_MODEL_FILE = MODELS_DIR / "active_model.json"
RETRAIN_HISTORY_FILE = MODELS_DIR / "retrain_history.json"


def path_ke_uri(path: str | Path) -> str:
    teks = str(path)
    if "://" in teks:
        return teks
    return Path(teks).resolve().as_uri()


def muat_pipeline_model(spark: SparkSession, path_model: str | Path) -> PipelineModel:
    """Muat PipelineModel via JVM pada Windows agar bebas crash Python worker."""
    uri = path_ke_uri(path_model)
    if os.name == "nt":
        return PipelineModel._from_java(
            spark._jvm.org.apache.spark.ml.PipelineModel.load(uri)
        )
    try:
        return PipelineModel.load(uri)
    except Exception:
        return PipelineModel._from_java(
            spark._jvm.org.apache.spark.ml.PipelineModel.load(uri)
        )


def hitung_fingerprint_data(dataset_dir: Path, features_v2_dir: Path) -> str:
    """Sidik jari SHA-256 dari ukuran & mtime file sumber + Parquet features_v2."""
    h = hashlib.sha256()
    for folder in (dataset_dir, features_v2_dir):
        if not folder.exists():
            continue
        for p in sorted(folder.rglob("*")):
            if p.is_file() and not p.name.endswith(".crc"):
                st = p.stat()
                h.update(f"{p.relative_to(folder)}|{st.st_size}|{int(st.st_mtime)}".encode())
    return h.hexdigest()


def baca_active_model() -> dict | None:
    """Baca registry model aktif; jika belum ada, inisialisasi dari metadata terbaru."""
    if ACTIVE_MODEL_FILE.exists():
        return json.loads(ACTIVE_MODEL_FILE.read_text(encoding="utf-8"))

    kandidat_meta = sorted(
        MODELS_DIR.glob("*_metadata.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not kandidat_meta:
        return None

    meta_path = kandidat_meta[0]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    versi = meta["versi_model"]
    model_dir = MODELS_DIR / versi
    if not model_dir.exists():
        return None

    active = {
        "versi_model": versi,
        "model": meta["model"],
        "versi_fitur": meta["versi_fitur"],
        "model_path": str(model_dir),
        "metadata_path": str(meta_path),
        "metrik_path": str(MODELS_DIR / f"{versi}_metrik.csv"),
        "parameter": meta.get("parameter", {}),
        "skema_split": meta.get("skema_split", {}),
        "metrik_validation": meta.get("metrik_validation", {}),
        "metrik_test": meta.get("metrik_test", {}),
        "data_fingerprint": None,
        "promoted_at": meta.get("dilatih_pada", datetime.now().isoformat(timespec="seconds")),
    }
    ACTIVE_MODEL_FILE.write_text(json.dumps(active, indent=2), encoding="utf-8")
    return active


def simpan_riwayat(catatan: dict) -> None:
    riwayat: list[dict] = []
    if RETRAIN_HISTORY_FILE.exists():
        try:
            riwayat = json.loads(RETRAIN_HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            riwayat = []
    riwayat.append(catatan)
    RETRAIN_HISTORY_FILE.write_text(
        json.dumps(riwayat, indent=2, default=str), encoding="utf-8"
    )


def fitur_numerik_untuk_versi(versi: str) -> list[str]:
    return list(FITUR_KALENDER) + (list(FITUR_HISTORIS) if versi == "v2" else [])


def baca_parquet_fitur(spark: SparkSession, path: str | Path, versi: str) -> DataFrame:
    df = spark.read.parquet(path_ke_uri(path))
    if "tahun" not in df.columns:
        df = df.withColumn("tahun", F.year(KOLOM_TANGGAL))
    if "bulan" not in df.columns:
        df = df.withColumn("bulan", F.month(KOLOM_TANGGAL).cast("int"))
    else:
        df = df.withColumn("bulan", F.col("bulan").cast("int"))

    numerik = fitur_numerik_untuk_versi(versi)
    wajib = [TARGET, KOLOM_TANGGAL, "tahun", "bulan"] + KOLOM_KATEGORIK + numerik
    hilang = [c for c in wajib if c not in df.columns]
    if hilang:
        raise ValueError(f"Kolom wajib hilang di features_{versi}: {hilang}")

    cast_map = {c: F.col(c).cast("double") for c in numerik}
    cast_map[TARGET] = F.col(TARGET).cast("double")
    return df.withColumns(cast_map).filter(
        F.col("titik_lokasi").isNotNull() & F.col(TARGET).isNotNull()
    )


def tentukan_skema_split(df: DataFrame) -> dict:
    harian = (
        df.groupBy(KOLOM_TANGGAL)
        .agg(F.count("*").alias("n"))
        .orderBy(KOLOM_TANGGAL)
        .collect()
    )
    total = sum(int(r["n"]) for r in harian)
    target_train = total * RASIO_TRAIN
    target_valid = total * (RASIO_TRAIN + RASIO_VALID)

    kumulatif = 0
    tgl_train = None
    tgl_valid = None
    for row in harian:
        kumulatif += int(row["n"])
        if tgl_train is None and kumulatif >= target_train:
            tgl_train = row[KOLOM_TANGGAL]
        if tgl_valid is None and kumulatif >= target_valid:
            tgl_valid = row[KOLOM_TANGGAL]
            break

    return {
        "mode": "kronologis_70_15_15",
        "rasio_target": {"train": RASIO_TRAIN, "validation": RASIO_VALID, "test": 0.15},
        "tanggal_train_akhir": str(tgl_train),
        "tanggal_valid_akhir": str(tgl_valid),
        "total_baris": total,
        "tanggal_maks": str(harian[-1][KOLOM_TANGGAL]),
    }


def terapkan_split(df: DataFrame, skema: dict) -> DataFrame:
    tgl_train = F.lit(skema["tanggal_train_akhir"]).cast("date")
    tgl_valid = F.lit(skema["tanggal_valid_akhir"]).cast("date")
    return df.withColumn(
        "split",
        F.when(F.col(KOLOM_TANGGAL) <= tgl_train, F.lit("train"))
        .when(F.col(KOLOM_TANGGAL) <= tgl_valid, F.lit("validation"))
        .otherwise(F.lit("test")),
    ).persist(StorageLevel.MEMORY_AND_DISK)


def pecah_partisi_split(df_split: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    train_tuning = df_split.filter(F.col("split") == "train")
    valid = df_split.filter(F.col("split") == "validation")
    train_full = df_split.filter(F.col("split") != "test")
    test = df_split.filter(F.col("split") == "test")
    if LATIH_TANPA_OUTLIER and "is_outlier" in df_split.columns:
        train_tuning = train_tuning.filter(F.col("is_outlier") == 0)
        train_full = train_full.filter(F.col("is_outlier") == 0)
    return train_tuning, valid, train_full, test


def metrik_regresi(df: DataFrame, pred_col: str = "prediction") -> dict[str, float]:
    df_eval = df.withColumn("_pred_nonneg", F.greatest(F.lit(0.0), F.col(pred_col)))
    return {
        m: float(
            RegressionEvaluator(
                labelCol=TARGET, predictionCol="_pred_nonneg", metricName=m
            ).evaluate(df_eval)
        )
        for m in ("rmse", "mae", "r2")
    }


def buat_indexers() -> list[StringIndexer]:
    return [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in KOLOM_KATEGORIK
    ]


def kardinalitas_maks(df: DataFrame) -> int:
    nilai = [int(df.select(c).distinct().count()) for c in KOLOM_KATEGORIK]
    return max(MAX_BINS_MINIMUM, max(nilai) + 2)


def bangun_lr(versi: str, reg_param: float = 0.01, elastic_net: float = 0.1) -> Pipeline:
    numerik = fitur_numerik_untuk_versi(versi)
    ohe = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in KOLOM_KATEGORIK],
        outputCols=[f"{c}_ohe" for c in KOLOM_KATEGORIK],
        handleInvalid="keep",
    )
    assembler = VectorAssembler(
        inputCols=numerik + [f"{c}_ohe" for c in KOLOM_KATEGORIK],
        outputCol="features",
        handleInvalid="skip",
    )
    scaler = StandardScaler(
        inputCol="features", outputCol="features_scaled", withMean=False, withStd=True
    )
    lr = LinearRegression(
        featuresCol="features_scaled",
        labelCol=TARGET,
        predictionCol="prediction",
        maxIter=100,
        regParam=reg_param,
        elasticNetParam=elastic_net,
    )
    return Pipeline(stages=buat_indexers() + [ohe, assembler, scaler, lr])


def bangun_rf(
    versi: str, num_trees: int, max_depth: int, min_instances: int, max_bins: int
) -> Pipeline:
    numerik = fitur_numerik_untuk_versi(versi)
    assembler = VectorAssembler(
        inputCols=numerik + [f"{c}_idx" for c in KOLOM_KATEGORIK],
        outputCol="features",
        handleInvalid="skip",
    )
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol=TARGET,
        predictionCol="prediction",
        numTrees=num_trees,
        maxDepth=max_depth,
        minInstancesPerNode=min_instances,
        maxBins=max_bins,
        subsamplingRate=0.8,
        featureSubsetStrategy="onethird",
        maxMemoryInMB=64,
        seed=SEED,
    )
    return Pipeline(stages=buat_indexers() + [assembler, rf])


def bangun_gbt(
    versi: str, max_iter: int, max_depth: int, step_size: float, max_bins: int
) -> Pipeline:
    numerik = fitur_numerik_untuk_versi(versi)
    assembler = VectorAssembler(
        inputCols=numerik + [f"{c}_idx" for c in KOLOM_KATEGORIK],
        outputCol="features",
        handleInvalid="skip",
    )
    gbt = GBTRegressor(
        featuresCol="features",
        labelCol=TARGET,
        predictionCol="prediction",
        maxIter=max_iter,
        maxDepth=max_depth,
        stepSize=step_size,
        minInstancesPerNode=5,
        subsamplingRate=0.8,
        maxBins=max_bins,
        maxMemoryInMB=64,
        seed=SEED,
    )
    return Pipeline(stages=buat_indexers() + [assembler, gbt])


def evaluasi_kandidat(
    nama_model: str,
    versi: str,
    params: dict,
    pipeline_valid: Pipeline,
    pipeline_final_fn,
    train_tuning: DataFrame,
    valid: DataFrame,
    train_full: DataFrame,
    test: DataFrame,
    skema: dict,
    max_bins_final: int | None = None,
) -> dict:
    t0 = time.perf_counter()
    m_valid = pipeline_valid.fit(train_tuning)
    d_valid = time.perf_counter() - t0
    pred_valid = m_valid.transform(valid)
    met_valid = metrik_regresi(pred_valid)

    t1 = time.perf_counter()
    m_final = pipeline_final_fn().fit(train_full)
    d_final = time.perf_counter() - t1
    pred_test = m_final.transform(test)
    met_test = metrik_regresi(pred_test)

    numerik = fitur_numerik_untuk_versi(versi)
    suffix = "_ohe" if nama_model == "LinearRegression" else "_idx"
    print(
        f"  [{nama_model} {versi}] params={params} -> "
        f"RMSE valid={met_valid['rmse']:.4f}, MAE valid={met_valid['mae']:.4f} | "
        f"RMSE test={met_test['rmse']:.4f}, MAE test={met_test['mae']:.4f}, "
        f"R2 test={met_test['r2']:.4f} ({d_valid + d_final:.1f} dtk)"
    )
    return {
        "model_name": nama_model,
        "versi_fitur": versi,
        "model": m_final,
        "params": params,
        "max_bins": max_bins_final,
        "metrik_valid": met_valid,
        "metrik_test": met_test,
        "durasi_latih_detik": round(d_valid + d_final, 3),
        "n_valid_dinilai": pred_valid.count(),
        "n_valid_total": valid.count(),
        "n_test_dinilai": pred_test.count(),
        "n_test_total": test.count(),
        "fitur_numerik": numerik,
        "fitur_kategorik": list(KOLOM_KATEGORIK),
        "fitur_model": numerik + [f"{c}{suffix}" for c in KOLOM_KATEGORIK],
        "skema_split": skema,
    }


def jalankan_retrain(
    mode: str = "fast",
    force: bool = False,
    min_improvement: float = 0.0,
) -> dict:
    """Jalankan 1 siklus auto-retrain Champion vs Challenger."""
    settings = get_settings()
    settings.pastikan_folder_lokal()

    v2_path = Path(
        os.environ.get("SIPEKA_FEATURES_V2", str(DATA_PROCESSED_DIR / "features_v2"))
    )
    if not v2_path.exists():
        raise FileNotFoundError(
            f"Parquet features_v2 tidak ditemukan di {v2_path}. Jalankan Step 3-4 dulu."
        )

    fingerprint = hitung_fingerprint_data(settings.dataset_dir, v2_path)
    champion = baca_active_model()

    if (
        not force
        and champion is not None
        and champion.get("data_fingerprint") == fingerprint
    ):
        keputusan = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "status": "SKIPPED_NO_NEW_DATA",
            "mode": mode,
            "data_fingerprint": fingerprint,
            "champion_aktif": champion["versi_model"],
            "alasan": "Tidak ada perubahan pada Dataset/ maupun features_v2.",
        }
        simpan_riwayat(keputusan)
        print(f"[Auto-Retrain] SKIPPED: data belum berubah (Champion: {champion['versi_model']}).")
        return keputusan

    spark = get_spark("SiPeka-AutoRetrain")
    try:
        df_v2 = baca_parquet_fitur(spark, v2_path, "v2")
        skema = tentukan_skema_split(df_v2)
        split = terapkan_split(df_v2, skema)
        train_tuning, valid, train_full, test = pecah_partisi_split(split)

        print(f"[Auto-Retrain] Mode={mode} | Skema={json.dumps(skema)}")
        max_bins_train = kardinalitas_maks(train_tuning)
        max_bins_full = kardinalitas_maks(train_full)

        # Evaluasi Champion lama secara adil (bebas leakage):
        # - Jika batas tanggal validation tidak berubah, gunakan metrik validation out-of-sample
        #   yang tercatat saat Champion dilatih (karena model tersimpan sudah di-refit pada train+valid).
        # - Jika ada data baru yang menggeser batas validation ke masa depan, evaluasi langsung pada valid baru.
        champion_rmse_valid: float | None = None
        champion_metrik_baru: dict | None = None
        if champion is not None and Path(champion["model_path"]).exists():
            try:
                m_champ = muat_pipeline_model(spark, champion["model_path"])
                champ_test_met = metrik_regresi(m_champ.transform(test))
                champ_valid_akhir = (champion.get("skema_split") or {}).get("tanggal_valid_akhir")
                champ_valid_tercatat = champion.get("metrik_validation") or {}

                if (
                    champ_valid_akhir == skema.get("tanggal_valid_akhir")
                    and "rmse" in champ_valid_tercatat
                ):
                    champ_valid_met = champ_valid_tercatat
                    sumber_valid = "tercatat (out-of-sample)"
                else:
                    champ_valid_met = metrik_regresi(m_champ.transform(valid))
                    sumber_valid = "evaluasi live pada valid baru"

                champion_rmse_valid = float(champ_valid_met["rmse"])
                champion_metrik_baru = {
                    "validation": champ_valid_met,
                    "validation_source": sumber_valid,
                    "test": champ_test_met,
                }
                print(
                    f"[Champion] {champion['versi_model']} ({sumber_valid}) -> "
                    f"RMSE valid={champion_rmse_valid:.4f}, RMSE test={champ_test_met['rmse']:.4f}"
                )
            except Exception as exc:
                print(f"[Champion] Gagal mengevaluasi model lama ({exc}); akan melatih ulang.")

        # Kandidat Challenger (mode 'fast' fokus pada arsitektur v2 terbaik agar cepat di cron).
        kandidat_hasil: list[dict] = []
        lr_grid = (
            [{"regParam": 0.01, "elasticNetParam": 0.1}, {"regParam": 0.05, "elasticNetParam": 0.1}]
            if mode == "fast"
            else [
                {"regParam": 0.005, "elasticNetParam": 0.1},
                {"regParam": 0.01, "elasticNetParam": 0.1},
                {"regParam": 0.05, "elasticNetParam": 0.2},
            ]
        )
        for p in lr_grid:
            kandidat_hasil.append(
                evaluasi_kandidat(
                    "LinearRegression",
                    "v2",
                    {**p, "maxIter": 100},
                    bangun_lr("v2", p["regParam"], p["elasticNetParam"]),
                    lambda p=p: bangun_lr("v2", p["regParam"], p["elasticNetParam"]),
                    train_tuning,
                    valid,
                    train_full,
                    test,
                    skema,
                )
            )

        rf_grid = (
            [{"num_trees": 50, "max_depth": 6, "min_instances": 10}]
            if mode == "fast"
            else [
                {"num_trees": 30, "max_depth": 6, "min_instances": 10},
                {"num_trees": 50, "max_depth": 6, "min_instances": 10},
                {"num_trees": 50, "max_depth": 8, "min_instances": 5},
            ]
        )
        for p in rf_grid:
            kandidat_hasil.append(
                evaluasi_kandidat(
                    "RandomForest",
                    "v2",
                    p,
                    bangun_rf("v2", p["num_trees"], p["max_depth"], p["min_instances"], max_bins_train),
                    lambda p=p: bangun_rf("v2", p["num_trees"], p["max_depth"], p["min_instances"], max_bins_full),
                    train_tuning,
                    valid,
                    train_full,
                    test,
                    skema,
                    max_bins_final=max_bins_full,
                )
            )

        gbt_grid = (
            [{"max_iter": 30, "max_depth": 4, "step_size": 0.1}]
            if mode == "fast"
            else [
                {"max_iter": 30, "max_depth": 4, "step_size": 0.1},
                {"max_iter": 50, "max_depth": 4, "step_size": 0.1},
                {"max_iter": 30, "max_depth": 6, "step_size": 0.1},
            ]
        )
        for p in gbt_grid:
            kandidat_hasil.append(
                evaluasi_kandidat(
                    "GBT",
                    "v2",
                    p,
                    bangun_gbt("v2", p["max_iter"], p["max_depth"], p["step_size"], max_bins_train),
                    lambda p=p: bangun_gbt("v2", p["max_iter"], p["max_depth"], p["step_size"], max_bins_full),
                    train_tuning,
                    valid,
                    train_full,
                    test,
                    skema,
                    max_bins_final=max_bins_full,
                )
            )

        challenger = min(kandidat_hasil, key=lambda r: r["metrik_valid"]["rmse"])
        challenger_rmse = challenger["metrik_valid"]["rmse"]

        harus_promosi = (
            champion_rmse_valid is None
            or challenger_rmse < (champion_rmse_valid - min_improvement)
        )

        tabel_kandidat = pd.DataFrame(
            [
                {
                    "model": r["model_name"],
                    "versi_fitur": r["versi_fitur"],
                    "params": json.dumps(r["params"]),
                    "rmse_valid": r["metrik_valid"]["rmse"],
                    "mae_valid": r["metrik_valid"]["mae"],
                    "r2_valid": r["metrik_valid"]["r2"],
                    "rmse_test": r["metrik_test"]["rmse"],
                    "mae_test": r["metrik_test"]["mae"],
                    "r2_test": r["metrik_test"]["r2"],
                    "durasi_latih_detik": r["durasi_latih_detik"],
                }
                for r in kandidat_hasil
            ]
        ).sort_values("rmse_valid")

        if harus_promosi:
            versi_baru = (
                f"{challenger['model_name'].lower()}_{challenger['versi_fitur']}_"
                f"{datetime.now():%Y%m%d_%H%M%S}"
            )
            path_model = MODELS_DIR / versi_baru
            challenger["model"].write().overwrite().save(path_ke_uri(path_model))

            metadata = {
                "versi_model": versi_baru,
                "model": challenger["model_name"],
                "versi_fitur": challenger["versi_fitur"],
                "fitur_numerik": challenger["fitur_numerik"],
                "fitur_kategorik": challenger["fitur_kategorik"],
                "fitur_model": challenger["fitur_model"],
                "parameter": challenger["params"],
                "max_bins": challenger["max_bins"],
                "skema_split": skema,
                "latih_tanpa_outlier": LATIH_TANPA_OUTLIER,
                "metrik_validation": challenger["metrik_valid"],
                "metrik_test": challenger["metrik_test"],
                "dilatih_pada": datetime.now().isoformat(timespec="seconds"),
                "catatan": f"Dipromosikan oleh auto-retrain ({mode}).",
            }
            path_meta = MODELS_DIR / f"{versi_baru}_metadata.json"
            path_meta.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

            path_metrik = MODELS_DIR / f"{versi_baru}_metrik.csv"
            tabel_kandidat.to_csv(path_metrik, index=False)

            active_payload = {
                "versi_model": versi_baru,
                "model": challenger["model_name"],
                "versi_fitur": challenger["versi_fitur"],
                "model_path": str(path_model),
                "metadata_path": str(path_meta),
                "metrik_path": str(path_metrik),
                "parameter": challenger["params"],
                "skema_split": skema,
                "metrik_validation": challenger["metrik_valid"],
                "metrik_test": challenger["metrik_test"],
                "data_fingerprint": fingerprint,
                "promoted_at": datetime.now().isoformat(timespec="seconds"),
            }
            ACTIVE_MODEL_FILE.write_text(
                json.dumps(active_payload, indent=2, default=str), encoding="utf-8"
            )
            try:
                from src.modeling.export_model import ekspor_model_aktif

                ekspor_model_aktif(spark_existing=spark)
            except Exception as exc:
                print(f"[Auto-Retrain] Peringatan ekspor .pkl/.h5: {exc}")
            status = "PROMOTED"
            versi_aktif = versi_baru
            print(
                f"[Auto-Retrain] PROMOTED: {versi_baru} "
                f"(RMSE valid {challenger_rmse:.4f} vs Champion {champion_rmse_valid})"
            )
        else:
            # Perbarui fingerprint pada Champion agar tidak terus retrain pada data yang sama.
            assert champion is not None
            champion["data_fingerprint"] = fingerprint
            champion["last_checked_at"] = datetime.now().isoformat(timespec="seconds")
            ACTIVE_MODEL_FILE.write_text(
                json.dumps(champion, indent=2, default=str), encoding="utf-8"
            )
            status = "KEPT_CHAMPION"
            versi_aktif = champion["versi_model"]
            print(
                f"[Auto-Retrain] KEPT CHAMPION: {versi_aktif} "
                f"(Champion RMSE valid {champion_rmse_valid:.4f} <= Challenger {challenger_rmse:.4f})"
            )

        split.unpersist()

        keputusan = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "status": status,
            "mode": mode,
            "data_fingerprint": fingerprint,
            "skema_split": skema,
            "champion_sebelum": champion["versi_model"] if champion else None,
            "champion_rmse_valid": champion_rmse_valid,
            "champion_metrik_evaluasi": champion_metrik_baru,
            "challenger_terbaik": {
                "model": challenger["model_name"],
                "versi_fitur": challenger["versi_fitur"],
                "params": challenger["params"],
                "metrik_validation": challenger["metrik_valid"],
                "metrik_test": challenger["metrik_test"],
            },
            "model_aktif_akhir": versi_aktif,
        }
        simpan_riwayat(keputusan)
        return keputusan
    finally:
        spark.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-retrain Champion vs Challenger SiPeka")
    parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--force", action="store_true", help="Latih ulang walau data belum berubah")
    parser.add_argument("--min-improvement", type=float, default=0.0)
    args = parser.parse_args()
    jalankan_retrain(mode=args.mode, force=args.force, min_improvement=args.min_improvement)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
