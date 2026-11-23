"""Ekspor model aktif PySpark ke format `.pkl` (Pickle/Joblib) dan `.h5` (HDF5).

Memungkinkan integrasi langsung ke dashboard Laravel / FastAPI / script Python
tanpa perlu menyalakan JVM / PySpark saat inferensi real-time (~1 ms per prediksi).
Juga otomatis dieksekusi setiap kali `train.py` atau `score.py` selesai berjalan.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import h5py
import joblib
import numpy as np
import pandas as pd

from config.settings import DATA_DIR, DATA_PROCESSED_DIR, DATASET_DIR, MODELS_DIR
from src.modeling.train import (
    ACTIVE_MODEL_FILE,
    FITUR_HISTORIS,
    FITUR_KALENDER,
    KOLOM_KATEGORIK,
    baca_active_model,
    baca_parquet_fitur,
    fitur_numerik_untuk_versi,
    muat_pipeline_model,
)
from src.processing.clean_features import HARI_LIBUR_NASIONAL
from src.utils.spark_session import get_spark

SERVING_DIR = DATA_DIR / "serving"
BULAN_MUSIM_HUJAN = {10, 11, 12, 1, 2, 3, 4}
HARI_LIBUR_SET = set(HARI_LIBUR_NASIONAL)

PKL_ACTIVE_PATH = MODELS_DIR / "active_model.pkl"
H5_ACTIVE_PATH = MODELS_DIR / "active_model.h5"


def normalisasi_nama_titik(teks: Any) -> str:
    """Normalisasi nama titik/LPS identik dengan Step 4 (`clean_features.py`)."""
    s = str(teks or "").strip().upper().replace(",", ".")
    s = re.sub(r"^[\s.,;:-]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def hitung_fitur_kalender_dict(tgl: date) -> dict[str, float]:
    """Hitung 14 fitur kalender identik 100% dengan PySpark Step 4 (`clean_features.py`)."""
    # Python isoweekday(): Senin=1..Minggu=7 -> konversi ke Spark dayofweek: Minggu=1, Senin=2..Sabtu=7
    iso = tgl.isoweekday()
    dow_spark = 1.0 if iso == 7 else float(iso + 1)
    bulan = float(tgl.month)
    hari_bulan = float(tgl.day)
    is_weekend = 1.0 if dow_spark in (1.0, 7.0) else 0.0
    minggu_ke = float(math.ceil(tgl.day / 7.0))
    kuartal = float((tgl.month - 1) // 3 + 1)
    hari_tahun = float(tgl.timetuple().tm_yday)
    is_musim_hujan = 1.0 if tgl.month in BULAN_MUSIM_HUJAN else 0.0
    is_libur = 1.0 if tgl.isoformat() in HARI_LIBUR_SET else 0.0

    sudut_bulan = 2.0 * math.pi * (bulan - 1.0) / 12.0
    sudut_hari = 2.0 * math.pi * (dow_spark - 1.0) / 7.0

    return {
        "tahun": float(tgl.year),
        "bulan": bulan,
        "hari_dalam_bulan": hari_bulan,
        "hari_dalam_minggu": dow_spark,
        "is_weekend": is_weekend,
        "minggu_ke": minggu_ke,
        "kuartal": kuartal,
        "hari_ke_dalam_tahun": hari_tahun,
        "bulan_sin": math.sin(sudut_bulan),
        "bulan_cos": math.cos(sudut_bulan),
        "hari_sin": math.sin(sudut_hari),
        "hari_cos": math.cos(sudut_hari),
        "is_musim_hujan": is_musim_hujan,
        "is_hari_libur": is_libur,
    }


class SiPekaPortableModel:
    """Model portabel NumPy/scikit-learn hasil ekstraksi bobot PySpark PipelineModel.

    Dapat disimpan ke `.pkl` dan dijalankan langsung dari Python/FastAPI/Laravel
    tanpa ketergantungan pada Spark atau Java runtime.
    """

    def __init__(self, bundle: dict[str, Any]) -> None:
        self.bundle = bundle
        self.versi_model: str = bundle["versi_model"]
        self.nama_model: str = bundle["model"]
        self.versi_fitur: str = bundle["versi_fitur"]
        self.fitur_numerik: list[str] = bundle["fitur_numerik"]
        self.kolom_kategorik: list[str] = bundle["kolom_kategorik"]
        self.indexer_labels: dict[str, list[str]] = bundle["indexer_labels"]
        self.indexer_maps: dict[str, dict[str, int]] = {
            kol: {lbl: i for i, lbl in enumerate(labels)}
            for kol, labels in self.indexer_labels.items()
        }
        self.ohe_sizes: list[int] = bundle["ohe_sizes"]
        self.scaler_std = np.asarray(bundle["scaler_std"], dtype=np.float64)
        self.coefficients = np.asarray(bundle["coefficients"], dtype=np.float64)
        self.intercept = float(bundle["intercept"])
        self.default_historis_global: dict[str, float] = bundle["default_historis_global"]
        self.profil_titik: dict[str, dict[str, Any]] = bundle["profil_titik"]
        self.metrik_validation: dict[str, float] = bundle.get("metrik_validation", {})
        self.metrik_test: dict[str, float] = bundle.get("metrik_test", {})

        # Pre-split bobot linear: bagian numerik + bagian OHE tiap kolom kategorik
        n_num = len(self.fitur_numerik)
        std_safe = np.where(self.scaler_std > 0, self.scaler_std, 1.0)
        std_mask = np.where(self.scaler_std > 0, 1.0, 0.0)
        self.effective_weights = (self.coefficients / std_safe) * std_mask
        self.num_weights = self.effective_weights[:n_num]

        self.cat_weight_slices: dict[str, np.ndarray] = {}
        offset = n_num
        for kol, size in zip(self.kolom_kategorik, self.ohe_sizes):
            self.cat_weight_slices[kol] = self.effective_weights[offset : offset + size]
            offset += size

    def _cari_profil(self, titik_lokasi: str) -> dict[str, Any]:
        kunci = normalisasi_nama_titik(titik_lokasi)
        if kunci in self.profil_titik:
            return self.profil_titik[kunci]
        return {}

    def predict_one(
        self,
        titik_lokasi: str,
        tanggal: str | date,
        kecamatan: str | None = None,
        wilayah: str | None = None,
        fitur_historis_override: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Prediksi volume sampah (m³) untuk 1 titik dan 1 tanggal (< 1 ms)."""
        if isinstance(tanggal, str):
            tgl = date.fromisoformat(tanggal[:10])
        else:
            tgl = tanggal

        kunci_titik = normalisasi_nama_titik(titik_lokasi)
        profil = self.profil_titik.get(kunci_titik, {})

        kec = normalisasi_nama_titik(kecamatan or profil.get("kecamatan") or "TIDAK DIKETAHUI")
        wil = normalisasi_nama_titik(wilayah or profil.get("wilayah") or "TIDAK DIKETAHUI")

        kal = hitung_fitur_kalender_dict(tgl)
        hist = dict(self.default_historis_global)
        if "fitur_historis" in profil:
            hist.update(profil["fitur_historis"])
        if fitur_historis_override:
            hist.update({k: float(v) for k, v in fitur_historis_override.items() if v is not None})

        semua_num = {**kal, **hist}
        x_num = np.array([float(semua_num.get(c, 0.0)) for c in self.fitur_numerik], dtype=np.float64)

        skor = float(self.intercept + np.dot(self.num_weights, x_num))

        nilai_kat = {
            "titik_lokasi": kunci_titik,
            "kecamatan": kec,
            "wilayah": wil,
        }
        for kol in self.kolom_kategorik:
            peta = self.indexer_maps[kol]
            idx = peta.get(nilai_kat[kol], len(peta))  # handleInvalid="keep" -> index terakhir
            w_slice = self.cat_weight_slices[kol]
            if 0 <= idx < len(w_slice):
                skor += float(w_slice[idx])

        y_pred = round(max(0.0, skor), 4)
        return {
            "tanggal": tgl.isoformat(),
            "titik_lokasi": kunci_titik,
            "lps_asli": profil.get("lps_asli", titik_lokasi),
            "kecamatan": kec,
            "wilayah": wil,
            "prediksi_m3": y_pred,
            "model_version": self.versi_model,
            "koordinat": {
                "lat": profil.get("lat"),
                "lng": profil.get("lng"),
            },
            "info_phb_v3": profil.get("info_phb_v3"),
        }

    def forecast_titik(
        self,
        titik_lokasi: str,
        tanggal_mulai: str | date,
        horizon_hari: int = 30,
    ) -> list[dict[str, Any]]:
        """Prediksi rentang `horizon_hari` ke depan untuk satu titik lokasi."""
        tgl0 = (
            date.fromisoformat(tanggal_mulai[:10])
            if isinstance(tanggal_mulai, str)
            else tanggal_mulai
        )
        return [
            self.predict_one(titik_lokasi, tgl0 + timedelta(days=i))
            for i in range(horizon_hari)
        ]


def muat_dataset_titik_v3() -> tuple[Path, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Muat dataset_titik_phb-v3.json terbaru (tanpa memakai titik_phb_silika.json lama)."""
    kandidat = [
        DATASET_DIR / "dataset_titik_phb-v3.json",
        DATA_PROCESSED_DIR / "dataset_titik_phb-v3.json",
        Path(r"c:\Users\anand\Downloads\dataset_titik_phb-v3.json"),
    ]
    sumber = next((p for p in kandidat if p.exists()), None)
    if sumber is None:
        raise FileNotFoundError("File dataset_titik_phb-v3.json tidak ditemukan.")

    raw_list: list[dict[str, Any]] = json.loads(sumber.read_text(encoding="utf-8"))

    # Pastikan salinan di data/processed/dataset_titik_phb-v3.json selalu sinkron,
    # dan hapus file titik_phb_silika.json lama jika masih ada.
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    target_processed = DATA_PROCESSED_DIR / "dataset_titik_phb-v3.json"
    if sumber.resolve() != target_processed.resolve():
        target_processed.write_text(
            json.dumps(raw_list, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    file_lama = DATA_PROCESSED_DIR / "titik_phb_silika.json"
    if file_lama.exists():
        try:
            file_lama.unlink()
        except Exception:
            pass

    peta_v3: dict[str, dict[str, Any]] = {}
    daftar_bersih: list[dict[str, Any]] = []
    for item in raw_list:
        kunci = normalisasi_nama_titik(item.get("lps"))
        wil_raw = normalisasi_nama_titik(item.get("kab") or item.get("wilayah_grup"))
        wil_sipeka = wil_raw.replace("KOTA ADM. ", "").strip()
        entri = {
            "id": item.get("id"),
            "titik_lokasi": kunci,
            "lps": item.get("lps"),
            "kode_kab": item.get("kode_kab"),
            "wilayah_grup": item.get("wilayah_grup"),
            "kab": item.get("kab"),
            "wilayah": wil_sipeka,
            "kec": normalisasi_nama_titik(item.get("kec")),
            "kel": normalisasi_nama_titik(item.get("kel")),
            "status_lahan": item.get("status_lahan"),
            "luas_lahan": item.get("luas_lahan"),
            "alat_berat": item.get("alat_berat"),
            "truk": item.get("truk"),
            "nama_koordinator": item.get("nama_koordinator"),
            "lat": float(item["lat"]) if item.get("lat") is not None else None,
            "lng": float(item["lng"]) if item.get("lng") is not None else None,
            "kat": item.get("kat"),
            "foto_url": item.get("foto_url"),
            "marker_icon": item.get("marker_icon"),
        }
        daftar_bersih.append(entri)
        if kunci and kunci not in peta_v3:
            peta_v3[kunci] = entri

    return target_processed, daftar_bersih, peta_v3


def simpan_ke_h5(path_h5: Path, bundle: dict[str, Any], v3_forecast: list[dict[str, Any]]) -> None:
    """Simpan seluruh bobot model, encoder, fitur, dan tabel profil ke format HDF5 (.h5)."""
    path_h5.parent.mkdir(parents=True, exist_ok=True)
    dt_str = h5py.string_dtype(encoding="utf-8")

    with h5py.File(path_h5, "w") as f:
        # Atribut utama pada root
        f.attrs["versi_model"] = bundle["versi_model"]
        f.attrs["model"] = bundle["model"]
        f.attrs["versi_fitur"] = bundle["versi_fitur"]
        f.attrs["exported_at"] = bundle["exported_at"]
        f.attrs["metrik_validation_json"] = json.dumps(bundle.get("metrik_validation", {}))
        f.attrs["metrik_test_json"] = json.dumps(bundle.get("metrik_test", {}))

        # Grup bobot numerik
        g_weights = f.create_group("weights")
        g_weights.create_dataset(
            "coefficients",
            data=np.asarray(bundle["coefficients"], dtype=np.float64),
            compression="gzip",
        )
        g_weights.create_dataset(
            "intercept", data=np.array([bundle["intercept"]], dtype=np.float64)
        )
        g_weights.create_dataset(
            "scaler_std",
            data=np.asarray(bundle["scaler_std"], dtype=np.float64),
            compression="gzip",
        )
        g_weights.create_dataset(
            "ohe_sizes", data=np.asarray(bundle["ohe_sizes"], dtype=np.int64)
        )

        # Grup fitur & encoder
        g_feat = f.create_group("features")
        g_feat.create_dataset(
            "fitur_numerik",
            data=np.asarray(bundle["fitur_numerik"], dtype=object),
            dtype=dt_str,
        )
        g_feat.create_dataset(
            "kolom_kategorik",
            data=np.asarray(bundle["kolom_kategorik"], dtype=object),
            dtype=dt_str,
        )
        g_feat.attrs["default_historis_global_json"] = json.dumps(
            bundle["default_historis_global"]
        )

        g_enc = f.create_group("encoders")
        for kol, labels in bundle["indexer_labels"].items():
            g_enc.create_dataset(
                kol,
                data=np.asarray(labels, dtype=object),
                dtype=dt_str,
                compression="gzip",
            )

        # Grup profil lokasi & forecast v3
        g_serv = f.create_group("serving")
        g_serv.create_dataset(
            "profil_titik_json",
            data=np.array([json.dumps(bundle["profil_titik"], ensure_ascii=False)], dtype=object),
            dtype=dt_str,
        )
        g_serv.create_dataset(
            "titik_phb_v3_forecast_json",
            data=np.array([json.dumps(v3_forecast, ensure_ascii=False)], dtype=object),
            dtype=dt_str,
        )


def ekspor_model_aktif(spark_existing=None) -> dict[str, Any]:
    """Ekstrak model PySpark aktif ke `.pkl` dan `.h5` + sinkronkan `dataset_titik_phb-v3.json`."""
    active = baca_active_model()
    if active is None:
        raise FileNotFoundError("models/active_model.json belum tersedia.")

    versi_model = active["versi_model"]
    versi_fitur = active.get("versi_fitur", "v2")
    fitur_numerik = fitur_numerik_untuk_versi(versi_fitur)

    path_v3, daftar_v3, peta_v3 = muat_dataset_titik_v3()

    milik_spark_sendiri = spark_existing is None
    spark = spark_existing or get_spark("SiPeka-ExportModel")
    try:
        pmodel = muat_pipeline_model(spark, active["model_path"])
        stages = pmodel.stages

        # Stage 0..2: StringIndexerModel
        indexer_labels: dict[str, list[str]] = {}
        for i, kol in enumerate(KOLOM_KATEGORIK):
            indexer_labels[kol] = list(stages[i].labels)

        # Stage 3: OneHotEncoderModel, Stage 5: StandardScalerModel, Stage 6: Regressor
        if active["model"] == "LinearRegression":
            ohe_model = stages[3]
            scaler_model = stages[5]
            lr_model = stages[6]
            ohe_sizes = list(ohe_model.categorySizes)
            scaler_std = list(scaler_model.std.toArray())
            coefficients = list(lr_model.coefficients.toArray())
            intercept = float(lr_model.intercept)
        else:
            # Fallback jika model pohon terpilih di masa depan
            ohe_sizes = [len(indexer_labels[k]) + 1 for k in KOLOM_KATEGORIK]
            n_total = len(fitur_numerik) + sum(ohe_sizes)
            scaler_std = [1.0] * n_total
            coefficients = [0.0] * n_total
            intercept = 0.0

        # Hitung profil historis per titik dari 30 hari terakhir features_v2
        v2_path = Path(
            os.environ.get("SIPEKA_FEATURES_V2", str(DATA_PROCESSED_DIR / "features_v2"))
        )
        from pyspark.sql import functions as F

        df_v2 = baca_parquet_fitur(spark, v2_path, "v2")
        max_tgl = df_v2.agg(F.max("tanggal")).first()[0]
        batas_30 = max_tgl - timedelta(days=30)
        df_ref = df_v2.filter(F.col("tanggal") > F.lit(batas_30))

        rata_global_row = (
            df_ref.agg(*[F.mean(c).alias(c) for c in FITUR_HISTORIS]).first().asDict()
        )
        default_historis_global = {
            k: float(v if v is not None else 0.0) for k, v in rata_global_row.items()
        }

        hist_pd = (
            df_ref.groupBy("titik_lokasi")
            .agg(
                F.first("kecamatan", ignorenulls=True).alias("kecamatan"),
                F.first("wilayah", ignorenulls=True).alias("wilayah"),
                *[F.mean(c).alias(c) for c in FITUR_HISTORIS],
            )
            .toPandas()
        )

        # Lengkapi titik yang tidak aktif di 30 hari terakhir dari seluruh histori
        semua_lok_pd = (
            df_v2.groupBy("titik_lokasi")
            .agg(
                F.last("kecamatan", ignorenulls=True).alias("kecamatan"),
                F.last("wilayah", ignorenulls=True).alias("wilayah"),
            )
            .toPandas()
        )
        gabung_pd = semua_lok_pd.merge(
            hist_pd[["titik_lokasi"] + FITUR_HISTORIS], on="titik_lokasi", how="left"
        )
        for c in FITUR_HISTORIS:
            gabung_pd[c] = gabung_pd[c].fillna(default_historis_global[c])

        profil_titik: dict[str, dict[str, Any]] = {}
        for row in gabung_pd.to_dict(orient="records"):
            kunci = row["titik_lokasi"]
            info_v3 = peta_v3.get(kunci)
            profil_titik[kunci] = {
                "titik_lokasi": kunci,
                "lps_asli": info_v3["lps"] if info_v3 else kunci,
                "kecamatan": row["kecamatan"],
                "wilayah": row["wilayah"],
                "lat": info_v3["lat"] if info_v3 else None,
                "lng": info_v3["lng"] if info_v3 else None,
                "fitur_historis": {c: float(row[c]) for c in FITUR_HISTORIS},
                "info_phb_v3": info_v3,
            }

        # Tambahkan juga titik dari dataset_titik_phb-v3.json yang belum pernah muncul di histori
        for kunci, info_v3 in peta_v3.items():
            if kunci not in profil_titik:
                profil_titik[kunci] = {
                    "titik_lokasi": kunci,
                    "lps_asli": info_v3["lps"],
                    "kecamatan": info_v3["kec"],
                    "wilayah": info_v3["wilayah"],
                    "lat": info_v3["lat"],
                    "lng": info_v3["lng"],
                    "fitur_historis": dict(default_historis_global),
                    "info_phb_v3": info_v3,
                }

        bundle = {
            "versi_model": versi_model,
            "model": active["model"],
            "versi_fitur": versi_fitur,
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "fitur_numerik": fitur_numerik,
            "kolom_kategorik": list(KOLOM_KATEGORIK),
            "indexer_labels": indexer_labels,
            "ohe_sizes": ohe_sizes,
            "scaler_std": scaler_std,
            "coefficients": coefficients,
            "intercept": intercept,
            "default_historis_global": default_historis_global,
            "profil_titik": profil_titik,
            "metrik_validation": active.get("metrik_validation", {}),
            "metrik_test": active.get("metrik_test", {}),
            "skema_split": active.get("skema_split", {}),
        }

        portable = SiPekaPortableModel(bundle)

        # Verifikasi akurasi numerik terhadap PySpark PipelineModel pada 25 baris sampel
        sampel_rows = df_ref.limit(25).collect()
        spark_preds = (
            pmodel.transform(df_ref.limit(25))
            .select("titik_lokasi", "tanggal", "prediction")
            .collect()
        )
        selisih_maks = 0.0
        for r_in, r_sp in zip(sampel_rows, spark_preds):
            d_in = r_in.asDict()
            hist_override = {c: float(d_in[c]) for c in FITUR_HISTORIS}
            p_py = portable.predict_one(
                titik_lokasi=d_in["titik_lokasi"],
                tanggal=d_in["tanggal"],
                kecamatan=d_in["kecamatan"],
                wilayah=d_in["wilayah"],
                fitur_historis_override=hist_override,
            )["prediksi_m3"]
            p_sp = round(max(0.0, float(r_sp["prediction"])), 4)
            selisih_maks = max(selisih_maks, abs(p_py - p_sp))

        # Bangun file siap pakai untuk peta Laravel dari seluruh 974 entri dataset_titik_phb-v3.json
        tgl_mulai_forecast = max_tgl + timedelta(days=1)
        v3_forecast: list[dict[str, Any]] = []
        for item in daftar_v3:
            kunci = item["titik_lokasi"]
            fc_30 = portable.forecast_titik(kunci, tgl_mulai_forecast, horizon_hari=30)
            nilai_30 = [d["prediksi_m3"] for d in fc_30]
            rata_harian = round(float(np.mean(nilai_30)), 3)
            total_30 = round(float(np.sum(nilai_30)), 2)
            v3_forecast.append(
                {
                    **item,
                    "ada_di_histori_sipeka": kunci in set(semua_lok_pd["titik_lokasi"]),
                    "rata_prediksi_harian_m3": rata_harian,
                    "total_prediksi_30hari_m3": total_30,
                    "prediksi_hari_pertama_m3": nilai_30[0] if nilai_30 else 0.0,
                    "model_version": versi_model,
                }
            )

        SERVING_DIR.mkdir(parents=True, exist_ok=True)
        v3_serving_path = SERVING_DIR / "titik_phb_v3_forecast.json"
        v3_serving_path.write_text(
            json.dumps(
                {
                    "model_version": versi_model,
                    "exported_at": bundle["exported_at"],
                    "sumber_titik": str(path_v3),
                    "jumlah_entri_v3": len(v3_forecast),
                    "jumlah_titik_unik_v3": len(peta_v3),
                    "jumlah_cocok_histori_sipeka": sum(
                        1 for x in v3_forecast if x["ada_di_histori_sipeka"]
                    ),
                    "forecast_mulai": tgl_mulai_forecast.isoformat(),
                    "horizon_hari": 30,
                    "data": v3_forecast,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # Simpan ke .pkl (versi aktif & berversi)
        pkl_versioned = MODELS_DIR / f"{versi_model}.pkl"
        with open(PKL_ACTIVE_PATH, "wb") as fp:
            pickle.dump(bundle, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(pkl_versioned, "wb") as fp:
            pickle.dump(bundle, fp, protocol=pickle.HIGHEST_PROTOCOL)

        # Simpan juga kompatibel dengan joblib
        joblib_path = MODELS_DIR / "active_model.joblib"
        joblib.dump(bundle, joblib_path, compress=3)

        # Simpan ke .h5 (versi aktif & berversi)
        h5_versioned = MODELS_DIR / f"{versi_model}.h5"
        simpan_ke_h5(H5_ACTIVE_PATH, bundle, v3_forecast)
        simpan_ke_h5(h5_versioned, bundle, v3_forecast)

        # Perbarui models/active_model.json agar mencantumkan path .pkl dan .h5
        active["pkl_path"] = str(PKL_ACTIVE_PATH)
        active["pkl_versioned_path"] = str(pkl_versioned)
        active["h5_path"] = str(H5_ACTIVE_PATH)
        active["h5_versioned_path"] = str(h5_versioned)
        active["titik_v3_forecast_path"] = str(v3_serving_path)
        ACTIVE_MODEL_FILE.write_text(json.dumps(active, indent=2), encoding="utf-8")

        hasil = {
            "versi_model": versi_model,
            "pkl_active": str(PKL_ACTIVE_PATH),
            "pkl_versioned": str(pkl_versioned),
            "h5_active": str(H5_ACTIVE_PATH),
            "h5_versioned": str(h5_versioned),
            "titik_v3_forecast": str(v3_serving_path),
            "jumlah_profil_titik": len(profil_titik),
            "jumlah_entri_v3": len(v3_forecast),
            "jumlah_cocok_v3_sipeka": sum(
                1 for x in v3_forecast if x["ada_di_histori_sipeka"]
            ),
            "max_abs_diff_vs_spark": round(selisih_maks, 8),
        }
        print("[Export] Berhasil mengekspor model ke .pkl dan .h5:")
        print(json.dumps(hasil, indent=2))
        return hasil
    finally:
        if milik_spark_sendiri:
            spark.stop()


def muat_model_pkl(path_pkl: str | Path = PKL_ACTIVE_PATH) -> SiPekaPortableModel:
    """Helper untuk memuat `active_model.pkl` tanpa Spark."""
    with open(path_pkl, "rb") as fp:
        bundle = pickle.load(fp)
    return SiPekaPortableModel(bundle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ekspor model PySpark aktif ke .pkl dan .h5")
    parser.parse_args()
    ekspor_model_aktif()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
