# Ringkasan Model Prediksi Volume Pengangkutan Sampah (SiPeka — Step 6 & Auto-Retraining)

Dokumen ini merangkum arsitektur, pembagian data, fitur, hasil evaluasi, analisis error, serta mekanisme **Continuous Auto-Retraining (Champion vs Challenger)** dan ekspor model **`.pkl` / `.h5`** untuk integrasi dashboard **Laravel**.

---

## 1. Identitas & Tujuan Model

- **Champion Aktif (`models/active_model.json`):** `linearregression_v2_20261123_104848` (dipromosikan otomatis dari baseline `linearregression_v2_20261121_170617`)
- **Format Tersimpan:**
  - PySpark `PipelineModel`: `models/linearregression_v2_20261123_104848/`
  - Pickle / Joblib (tanpa Spark, inferensi `< 1 ms`): `models/active_model.pkl` & `models/active_model.joblib`
  - HDF5: `models/active_model.h5`
- **Jenis Analitik:** **Regresi** (memprediksi nilai kontinu, bukan klasifikasi)
- **Target Prediksi:** `volume_sampah_perhari` (satuan **`m³`**) per `titik_lokasi` per hari
- **Master Lokasi GIS:** `Dataset/dataset_titik_phb-v3.json` (`974` titik PHB/LPS SILIKA DLH DKI Jakarta, `963` nama unik, `882` titik cocok langsung dengan histori pengangkutan SiPeka).

---

## 2. Dataset & Skema Pembagian Waktu (Split Kronologis 70 / 15 / 15)

Dari `1.786.773` baris mentah tahun `2018`, `2020`, dan `2021` (tahun `2019` tidak tersedia pada sumber data), setelah validasi ingestion (`1.784.956` valid, `1.817` karantina), pembuangan target kosong (`491.292` baris), normalisasi tanda baca `titik_lokasi`, dan agregasi penjumlahan duplikat harian `(titik_lokasi, tanggal)`, diperoleh **`1.068.562` baris bersih** (`2.991` titik lokasi unik).

Pembagian data dilakukan berbasis urutan tanggal ketat (**`train < validation < test`**, tanpa *random split*) untuk mencegah *data leakage*:

| Partisi | Rentang Tanggal | Jumlah Baris | Proporsi Aktual | Jumlah `titik_lokasi` |
|---------|-----------------|-------------:|----------------:|----------------------:|
| **Train** | `2018-01-01` s.d. `2021-07-03` | `748.979` | `70,09%` | `2.987` |
| **Validation** | `2021-07-04` s.d. `2021-10-02` | `160.615` | `15,03%` | `1.768` |
| **Test** | `2021-10-03` s.d. `2021-12-31` | `158.968` | `14,88%` | `1.767` |

- **Penanganan Outlier saat Training:** baris dengan `is_outlier == 1` (`> P99,9 = 65,52 m³`, sebanyak `1.068` baris atau `0,10%`, termasuk `72` anomali pada Juli 2018 hingga `156.249.999,0 m³`) dikecualikan **hanya dari data latih (`train`)**, sedangkan `validation` dan `test` dievaluasi penuh apa adanya.

---

## 3. Daftar Fitur (`v1` vs `v2`)

### A. Fitur Kategorik Lokasi (3 Kolom)
- `titik_lokasi` (`2.991` titik unik global)
- `kecamatan` (`45` kecamatan)
- `wilayah` (`5` kota administrasi DKI Jakarta)

### B. Fitur Kalender — `v1` (14 Kolom)
- `tahun`, `bulan`, `hari_dalam_bulan`, `hari_dalam_minggu` (`1=Minggu` .. `7=Sabtu`), `is_weekend`, `minggu_ke`, `kuartal`, `hari_ke_dalam_tahun`
- Encoding siklik: `bulan_sin`, `bulan_cos`, `hari_sin`, `hari_cos`
- Indikator domain: `is_musim_hujan` (Oktober–April) dan `is_hari_libur` (hari libur nasional SKB 3 Menteri)

### C. Fitur Historis Bebas-Leakage — Tambahan `v2` (10 Kolom, Total 27 Fitur di `v2`)
Seluruh fitur historis dihitung per `titik_lokasi` menggunakan window masa lalu murni (`rowsBetween(..., -1)`) dari volume yang di-winsorize pada `P99,9 = 65,52 m³`:
- **Lag harian:** `lag_1`, `lag_7`, `lag_14`
- **Rolling window:** `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_30`, `rolling_std_7`, `rolling_max_7`
- **Profil kumulatif titik:** `titik_hist_mean` (rata-rata kumulatif masa lalu) dan `titik_hist_zero_rate` (proporsi hari bervolume `0 m³` di masa lalu)

---

## 4. Arsitektur Pipeline & Hyperparameter Model

### Pipeline Model Terpilih (`LinearRegression v2`)
1. `StringIndexer` (`titik_lokasi`, `kecamatan`, `wilayah`; `handleInvalid="keep"`)
2. `OneHotEncoder` (`titik_lokasi_ohe`, `kecamatan_ohe`, `wilayah_ohe`; `handleInvalid="keep"`)
3. `VectorAssembler` (`24` fitur numerik + `3` vektor OHE; `handleInvalid="skip"`)
4. `StandardScaler` (`withMean=False`, `withStd=True` agar representasi vektor tetap *sparse* di memori)
5. `LinearRegression` dengan regularisasi **ElasticNet** (`regParam = 0,05` pada Champion aktif, `elasticNetParam = 0,1`, `maxIter = 100`)
6. Post-processing fisik: prediksi di-clamp `>= 0,0 m³` (`F.greatest(0.0, prediction)`).

---

## 5. Tabel Perbandingan Hasil Evaluasi

| Model | Versi Fitur | RMSE Valid | MAE Valid | R² Valid | RMSE Test | MAE Test | R² Test | Baris Test Dinilai |
|-------|-------------|-----------:|----------:|---------:|----------:|---------:|--------:|-------------------:|
| Baseline (mean per titik) | `-` | `3,9426` | `1,8971` | `-0,3089` | `4,1141` | `1,5136` | `-0,0220` | `158.968 / 158.968` |
| LinearRegression | `v1` | `4,0781` | `2,1675` | `-0,4004` | `4,2482` | `1,8090` | `-0,0897` | `158.968 / 158.968` |
| RandomForest (`50`, depth `8`, min `5`) | `v1` | `3,9283` | `1,9235` | `-0,2994` | `4,0993` | `1,4795` | `-0,0147` | `158.968 / 158.968` |
| GBT (`30`, depth `6`, step `0,1`) | `v1` | `4,1169` | `1,9077` | `-0,4271` | `4,9258` | `1,8406` | `-0,4651` | `158.968 / 158.968` |
| LinearRegression (`regParam=0,01`) | `v2` | `3,2970` | `0,9474` | `+0,0848` | `3,7404` | `0,9284` | `+0,1555` | `158.918 / 158.968` |
| **LinearRegression (`regParam=0,05`)** *(Champion Aktif)* | **`v2`** | **`3,2933`** | **`0,9446`** | **`+0,0869`** | **`3,7333`** | **`0,8952`** | **`+0,1587`** | `158.918 / 158.968` |
| RandomForest (`50`, depth `6`, min `10`) | `v2` | `3,6390` | `1,4169` | `-0,1149` | `3,7520` | **`0,7462`** | `+0,1503` | `158.918 / 158.968` |
| GBT (`30`, depth `4`, step `0,1`) | `v2` | `3,5662` | `1,0763` | `-0,0707` | `3,7416` | `0,7546` | `+0,1549` | `158.918 / 158.968` |

---

## 6. Mekanisme Auto-Retraining & Ekspor `.pkl` / `.h5`

1. **Deteksi Data Baru (SHA-256 Fingerprint):** `scripts/auto_retrain.py` dan `src/modeling/train.py` memeriksa sidik jari file di `Dataset/` dan `data/processed/features_v2`. Jika belum ada perubahan data, pelatihan dilewati otomatis (`SKIPPED_NO_NEW_DATA`).
2. **Gerbang Champion vs Challenger:** Saat ada data baru (atau `--force`), kandidat Challenger dilatih pada split kronologis `70/15/15` terbaru dan dibandingkan dengan Champion aktif pada partisi `validation`. Model baru hanya dipromosikan ke `models/active_model.json` jika `RMSE validation` lebih rendah dari Champion.
3. **Ekspor Otomatis ke `.pkl` & `.h5`:** `src/modeling/export_model.py` mengekstrak bobot model aktif ke `models/active_model.pkl` dan `models/active_model.h5` (terverifikasi `max_abs_diff = 0,0` terhadap PySpark) serta memperbarui `data/serving/titik_phb_v3_forecast.json` (`974` titik PHB v3) untuk dikonsumsi langsung oleh dashboard Laravel.
