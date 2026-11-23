"""CLI & FastAPI Microservice ringan berbasis `models/active_model.pkl` untuk Laravel.

Berjalan TANPA PySpark / Java (waktu muat ~15 ms, prediksi < 1 ms).
Cara pakai dari Laravel:
  1. Via CLI (`Symfony\\Component\\Process\\Process`):
     `python scripts/predict_api.py --titik "Sungai Utan Kayu Jl Pengayoman" --tanggal 2026-11-24 --horizon 7`
  2. Via HTTP REST API (`Http::get('http://127.0.0.1:8000/...')`):
     `python scripts/predict_api.py --serve --port 8000`
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.export_model import (
    H5_ACTIVE_PATH,
    PKL_ACTIVE_PATH,
    SERVING_DIR,
    muat_model_pkl,
)


def buat_fastapi_app():
    from fastapi import FastAPI, HTTPException, Query

    app = FastAPI(
        title="SiPeka Prediction API (Laravel Bridge)",
        version="2.0.0",
        description="API prediksi volume sampah berbasis active_model.pkl & dataset_titik_phb-v3.json",
    )

    state = {"model": None, "mtime": None}

    def get_model():
        if not PKL_ACTIVE_PATH.exists():
            raise HTTPException(status_code=503, detail="active_model.pkl belum tersedia.")
        mtime = PKL_ACTIVE_PATH.stat().st_mtime
        if state["model"] is None or state["mtime"] != mtime:
            state["model"] = muat_model_pkl(PKL_ACTIVE_PATH)
            state["mtime"] = mtime
        return state["model"]

    @app.get("/api/model-info")
    def model_info():
        m = get_model()
        return {
            "versi_model": m.versi_model,
            "model": m.nama_model,
            "versi_fitur": m.versi_fitur,
            "pkl_path": str(PKL_ACTIVE_PATH),
            "h5_path": str(H5_ACTIVE_PATH),
            "jumlah_profil_titik": len(m.profil_titik),
            "metrik_validation": m.metrik_validation,
            "metrik_test": m.metrik_test,
        }

    @app.get("/api/predict")
    def predict(
        titik: str = Query(..., description="Nama titik lokasi / LPS"),
        tanggal: str = Query(default_factory=lambda: date.today().isoformat()),
        horizon: int = Query(1, ge=1, le=90),
    ):
        m = get_model()
        if horizon == 1:
            return m.predict_one(titik, tanggal)
        return {
            "titik_lokasi": titik,
            "tanggal_mulai": tanggal,
            "horizon_hari": horizon,
            "model_version": m.versi_model,
            "forecast": m.forecast_titik(titik, tanggal, horizon_hari=horizon),
        }

    @app.get("/api/titik-phb")
    def titik_phb(wilayah: str | None = None, kec: str | None = None):
        path = SERVING_DIR / "titik_phb_v3_forecast.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="titik_phb_v3_forecast.json belum tersedia.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        data = payload.get("data", [])
        if wilayah:
            w = wilayah.strip().upper()
            data = [
                x
                for x in data
                if w in str(x.get("wilayah", "")).upper()
                or w in str(x.get("kab", "")).upper()
            ]
        if kec:
            k = kec.strip().upper()
            data = [x for x in data if k == str(x.get("kec", "")).upper()]
        return {**payload, "jumlah_filter": len(data), "data": data}

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Prediksi cepat SiPeka dari active_model.pkl")
    parser.add_argument("--titik", type=str, help="Nama titik lokasi / LPS")
    parser.add_argument("--tanggal", type=str, default=date.today().isoformat())
    parser.add_argument("--horizon", type=int, default=1, help="Jumlah hari prediksi (1..90)")
    parser.add_argument("--serve", action="store_true", help="Jalankan REST API FastAPI untuk Laravel")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.serve:
        import uvicorn

        app = buat_fastapi_app()
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    model = muat_model_pkl(PKL_ACTIVE_PATH)
    if not args.titik:
        print(
            json.dumps(
                {
                    "versi_model": model.versi_model,
                    "model": model.nama_model,
                    "jumlah_profil_titik": len(model.profil_titik),
                    "metrik_validation": model.metrik_validation,
                    "metrik_test": model.metrik_test,
                },
                indent=2,
            )
        )
        return 0

    if args.horizon <= 1:
        hasil = model.predict_one(args.titik, args.tanggal)
    else:
        hasil = {
            "titik_lokasi": args.titik,
            "tanggal_mulai": args.tanggal,
            "horizon_hari": args.horizon,
            "model_version": model.versi_model,
            "forecast": model.forecast_titik(args.titik, args.tanggal, args.horizon),
        }
    print(json.dumps(hasil, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
