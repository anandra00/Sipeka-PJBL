"""Orchestrator auto-retrain end-to-end SiPeka.

Menjalankan siklus berkelanjutan:
  1. Deteksi apakah ada file baru/berubah di `Dataset/`
  2. Jika ada perubahan (atau `--run-etl`), jalankan Step 3 (`ingest.py`) & Step 4 (`clean_features.py`)
  3. Jalankan Step 6 auto-retrain (`src/modeling/train.py`, Champion vs Challenger)
  4. Jika model baru dipromosikan atau data berubah (atau `--force`), jalankan Step 7 (`src/modeling/score.py`)

Dapat dijalankan:
  - Sekali jalan via Windows Task Scheduler / Cron: `python scripts/auto_retrain.py`
  - Mode daemon berulang: `python scripts/auto_retrain.py --watch --interval-minutes 1440`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
STATE_FILE = BASE_DIR / "models" / "pipeline_state.json"


def fingerprint_dataset() -> str:
    dataset_dir = BASE_DIR / "Dataset"
    h = hashlib.sha256()
    if dataset_dir.exists():
        for p in sorted(dataset_dir.glob("*")):
            if p.is_file():
                st = p.stat()
                h.update(f"{p.name}|{st.st_size}|{int(st.st_mtime)}".encode())
    return h.hexdigest()


def baca_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def simpan_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def jalankan_langkah(nama: str, cmd: list[str]) -> None:
    print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] >>> {nama}: {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR)
    env.setdefault("SPARK_DRIVER_MEMORY", "3g" if os.name == "nt" else "1g")
    env.setdefault("SPARK_MASTER", "local[4]")
    env.setdefault("PYSPARK_PYTHON", sys.executable)
    env.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    hasil = subprocess.run(cmd, cwd=str(BASE_DIR), env=env)
    if hasil.returncode != 0:
        raise RuntimeError(f"Langkah '{nama}' gagal dengan exit code {hasil.returncode}")


def jalankan_siklus(args: argparse.Namespace) -> None:
    state = baca_state()
    fp_raw = fingerprint_dataset()
    raw_berubah = fp_raw != state.get("dataset_fingerprint")
    features_v2_ada = (BASE_DIR / "data" / "processed" / "features_v2").exists()

    if args.run_etl or not features_v2_ada or (raw_berubah and state.get("dataset_fingerprint") is not None):
        jalankan_langkah("Step 3 Ingestion", [sys.executable, "src/ingestion/ingest.py"])
        cmd_clean = [sys.executable, "src/processing/clean_features.py"]
        if args.postgres:
            cmd_clean.append("--postgres")
        jalankan_langkah("Step 4 Cleaning & Features", cmd_clean)
    else:
        print("[Pipeline] Dataset mentah belum berubah; melewati Step 3-4.")

    active_file = BASE_DIR / "models" / "active_model.json"
    versi_sebelum = None
    if active_file.exists():
        try:
            versi_sebelum = json.loads(active_file.read_text(encoding="utf-8")).get("versi_model")
        except Exception:
            versi_sebelum = None

    cmd_train = [sys.executable, "src/modeling/train.py", "--mode", args.mode]
    if args.force:
        cmd_train.append("--force")
    jalankan_langkah("Step 6 Auto-Retrain (Champion vs Challenger)", cmd_train)

    versi_sesudah = None
    if active_file.exists():
        try:
            versi_sesudah = json.loads(active_file.read_text(encoding="utf-8")).get("versi_model")
        except Exception:
            versi_sesudah = None

    model_berubah = versi_sesudah != versi_sebelum
    serving_summary = BASE_DIR / "data" / "serving" / "active_model_summary.json"
    perlu_score = args.force or raw_berubah or model_berubah or not serving_summary.exists()
    if perlu_score:
        cmd_score = [
            sys.executable,
            "src/modeling/score.py",
            "--horizon",
            str(args.horizon),
        ]
        if args.postgres:
            cmd_score.append("--postgres")
        jalankan_langkah("Step 7 Batch Scoring & Forecast", cmd_score)
    else:
        print("[Pipeline] Data serving sudah mutakhir; melewati Step 7.")

    simpan_state(
        {
            "dataset_fingerprint": fp_raw,
            "last_run_at": datetime.now().isoformat(timespec="seconds"),
            "mode": args.mode,
            "horizon": args.horizon,
        }
    )
    print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] Siklus auto-retrain selesai.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Runner otomatis pipeline SiPeka")
    parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--force", action="store_true", help="Paksa retrain & scoring")
    parser.add_argument("--run-etl", action="store_true", help="Paksa jalankan ulang Step 3-4")
    parser.add_argument("--postgres", action="store_true", help="Tulis output ke PostgreSQL")
    parser.add_argument("--watch", action="store_true", help="Jalankan terus secara periodik")
    parser.add_argument("--interval-minutes", type=int, default=1440, help="Interval watch (menit)")
    args = parser.parse_args()

    if not args.watch:
        jalankan_siklus(args)
        return 0

    print(f"[Watch Mode] Auto-retrain berjalan setiap {args.interval_minutes} menit. Tekan Ctrl+C untuk berhenti.")
    while True:
        try:
            jalankan_siklus(args)
        except Exception as exc:
            print(f"[Watch Mode] Error pada siklus ini: {exc}", flush=True)
        time.sleep(max(60, args.interval_minutes * 60))


if __name__ == "__main__":
    raise SystemExit(main())
