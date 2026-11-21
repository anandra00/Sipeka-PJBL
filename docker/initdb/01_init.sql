-- ============================================================
-- SiPeka — init script PostgreSQL
-- Dijalankan otomatis SEKALI saat volume postgres pertama kali dibuat.
-- Kalau perlu mengulang: docker compose down -v lalu up lagi.
-- ============================================================

-- Zona waktu: data DKI Jakarta memakai WIB
ALTER DATABASE sipeka SET timezone TO 'Asia/Jakarta';

-- Schema untuk tabel uji koneksi Step 1.
-- Tabel domain (predictions, model_metrics, dll) dibuat di Step 7,
-- bukan di sini, supaya Step 1 tetap sesuai lingkup PRD.
CREATE SCHEMA IF NOT EXISTS staging;

-- Tabel penanda bahwa lingkungan sudah siap.
CREATE TABLE IF NOT EXISTS staging.koneksi_test (
    id           SERIAL PRIMARY KEY,
    sumber       TEXT        NOT NULL,   -- 'pyspark' / 'python' / manual
    pesan        TEXT,
    dibuat_pada  TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE staging.koneksi_test IS
    'Tabel uji koneksi Step 1. Bukan tabel domain SiPeka.';
