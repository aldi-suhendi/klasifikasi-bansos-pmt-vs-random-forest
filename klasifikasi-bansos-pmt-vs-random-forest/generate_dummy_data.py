from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_DATE = pd.Timestamp("2025-12-31")
TITLE = "DATA DUMMY SINTETIS - bukan data warga sungguhan"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def tilted_choice(rng, categories, base_probs, effects, z, strength=1.0):
    base = np.asarray(base_probs, dtype=float)
    base = base / base.sum()
    eff = np.asarray(effects, dtype=float)
    logits = np.log(base)[None, :] + strength * z[:, None] * eff[None, :]
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    cumulative = probs.cumsum(axis=1)
    draw = rng.random(len(z))[:, None]
    index = (draw > cumulative).sum(axis=1).clip(0, len(categories) - 1)
    return np.asarray(categories, dtype=object)[index]


def binary_asset(rng, z, base_rate, slope):
    intercept = np.log(base_rate / (1 - base_rate))
    return (rng.random(len(z)) < sigmoid(intercept + slope * z)).astype(int)


def generate(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=rows)

    jobs = [
        ("KARYAWAN SWASTA", 0.32, 0.5), ("BURUH HARIAN LEPAS", 0.22, -1.2),
        ("WIRASWASTA", 0.19, 0.3), ("BURUH PABRIK", 0.075, 0.1),
        ("PEKERJA JASA", 0.07, -0.6), ("PEDAGANG", 0.065, -0.5),
        ("PEGAWAI HONORER", 0.01, -0.5), ("TUKANG", 0.01, -0.9),
        ("APARATUR PEMERINTAHAN", 0.006, 1.2), ("ASISTEN RUMAH TANGGA", 0.005, -1.4),
        ("SATPAM", 0.003, 0.0), ("TIDAK/BELUM BEKERJA", 0.003, -1.6),
        ("MEKANIK", 0.002, 0.2), ("KURIR", 0.002, -0.8), ("SOPIR", 0.002, -0.7),
        ("TNI", 0.0015, 1.4), ("PNS", 0.0015, 1.5), ("PETANI", 0.0015, -1.2),
        ("KARYAWAN BUMN", 0.0015, 1.4), ("POLRI", 0.0015, 1.4),
        ("PENSIUNAN", 0.001, 1.0), ("BIDAN", 0.001, 1.3), ("PENATA BUSANA", 0.0005, 0.4),
        ("BURUH BANGUNAN", 0.0005, -1.2), ("DOKTER", 0.0005, 1.8),
        ("KARYAWAN BUMD", 0.0005, 1.3),
    ]
    job = tilted_choice(rng, [j[0] for j in jobs], [j[1] for j in jobs], [j[2] for j in jobs], z)

    education = tilted_choice(
        rng,
        ["TIDAK TAMAT SD/SEDERAJAT", "SD/SEDERAJAT", "SMP/SEDERAJAT", "SMA/SEDERAJAT", "DIPLOMA/SARJANA"],
        [0.03, 0.15, 0.20, 0.54, 0.08], [-1.2, -0.8, -0.4, 0.1, 1.2], z,
    )
    n_member = tilted_choice(
        rng, [1, 2, 3, 4, 5, 6, 7], [0.03, 0.15, 0.28, 0.45, 0.07, 0.015, 0.005],
        [0.2, 0.2, 0.1, 0.0, -0.3, -0.5, -0.6], z, strength=0.6,
    ).astype(int)
    house_status = tilted_choice(
        rng, ["MILIK SENDIRI", "KONTRAK/SEWA", "BEBAS SEWA/NUMPANG"], [0.90, 0.08, 0.02], [0.2, -0.8, -1.0], z,
    )
    elderly = tilted_choice(rng, [0, 1, 2], [0.91, 0.07, 0.02], [0.0, -0.3, -0.5], z, 0.5).astype(int)
    disability = tilted_choice(
        rng, ["TIDAK ADA", "FISIK", "MENTAL", "RUNGU/WICARA", "SENSORIK"],
        [0.9965, 0.0015, 0.0005, 0.0008, 0.0007], [0.0, -0.8, -0.8, -0.8, -0.8], z, 0.5,
    )
    chronic = tilted_choice(
        rng, ["TIDAK ADA", "HIPERTENSI", "DIABETES", "JANTUNG", "ASMA KRONIS", "TBC", "REMATIK"],
        [0.95, 0.02, 0.012, 0.01, 0.007, 0.0035, 0.0015], [0.0, -0.2, -0.2, -0.3, -0.3, -0.8, -0.4], z, 0.5,
    )
    floor = tilted_choice(
        rng, ["GRANIT/MARMER", "KERAMIK", "UBIN/TEGEL/TERASO", "SEMEN"], [0.004, 0.925, 0.043, 0.028], [2.0, 0.2, -0.3, -1.4], z,
    )
    wall = tilted_choice(rng, ["TEMBOK", "KAYU/PAPAN"], [0.975, 0.025], [0.1, -1.2], z)
    roof = tilted_choice(
        rng, ["BETON", "GENTENG", "ASBES", "SENG", "RUMBIA"], [0.001, 0.77, 0.22, 0.01, 0.002], [1.5, 0.2, -0.3, -1.0, -1.5], z,
    )
    power = tilted_choice(
        rng, [450, 900, 1300, 2200, 3500], [0.09, 0.71, 0.19, 0.0095, 0.0005], [-1.2, -0.1, 0.8, 1.6, 2.0], z,
    ).astype(int)
    fuel = tilted_choice(
        rng, ["GAS 3 KG", "GAS 5,5 KG", "KAYU BAKAR"], [0.99, 0.006, 0.004], [-0.05, 1.5, -1.5], z, 0.6,
    )
    toilet_access = tilted_choice(rng, ["SENDIRI", "BERSAMA"], [0.993, 0.007], [0.1, -1.5], z)
    toilet_type = tilted_choice(rng, ["LEHER ANGSA", "CEMPLUNG"], [0.99, 0.01], [0.1, -1.5], z)
    sewage = tilted_choice(rng, ["TANGKI SEPTIK", "LUBANG TANAH"], [0.91, 0.09], [0.3, -1.2], z)
    water = tilted_choice(
        rng, ["SUMUR BOR/POMPA", "AIR ISI ULANG", "AIR KEMASAN BERMEREK"], [0.51, 0.36, 0.13], [-0.5, 0.0, 0.9], z,
    )

    frame = pd.DataFrame(
        {
            "NAMA_KEPALA_KELUARGA": [f"WARGA DUMMY {i:04d}" for i in range(1, rows + 1)],
            "TANGGAL_LAHIR": np.nan,
            "KECAMATAN": "PINANG",
            "DESA/KELURAHAN": "PAKOJAN",
            "ALAMAT": [f"ALAMAT DUMMY {i:04d}" for i in range(1, rows + 1)],
            "PENDIDIKAN": education,
            "PEKERJAAN": job,
            "JUMLAH-ART": n_member,
            "KEPEMILIKAN_RUMAH": house_status,
            "LANSIA": elderly,
            "DISABILITAS": disability,
            "SAKIT_KRONIS": chronic,
            "LANTAI_TERLUAS": floor,
            "DINDING_TERLUAS": wall,
            "ATAP_TERLUAS": roof,
            "SUMBER_PENERANGAN_UTAMA": "LISTRIK PLN",
            "DAYA_LISTRIK_TERPASANG": power,
            "BAHAN_BAKAR_MEMASAK": fuel,
            "FASILITAS_BAB": toilet_access,
            "JENIS_KLOSET": toilet_type,
            "PEMBUANGAN_AKHIR_TINJA": sewage,
            "SUMBER_AIR_MINUM": water,
            "LEMARI_ES": binary_asset(rng, z, 0.68, 1.0),
            "AC": binary_asset(rng, z, 0.11, 1.5),
            "TV_LAYAR_DATAR": binary_asset(rng, z, 0.87, 0.8),
            "KOMPUTER/LAPTOP": binary_asset(rng, z, 0.18, 1.0),
            "SEPEDA_MOTOR": binary_asset(rng, z, 0.97, 1.0),
            "SEPEDA": binary_asset(rng, z, 0.18, 0.2),
            "MOBIL": binary_asset(rng, z, 0.08, 1.6),
            "PERAHU": 0,
            "KAPAL_MOTOR": 0,
            "SMARTPHONE": binary_asset(rng, z, 0.98, 1.0),
            "TERNAK_BESAR": binary_asset(rng, z, 0.002, 0.0),
            "TERNAK_KECIL": binary_asset(rng, z, 0.012, 0.0),
        }
    )

    age = np.clip(np.round(rng.normal(44, 9, rows)), 20, 85).astype(int)
    day_offset = rng.integers(0, 365, rows)
    birth = [REFERENCE_DATE - pd.DateOffset(years=int(a)) - pd.Timedelta(days=int(d)) for a, d in zip(age, day_offset)]
    frame["TANGGAL_LAHIR"] = pd.to_datetime(birth).normalize()

    low_jobs = {"BURUH HARIAN LEPAS", "PEDAGANG", "PEKERJA JASA", "TUKANG", "ASISTEN RUMAH TANGGA", "KURIR",
                "SOPIR", "PETANI", "TIDAK/BELUM BEKERJA", "BURUH BANGUNAN", "PEGAWAI HONORER"}
    high_jobs = {"PNS", "TNI", "POLRI", "PENSIUNAN", "KARYAWAN BUMN", "KARYAWAN BUMD",
                 "APARATUR PEMERINTAHAN", "BIDAN", "DOKTER"}
    score = np.zeros(rows)
    score += np.where(frame["PEKERJAAN"].isin(low_jobs), 1.3, np.where(frame["PEKERJAAN"].isin(high_jobs), -1.2, 0.0))
    score += frame["KEPEMILIKAN_RUMAH"].map({"MILIK SENDIRI": 0.0, "KONTRAK/SEWA": 0.7, "BEBAS SEWA/NUMPANG": 1.0}).to_numpy()
    score += frame["LANTAI_TERLUAS"].map({"GRANIT/MARMER": -1.2, "KERAMIK": 0.0, "UBIN/TEGEL/TERASO": 0.3, "SEMEN": 1.0}).to_numpy()
    score += np.where(frame["DINDING_TERLUAS"] == "KAYU/PAPAN", 0.8, 0.0)
    score += frame["ATAP_TERLUAS"].map({"BETON": -1.0, "GENTENG": 0.0, "ASBES": 0.3, "SENG": 0.7, "RUMBIA": 1.0}).to_numpy()
    score += frame["DAYA_LISTRIK_TERPASANG"].map({450: 0.8, 900: 0.0, 1300: -0.5, 2200: -1.0, 3500: -1.3}).to_numpy()
    score += frame["PENDIDIKAN"].map({"TIDAK TAMAT SD/SEDERAJAT": 0.8, "SD/SEDERAJAT": 0.5, "SMP/SEDERAJAT": 0.2,
                                       "SMA/SEDERAJAT": 0.0, "DIPLOMA/SARJANA": -0.7}).to_numpy()
    score += np.where(frame["PEMBUANGAN_AKHIR_TINJA"] == "LUBANG TANAH", 0.6, 0.0)
    score += np.where(frame["SUMBER_AIR_MINUM"] == "AIR KEMASAN BERMEREK", -0.5, 0.0)
    score += 0.25 * (frame["JUMLAH-ART"].to_numpy() >= 5) + 0.3 * (frame["LANSIA"].to_numpy() > 0)
    score += 0.4 * (frame["SAKIT_KRONIS"] != "TIDAK ADA").to_numpy() + 0.4 * (frame["DISABILITAS"] != "TIDAK ADA").to_numpy()
    score += 0.6 * (1 - frame["LEMARI_ES"].to_numpy()) + 0.5 * (1 - frame["TV_LAYAR_DATAR"].to_numpy())
    score += 1.1 * (1 - frame["SEPEDA_MOTOR"].to_numpy()) + 0.8 * (1 - frame["SMARTPHONE"].to_numpy())
    score -= 1.3 * frame["AC"].to_numpy() + 1.6 * frame["MOBIL"].to_numpy() + 0.5 * frame["KOMPUTER/LAPTOP"].to_numpy()
    score += rng.normal(0, 0.9, rows)

    cutoff = np.quantile(score, 0.70)
    frame["KETERANGAN"] = np.where(score > cutoff, "LAYAK", "TIDAK LAYAK")
    return frame


def save_excel(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Sheet1", index=False, startrow=1)
        sheet = writer.sheets["Sheet1"]
        sheet["A1"] = TITLE
        sheet.freeze_panes = "A3"


def main() -> None:
    parser = argparse.ArgumentParser(description="Membuat data dummy sintetis untuk demo proyek.")
    parser.add_argument("--rows", type=int, default=2000, help="Jumlah baris (default 2000).")
    parser.add_argument("--seed", type=int, default=2026, help="Seed acak agar hasil bisa diulang.")
    parser.add_argument("--output", default="data/Data-Dummy.xlsx", help="Lokasi file Excel keluaran.")
    args = parser.parse_args()

    frame = generate(args.rows, args.seed)
    output = Path(args.output)
    save_excel(frame, output)
    share = (frame["KETERANGAN"] == "LAYAK").mean() * 100
    print(f"Data dummy dibuat: {output} ({len(frame)} baris, {frame.shape[1]} kolom, LAYAK {share:.1f}%)")


if __name__ == "__main__":
    main()
