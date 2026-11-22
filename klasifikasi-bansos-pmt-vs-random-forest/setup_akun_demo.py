from __future__ import annotations

from getpass import getpass

from src.auth import DEFAULT_ACCOUNT_FILE, create_operator_account


def main() -> None:
    print("BUAT AKUN OPERATOR LOKAL")
    print("Kata sandi: minimal 8 karakter, ada huruf besar, huruf kecil, angka, dan simbol; tanpa spasi.\n")

    if DEFAULT_ACCOUNT_FILE.exists():
        jawab = input("Akun operator sudah ada. Timpa dengan akun baru? [y/N]: ").strip().lower()
        if jawab != "y":
            raise SystemExit("Dibatalkan. Akun lama tidak diubah.")

    username = input("Nama pengguna [operator]: ").strip() or "operator"
    display_name = input("Nama tampilan [Operator Kelurahan]: ").strip() or "Operator Kelurahan"
    tampil = input("Tampilkan kata sandi saat diketik? (pilih y jika terminal tidak merespons) [y/N]: ").strip().lower() == "y"
    baca = input if tampil else getpass
    password = baca("Kata sandi: ")
    confirm = baca("Ulangi kata sandi: ")
    if password != confirm:
        raise SystemExit("Kata sandi dan konfirmasi tidak sama.")

    try:
        path = create_operator_account(username=username, password=password, display_name=display_name)
    except ValueError as exc:
        raise SystemExit(f"Gagal membuat akun: {exc}") from exc
    print(f"\nAkun berhasil dibuat: {path}")
    print("Sekarang jalankan: python aplikasi-skripsi.py")


if __name__ == "__main__":
    main()
