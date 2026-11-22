from __future__ import annotations

from getpass import getpass

from src.auth import AuthenticationError, create_operator_account, verify_operator_credentials


def main() -> None:
    print("BUAT / GANTI AKUN OPERATOR LOKAL")
    print("Aplikasi menggunakan satu akun operator. Proses ini mengganti akun lama.")

    current_username = input("Nama pengguna lama [operator]: ").strip() or "operator"
    current_password = getpass("Kata sandi lama: ")
    try:
        profile = verify_operator_credentials(current_username, current_password)
    except AuthenticationError as exc:
        raise SystemExit(f"Konfigurasi akun bermasalah: {exc}") from exc
    if profile is None:
        raise SystemExit("Nama pengguna atau kata sandi lama tidak sesuai.")

    new_username = input("Nama pengguna baru: ").strip()
    display_name = input("Nama tampilan [Operator Kelurahan]: ").strip() or "Operator Kelurahan"
    new_password = getpass("Kata sandi baru (minimal 8 karakter): ")
    confirm_password = getpass("Ulangi kata sandi baru: ")
    if new_password != confirm_password:
        raise SystemExit("Kata sandi baru dan konfirmasi tidak sama.")

    try:
        path = create_operator_account(
            username=new_username,
            password=new_password,
            display_name=display_name,
        )
    except ValueError as exc:
        raise SystemExit(f"Gagal membuat akun: {exc}") from exc
    print(f"Akun operator berhasil diperbarui. Konfigurasi: {path}")


if __name__ == "__main__":
    main()
