from __future__ import annotations

from getpass import getpass

from src.auth import AuthenticationError, change_operator_password


def main() -> None:
    print("GANTI KATA SANDI OPERATOR")
    username = input("Nama pengguna [operator]: ").strip() or "operator"
    old_password = getpass("Kata sandi lama: ")
    new_password = getpass("Kata sandi baru (min. 8 karakter, huruf besar/kecil, angka, simbol): ")
    confirm_password = getpass("Ulangi kata sandi baru: ")
    if new_password != confirm_password:
        raise SystemExit("Kata sandi baru dan konfirmasi tidak sama.")
    try:
        path = change_operator_password(username, old_password, new_password)
    except (AuthenticationError, ValueError) as exc:
        raise SystemExit(f"Gagal mengganti kata sandi: {exc}") from exc
    print(f"Kata sandi berhasil diperbarui. Konfigurasi: {path}")


if __name__ == "__main__":
    main()
