from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from pathlib import Path
from typing import Any, Dict

PBKDF2_ALGORITHM = "sha256"
DEFAULT_ITERATIONS = 260_000
PASSWORD_MIN_LENGTH = 8


class AuthenticationError(RuntimeError):
    pass


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


DEFAULT_ACCOUNT_FILE = _project_root() / "config" / "operator_account.json"


def _derive_password_hash(password: str, salt: bytes, iterations: int) -> bytes:
    return hashlib.pbkdf2_hmac(
        PBKDF2_ALGORITHM,
        password.encode("utf-8"),
        salt,
        iterations,
    )


def validate_password_strength(password: str) -> None:
    missing = []
    if len(password) < PASSWORD_MIN_LENGTH:
        missing.append(f"minimal {PASSWORD_MIN_LENGTH} karakter")
    if not any(char.isupper() for char in password):
        missing.append("huruf besar")
    if not any(char.islower() for char in password):
        missing.append("huruf kecil")
    if not any(char.isdigit() for char in password):
        missing.append("angka")
    if not any(not char.isalnum() and not char.isspace() for char in password):
        missing.append("simbol")
    if any(char.isspace() for char in password):
        missing.append("tanpa spasi")
    if missing:
        raise ValueError(
            "Kata sandi harus menggunakan " + ", ".join(missing) + "."
        )


def create_operator_account(
    username: str,
    password: str,
    *,
    display_name: str = "Operator",
    account_file: str | Path = DEFAULT_ACCOUNT_FILE,
    iterations: int = DEFAULT_ITERATIONS,
) -> Path:
    username = username.strip().lower()
    if not username:
        raise ValueError("Nama pengguna tidak boleh kosong.")
    validate_password_strength(password)

    salt = secrets.token_bytes(16)
    password_hash = _derive_password_hash(password, salt, iterations)
    payload: Dict[str, Any] = {
        "version": 1,
        "username": username,
        "display_name": display_name.strip() or "Operator",
        "role": "operator",
        "enabled": True,
        "algorithm": f"pbkdf2_hmac_{PBKDF2_ALGORITHM}",
        "iterations": iterations,
        "salt": base64.b64encode(salt).decode("ascii"),
        "password_hash": base64.b64encode(password_hash).decode("ascii"),
    }

    path = Path(account_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_operator_account(account_file: str | Path = DEFAULT_ACCOUNT_FILE) -> Dict[str, Any]:
    path = Path(account_file)
    if not path.exists():
        raise AuthenticationError(
            f"Berkas akun operator tidak ditemukan: {path}. "
            "Pulihkan folder config atau jalankan utilitas penggantian kata sandi."
        )
    try:
        account = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthenticationError(f"Berkas akun operator tidak dapat dibaca: {exc}") from exc

    required = {"username", "password_hash", "salt", "iterations", "enabled"}
    missing = required.difference(account)
    if missing:
        raise AuthenticationError(
            "Konfigurasi akun operator tidak lengkap: " + ", ".join(sorted(missing))
        )
    return account


def verify_operator_credentials(
    username: str,
    password: str,
    account_file: str | Path = DEFAULT_ACCOUNT_FILE,
) -> Dict[str, Any] | None:
    account = load_operator_account(account_file)
    if not bool(account.get("enabled", False)):
        return None

    normalized_username = username.strip().lower()
    stored_username = str(account.get("username", "")).strip().lower()
    username_matches = hmac.compare_digest(normalized_username, stored_username)

    try:
        salt = base64.b64decode(account["salt"], validate=True)
        expected_hash = base64.b64decode(account["password_hash"], validate=True)
        iterations = int(account["iterations"])
    except (ValueError, TypeError) as exc:
        raise AuthenticationError("Konfigurasi hash akun operator tidak valid.") from exc

    actual_hash = _derive_password_hash(password, salt, iterations)
    password_matches = hmac.compare_digest(actual_hash, expected_hash)
    if not (username_matches and password_matches):
        return None

    return {
        "username": stored_username,
        "display_name": str(account.get("display_name") or "Operator"),
        "role": "operator",
    }


def change_operator_password(
    username: str,
    old_password: str,
    new_password: str,
    account_file: str | Path = DEFAULT_ACCOUNT_FILE,
) -> Path:
    account = verify_operator_credentials(username, old_password, account_file)
    if account is None:
        raise AuthenticationError("Nama pengguna atau kata sandi lama tidak sesuai.")
    return create_operator_account(
        username=account["username"],
        password=new_password,
        display_name=account["display_name"],
        account_file=account_file,
    )
