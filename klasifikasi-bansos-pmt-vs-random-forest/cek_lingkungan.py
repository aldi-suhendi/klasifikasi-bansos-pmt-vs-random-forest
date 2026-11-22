from __future__ import annotations

from src.environment import collect_environment, validate_environment


def main() -> None:
    values = validate_environment(include_gui=True)
    print("LINGKUNGAN SESUAI")
    for name, value in values.items():
        print(f"{name:15}: {value}")


if __name__ == "__main__":
    main()
