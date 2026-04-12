"""Stub module — delegates to the real implementation in email_triage_env."""
from email_triage_env.server.app import app  # noqa: F401
from email_triage_env.server.app import main as _real_main


def main(host: str = "0.0.0.0", port: int = 8000):
    _real_main(host=host, port=port)


if __name__ == "__main__":
    main()
