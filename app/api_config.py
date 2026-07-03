"""Load local configuration without requiring a Streamlit runtime."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import toml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SECRETS_PATH = _PROJECT_ROOT / ".streamlit" / "secrets.toml"
_ENV_PATH = _PROJECT_ROOT / ".env"


def get_cricapi_key() -> Optional[str]:
    """Read CricAPI key from environment, .env, or Streamlit secrets file."""
    key = os.getenv("CRICAPI_KEY")
    if key:
        return key

    if _ENV_PATH.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(_ENV_PATH, override=False)
        except ImportError:
            pass
        key = os.getenv("CRICAPI_KEY")
        if key:
            return key

    if _SECRETS_PATH.exists():
        try:
            secrets = toml.load(_SECRETS_PATH)
        except Exception:
            secrets = {}
        key = secrets.get("CRICAPI_KEY")
        if key and key not in {"PASTE_YOUR_API_KEY_HERE", "your_cricapi_key_here"}:
            return key

    return None


def get_sportmonks_key() -> Optional[str]:
    """Read Sportmonks API key from environment, .env, or Streamlit secrets file."""
    key = os.getenv("SPORTMONKS_API_KEY")
    if key:
        return key

    if _ENV_PATH.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(_ENV_PATH, override=False)
        except ImportError:
            pass
        key = os.getenv("SPORTMONKS_API_KEY")
        if key:
            return key

    if _SECRETS_PATH.exists():
        try:
            secrets = toml.load(_SECRETS_PATH)
        except Exception:
            secrets = {}
        key = secrets.get("SPORTMONKS_API_KEY")
        if key and key not in {"PASTE_YOUR_API_KEY_HERE", "your_sportmonks_key_here"}:
            return key

    return None
