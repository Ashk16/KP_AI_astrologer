"""Refresh bundled fixtures from CricAPI and save to the committed cache file.

Run locally before pushing/deploying so Streamlit Cloud always ships
with a recent match list even after the app sleeps or restarts.

Usage:
  python scripts/update_bundled_fixtures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api_config import get_cricapi_key
from app.fixtures_cache import cache_status, refresh_cache, save_bundled_cache


def main() -> int:
    api_key = get_cricapi_key()
    if not api_key:
        print("CRICAPI_KEY is not configured.")
        return 1

    cache = refresh_cache(api_key)
    save_bundled_cache(cache)
    status = cache_status(cache)

    print("Bundled fixtures updated:")
    print(f"  file: data/fixtures_cache_bundled.json")
    print(f"  fixtures: {status['fixture_count']}")
    print(f"  window: {status['window_start']} to {status['window_end']}")
    print(f"  last refresh: {status['last_refresh_utc']}")
    print("Commit this file before deploying to Streamlit Cloud.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
