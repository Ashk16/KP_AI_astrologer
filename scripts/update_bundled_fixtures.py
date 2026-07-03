"""Refresh bundled fixtures from CricAPI + Sportmonks and save to the committed cache file.

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

from app.api_config import get_cricapi_key, get_sportmonks_key
from app.fixtures_cache import cache_status, refresh_cache, save_bundled_cache


def main() -> int:
    cricapi_key = get_cricapi_key()
    sportmonks_key = get_sportmonks_key()
    if not cricapi_key and not sportmonks_key:
        print("Configure at least one of CRICAPI_KEY or SPORTMONKS_API_KEY.")
        return 1

    cache = refresh_cache(cricapi_key=cricapi_key, sportmonks_key=sportmonks_key)
    save_bundled_cache(cache)
    status = cache_status(cache)

    print("Bundled fixtures updated:")
    print("  file: data/fixtures_cache_bundled.json")
    print(f"  fixtures: {status['fixture_count']}")
    print(f"  providers: {', '.join(status.get('providers') or [])}")
    print(f"  window: {status['window_start']} to {status['window_end']}")
    print(f"  last refresh: {status['last_refresh_utc']}")
    print(f"  requests: cricapi={status['cricapi_requests_used']}, sportmonks={status['sportmonks_requests_used']}")
    print("Commit this file before deploying to Streamlit Cloud.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
