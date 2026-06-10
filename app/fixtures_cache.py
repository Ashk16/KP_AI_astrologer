"""Persist cricket fixtures locally and refresh from CricAPI at most every 24 hours."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.cricapi_client import _fixture_id, _normalize_match, _request_matches

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_CACHE_PATH = _PROJECT_ROOT / "data" / "fixtures_cache.json"
BUNDLED_CACHE_PATH = _PROJECT_ROOT / "data" / "fixtures_cache_bundled.json"

PAST_DAYS = 20
FUTURE_DAYS = 30
REFRESH_INTERVAL = dt.timedelta(hours=24)
MATCH_OFFSETS = (0, 25, 50, 75, 100, 125, 150)


def fixture_window(reference_date: Optional[dt.date] = None) -> Tuple[dt.date, dt.date]:
    today = reference_date or dt.date.today()
    return today - dt.timedelta(days=PAST_DAYS), today + dt.timedelta(days=FUTURE_DAYS)


def _empty_cache() -> Dict[str, Any]:
    start, end = fixture_window()
    return {
        "last_refresh_utc": None,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "api_requests_used": 0,
        "fixtures": [],
        "cache_source": "empty",
    }


def _read_cache_file(path: Path, source: str) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        cache = json.load(fh)
    cache["cache_source"] = source
    return cache


def load_cache() -> Dict[str, Any]:
    """Load runtime cache first; fall back to bundled cache shipped with the repo."""
    runtime_cache = _read_cache_file(RUNTIME_CACHE_PATH, "runtime")
    if runtime_cache and runtime_cache.get("fixtures"):
        return runtime_cache

    bundled_cache = _read_cache_file(BUNDLED_CACHE_PATH, "bundled")
    if bundled_cache and bundled_cache.get("fixtures"):
        return bundled_cache

    return _empty_cache()


def save_runtime_cache(cache: Dict[str, Any]) -> None:
    RUNTIME_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in cache.items() if key != "cache_source"}
    with open(RUNTIME_CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def save_bundled_cache(cache: Dict[str, Any]) -> None:
    """Update the bundled cache file (for local maintenance before deploy)."""
    BUNDLED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in cache.items() if key != "cache_source"}
    with open(BUNDLED_CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _parse_refresh_time(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except ValueError:
        return None


def cache_is_stale(cache: Dict[str, Any], reference_date: Optional[dt.date] = None) -> bool:
    last_refresh = _parse_refresh_time(cache.get("last_refresh_utc"))
    if last_refresh is None:
        return True
    if dt.datetime.now(dt.timezone.utc) - last_refresh >= REFRESH_INTERVAL:
        return True

    window_start, window_end = fixture_window(reference_date)
    cached_start = cache.get("window_start")
    cached_end = cache.get("window_end")
    return cached_start != window_start.isoformat() or cached_end != window_end.isoformat()


def _fetch_raw_matches(api_key: str) -> Tuple[List[Dict[str, Any]], int]:
    raw_matches: List[Dict[str, Any]] = []
    seen_ids = set()
    requests_used = 0

    endpoints = [("currentMatches", None)]
    endpoints.extend(("matches", {"offset": offset}) for offset in MATCH_OFFSETS)

    for endpoint, params in endpoints:
        try:
            matches = _request_matches(endpoint, api_key, params)
            requests_used += 1
        except Exception:
            continue

        if not matches:
            continue

        for match in matches:
            fixture_id = _fixture_id(match)
            if fixture_id in seen_ids:
                continue
            seen_ids.add(fixture_id)
            raw_matches.append(match)

    return raw_matches, requests_used


def refresh_cache(api_key: str, reference_date: Optional[dt.date] = None) -> Dict[str, Any]:
    window_start, window_end = fixture_window(reference_date)
    raw_matches, requests_used = _fetch_raw_matches(api_key)

    fixtures: List[Dict[str, Any]] = []
    seen_fixture_ids = set()
    for match in raw_matches:
        fixture = _normalize_match(match)
        if fixture is None:
            continue
        if fixture.match_date < window_start or fixture.match_date > window_end:
            continue
        if fixture.fixture_id in seen_fixture_ids:
            continue
        seen_fixture_ids.add(fixture.fixture_id)
        fixtures.append(fixture.to_dict())

    fixtures.sort(
        key=lambda item: (
            item.get("match_date") or "",
            item.get("start_time_utc") or "",
        )
    )

    cache = {
        "last_refresh_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "api_requests_used": requests_used,
        "fixtures": fixtures,
        "cache_source": "runtime",
    }
    save_runtime_cache(cache)
    return cache


def ensure_cache(api_key: str, force_refresh: bool = False) -> Dict[str, Any]:
    cache = load_cache()
    if not force_refresh and not cache_is_stale(cache):
        return cache

    try:
        return refresh_cache(api_key)
    except Exception:
        if cache.get("fixtures"):
            return cache
        raise


def get_cached_fixtures_for_date(
    api_key: str,
    target_date: dt.date,
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    cache = ensure_cache(api_key, force_refresh=force_refresh)
    target_iso = target_date.isoformat()
    fixtures = [
        fixture
        for fixture in cache.get("fixtures", [])
        if fixture.get("match_date") == target_iso
    ]
    fixtures.sort(key=lambda item: item.get("start_time_utc") or "")
    return fixtures


def cache_status(cache: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cache = cache or load_cache()
    last_refresh = _parse_refresh_time(cache.get("last_refresh_utc"))
    next_refresh = None
    if last_refresh is not None:
        next_refresh = last_refresh + REFRESH_INTERVAL

    return {
        "last_refresh_utc": cache.get("last_refresh_utc"),
        "next_refresh_utc": next_refresh.isoformat() if next_refresh else None,
        "window_start": cache.get("window_start"),
        "window_end": cache.get("window_end"),
        "fixture_count": len(cache.get("fixtures", [])),
        "api_requests_used": cache.get("api_requests_used", 0),
        "is_stale": cache_is_stale(cache),
        "cache_source": cache.get("cache_source", "unknown"),
    }
