"""Merge fixture lists from multiple cricket data providers."""

from __future__ import annotations

from typing import Dict, List

from app.sportmonks_fixtures import fixture_dedupe_key

_SOURCE_PRIORITY = {"sportmonks": 0, "cricapi": 1}


def merge_fixture_lists(*fixture_groups: List[Dict]) -> List[Dict]:
    """Union fixtures from multiple APIs, deduping identical team/date pairs.

    When both providers list the same match, keep the Sportmonks entry because
    it usually has better venue metadata.
    """
    best_by_key: Dict[tuple, Dict] = {}

    for fixtures in fixture_groups:
        for fixture in fixtures:
            key = fixture_dedupe_key(fixture)
            if not key[0] or not key[1] or not key[2]:
                continue
            existing = best_by_key.get(key)
            if existing is None:
                best_by_key[key] = fixture
                continue
            new_priority = _SOURCE_PRIORITY.get(fixture.get("source") or "", 99)
            old_priority = _SOURCE_PRIORITY.get(existing.get("source") or "", 99)
            if new_priority < old_priority:
                best_by_key[key] = fixture
            elif new_priority == old_priority and not existing.get("venue") and fixture.get("venue"):
                best_by_key[key] = fixture

    merged = list(best_by_key.values())
    merged.sort(
        key=lambda item: (
            item.get("match_date") or "",
            item.get("start_time_utc") or "",
            item.get("label") or "",
        )
    )
    return merged
