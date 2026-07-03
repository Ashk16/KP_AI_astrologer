"""Fetch and normalize Sportmonks fixtures into the shared fixture dict shape."""

from __future__ import annotations

import datetime as dt
import re
from typing import Any, Dict, List, Optional, Tuple

from app.sportmonks_client import SportmonksClient, SportmonksError, _unwrap_obj, resolve_venue_payload


def _parse_datetime_utc(value: Any) -> Optional[dt.datetime]:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except ValueError:
        return None


def _format_venue(venue: Dict[str, Any]) -> str:
    name = str(venue.get("name") or "").strip()
    city = str(venue.get("city") or "").strip()
    if name and city:
        return f"{name}, {city}"
    return name or city


def _team_name(team: Dict[str, Any]) -> str:
    return str(team.get("name") or team.get("code") or "").strip() or "Team"


def _normalize_team_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def normalize_sportmonks_fixture(
    raw: Dict[str, Any],
    client: Optional[SportmonksClient] = None,
) -> Optional[Dict[str, Any]]:
    start_time_utc = _parse_datetime_utc(raw.get("starting_at"))
    if start_time_utc is None:
        return None

    localteam = _unwrap_obj(raw.get("localteam"))
    visitorteam = _unwrap_obj(raw.get("visitorteam"))
    team_a = _team_name(localteam)
    team_b = _team_name(visitorteam)

    venue_payload = _unwrap_obj(raw.get("venue"))
    if client is not None and not venue_payload.get("name"):
        venue_payload = resolve_venue_payload(client, raw)
    venue = _format_venue(venue_payload)

    fixture_id = raw.get("id")
    if fixture_id is None:
        return None

    match_type = str(raw.get("type") or raw.get("round") or "").strip()
    status = str(raw.get("status") or "").strip()

    time_hint = f" {start_time_utc.strftime('%H:%M')} UTC"
    label_parts = [f"{team_a} vs {team_b}"]
    if match_type:
        label_parts.append(match_type)
    if venue:
        label_parts.append(venue)
    label = " | ".join(label_parts) + time_hint

    return {
        "fixture_id": f"sportmonks:{fixture_id}",
        "label": label,
        "team_a": team_a,
        "team_b": team_b,
        "venue": venue,
        "match_date": start_time_utc.date().isoformat(),
        "start_time_utc": start_time_utc.isoformat(),
        "match_type": match_type,
        "status": status,
        "source": "sportmonks",
    }


def fetch_sportmonks_fixtures_for_window(
    api_key: str,
    window_start: dt.date,
    window_end: dt.date,
) -> Tuple[List[Dict[str, Any]], int]:
    """Return normalized fixtures in the date window and an API request count."""
    client = SportmonksClient(api_token=api_key)
    requests_used = 0

    try:
        raw_fixtures = client.get_fixtures_between(
            window_start.isoformat(),
            window_end.isoformat(),
        )
        requests_used += 1
    except SportmonksError:
        raw_fixtures = []

    seen_ids = {fixture.get("id") for fixture in raw_fixtures}
    try:
        for live_fixture in client.get_livescores():
            start = str(live_fixture.get("starting_at") or "")
            if not start:
                continue
            try:
                live_date = dt.date.fromisoformat(start[:10])
            except ValueError:
                continue
            if window_start <= live_date <= window_end and live_fixture.get("id") not in seen_ids:
                raw_fixtures.append(live_fixture)
                seen_ids.add(live_fixture.get("id"))
        requests_used += 1
    except SportmonksError:
        pass

    fixtures: List[Dict[str, Any]] = []
    for raw in raw_fixtures:
        normalized = normalize_sportmonks_fixture(raw, client=client)
        if normalized is None:
            continue
        match_date = dt.date.fromisoformat(normalized["match_date"])
        if match_date < window_start or match_date > window_end:
            continue
        fixtures.append(normalized)

    fixtures.sort(
        key=lambda item: (
            item.get("match_date") or "",
            item.get("start_time_utc") or "",
        )
    )
    return fixtures, requests_used


def fixture_dedupe_key(fixture: Dict[str, Any]) -> Tuple[str, str, str]:
    team_a, team_b = sorted(
        (_normalize_team_key(fixture.get("team_a") or ""), _normalize_team_key(fixture.get("team_b") or ""))
    )
    return team_a, team_b, str(fixture.get("match_date") or "")
