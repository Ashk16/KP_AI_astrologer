"""Fetch cricket fixtures from CricketData.org / CricAPI."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

CRICAPI_BASE_URL = "https://api.cricapi.com/v1"


@dataclass
class Fixture:
    fixture_id: str
    label: str
    team_a: str
    team_b: str
    venue: str
    match_date: dt.date
    start_time_utc: Optional[dt.datetime]
    match_type: str = ""
    status: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "label": self.label,
            "team_a": self.team_a,
            "team_b": self.team_b,
            "venue": self.venue,
            "match_date": self.match_date.isoformat(),
            "start_time_utc": self.start_time_utc.isoformat() if self.start_time_utc else None,
            "match_type": self.match_type,
            "status": self.status,
        }


def _parse_date(value: Any) -> Optional[dt.date]:
    if not value:
        return None
    text = str(value)
    try:
        if "T" in text:
            return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        return dt.date.fromisoformat(text[:10])
    except ValueError:
        return None


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


def _extract_teams(match: Dict[str, Any]) -> tuple[str, str]:
    teams = match.get("teams")
    if isinstance(teams, list) and len(teams) >= 2:
        return str(teams[0]).strip(), str(teams[1]).strip()

    team_a = match.get("team-1") or match.get("team1")
    team_b = match.get("team-2") or match.get("team2")
    if team_a and team_b:
        return str(team_a).strip(), str(team_b).strip()

    name = str(match.get("name") or "").strip()
    if " vs " in name:
        left, right = name.split(" vs ", 1)
        return left.strip(), right.split(",", 1)[0].strip()

    return "Team A", "Team B"


def _fixture_id(match: Dict[str, Any]) -> str:
    for key in ("id", "unique_id", "matchId"):
        if match.get(key) is not None:
            return str(match[key])
    name = str(match.get("name") or "match")
    date_value = match.get("dateTimeGMT") or match.get("date") or ""
    return f"{name}|{date_value}"


def _normalize_match(match: Dict[str, Any]) -> Optional[Fixture]:
    team_a, team_b = _extract_teams(match)
    match_date = _parse_date(match.get("dateTimeGMT") or match.get("date"))
    if match_date is None:
        return None

    venue = str(match.get("venue") or match.get("location") or "").strip()
    start_time_utc = _parse_datetime_utc(match.get("dateTimeGMT"))
    fixture_id = _fixture_id(match)
    match_type = str(match.get("matchType") or match.get("type") or "").strip()
    status = str(match.get("status") or "").strip()

    time_hint = ""
    if start_time_utc is not None:
        time_hint = f" {start_time_utc.strftime('%H:%M')} UTC"

    label_parts = [f"{team_a} vs {team_b}"]
    if match_type:
        label_parts.append(match_type)
    if venue:
        label_parts.append(venue)
    label = " | ".join(label_parts) + time_hint

    return Fixture(
        fixture_id=fixture_id,
        label=label,
        team_a=team_a,
        team_b=team_b,
        venue=venue,
        match_date=match_date,
        start_time_utc=start_time_utc,
        match_type=match_type,
        status=status,
    )


def _request_matches(endpoint: str, api_key: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    query = {"apikey": api_key}
    if params:
        query.update(params)

    response = requests.get(f"{CRICAPI_BASE_URL}/{endpoint}", params=query, timeout=20)
    response.raise_for_status()
    payload = response.json()

    if payload.get("status") == "failure":
        reason = payload.get("reason") or payload.get("message") or "Unknown CricAPI error"
        raise ValueError(reason)

    data = payload.get("data")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("matches", "matchList", "fixture", "fixtures"):
            matches = data.get(key)
            if isinstance(matches, list):
                return matches

    matches = payload.get("matches")
    if isinstance(matches, list):
        return matches

    return []


def fetch_fixtures_for_date(api_key: str, target_date: dt.date) -> List[Fixture]:
    """Fetch fixtures for a specific date from current and upcoming match endpoints."""
    raw_matches: List[Dict[str, Any]] = []
    seen_ids = set()

    for endpoint, params in (
        ("currentMatches", None),
        ("matches", {"offset": 0}),
        ("matches", {"offset": 25}),
    ):
        try:
            matches = _request_matches(endpoint, api_key, params)
        except (requests.RequestException, ValueError):
            continue

        for match in matches:
            fixture_id = _fixture_id(match)
            if fixture_id in seen_ids:
                continue
            seen_ids.add(fixture_id)
            raw_matches.append(match)

    fixtures: List[Fixture] = []
    for match in raw_matches:
        fixture = _normalize_match(match)
        if fixture is None or fixture.match_date != target_date:
            continue
        fixtures.append(fixture)

    fixtures.sort(key=lambda item: item.start_time_utc or dt.datetime.combine(item.match_date, dt.time.min, tzinfo=dt.timezone.utc))
    return fixtures
