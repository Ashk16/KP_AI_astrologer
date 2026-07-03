"""Thin Sportmonks Cricket API client for fixture schedules."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

DEFAULT_BASE_URL = "https://cricket.sportmonks.com/api/v2.0"
FIXTURE_LIST_INCLUDES = "localteam,visitorteam,venue"


class SportmonksError(RuntimeError):
    pass


class SportmonksClient:
    def __init__(
        self,
        api_token: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 20,
    ):
        self.api_token = api_token or os.getenv("SPORTMONKS_API_KEY")
        self.base_url = (base_url or os.getenv("SPORTMONKS_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        if not self.api_token:
            raise SportmonksError(
                "No Sportmonks API token. Set SPORTMONKS_API_KEY in the environment/.env."
            )

    def _get_raw_abs(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = dict(params or {})
        payload["api_token"] = self.api_token
        resp = requests.get(url, params=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise SportmonksError(f"GET {url} -> {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def _get_pages(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        max_pages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        out: List[Dict[str, Any]] = []
        page = 1
        while True:
            page_params = dict(params or {})
            page_params["page"] = page
            payload = self._get_raw_abs(url, page_params)
            data = payload.get("data") or []
            out.extend(data)
            last_page = (payload.get("meta") or {}).get("last_page")
            page += 1
            if not last_page or page > last_page:
                break
            if max_pages is not None and page > max_pages:
                break
        return out

    def get_venue(self, venue_id: int) -> Dict[str, Any]:
        try:
            payload = self._get_raw_abs(f"{self.base_url}/venues/{venue_id}")
            return _unwrap_obj(payload.get("data", payload))
        except SportmonksError:
            return {}

    def get_livescores(self, includes: str = FIXTURE_LIST_INCLUDES) -> List[Dict[str, Any]]:
        payload = self._get_raw_abs(f"{self.base_url}/livescores", {"include": includes})
        return _unwrap_list(payload.get("data", payload))

    def get_fixtures_between(
        self,
        date_from: str,
        date_to: str,
        includes: str = FIXTURE_LIST_INCLUDES,
    ) -> List[Dict[str, Any]]:
        params = {
            "include": includes,
            "filter[starts_between]": _starts_between_filter(date_from, date_to),
        }
        return self._get_pages("fixtures", params)


def resolve_venue_payload(client: SportmonksClient, fixture: Dict[str, Any]) -> Dict[str, Any]:
    venue = _unwrap_obj(fixture.get("venue"))
    if venue.get("name"):
        return venue
    venue_id = fixture.get("venue_id") or venue.get("id")
    if venue_id is not None:
        return client.get_venue(int(venue_id))
    return venue


def _is_date_only(value: str) -> bool:
    text = value.strip()
    return len(text) == 10 and text[4] == "-" and text[7] == "-"


def _starts_between_filter(date_from: str, date_to: str) -> str:
    start = date_from.strip()
    end = date_to.strip()
    if _is_date_only(start):
        start = f"{start} 00:00:00"
    if _is_date_only(end):
        end = f"{end} 23:59:59"
    return f"{start},{end}"


def _unwrap_list(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict) and "data" in value:
        return value["data"] or []
    if isinstance(value, list):
        return value
    return []


def _unwrap_obj(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict) and "data" in value and isinstance(value["data"], dict):
        return value["data"]
    if isinstance(value, dict):
        return value
    return {}
