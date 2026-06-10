"""Resolve cricket ground coordinates from the curated ground library."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LIBRARY = _PROJECT_ROOT / "data" / "ground_coords.json"
_CRICKET_KP_LIBRARY = Path.home() / "Cricket_KP" / "data" / "ground_coords.json"


@dataclass
class CoordResult:
    lat: Optional[float]
    lon: Optional[float]
    source: str
    matched_name: Optional[str] = None

    @property
    def resolved(self) -> bool:
        return self.lat is not None and self.lon is not None


def _library_path() -> Path:
    override = os.getenv("GROUND_COORDS_PATH")
    if override:
        return Path(override)
    if _DEFAULT_LIBRARY.exists():
        return _DEFAULT_LIBRARY
    return _CRICKET_KP_LIBRARY


def _normalize(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _valid(lat: Any, lon: Any) -> bool:
    try:
        return -90.0 <= float(lat) <= 90.0 and -180.0 <= float(lon) <= 180.0
    except (TypeError, ValueError):
        return False


def load_library(path: Optional[Path] = None) -> Dict[str, Any]:
    library_path = path or _library_path()
    if not library_path.exists():
        return {"grounds": []}
    with open(library_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _lookup_by_name(grounds: List[Dict[str, Any]], name: Optional[str]) -> Optional[Dict[str, Any]]:
    key_name = _normalize(name)
    if not key_name:
        return None

    for ground in grounds:
        if not _valid(ground.get("lat"), ground.get("lon")):
            continue
        if _normalize(ground.get("name")) == key_name:
            return ground

    for ground in grounds:
        if not _valid(ground.get("lat"), ground.get("lon")):
            continue
        ground_name = _normalize(ground.get("name"))
        if key_name in ground_name or ground_name in key_name:
            return ground

    return None


def _venue_name_candidates(venue: str) -> List[str]:
    venue = venue.strip()
    if not venue:
        return []

    candidates = [venue]
    if "," in venue:
        candidates.append(venue.split(",", 1)[0].strip())
    if " - " in venue:
        candidates.append(venue.split(" - ", 1)[0].strip())

    deduped: List[str] = []
    seen = set()
    for candidate in candidates:
        normalized = _normalize(candidate)
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(candidate)
    return deduped


def resolve_venue_coordinates(venue: Optional[str]) -> CoordResult:
    """Resolve lat/lon for a CricAPI venue string using the ground library."""
    if not venue:
        return CoordResult(None, None, "none")

    library = load_library()
    grounds = library.get("grounds", [])

    for candidate in _venue_name_candidates(venue):
        hit = _lookup_by_name(grounds, candidate)
        if hit is not None:
            return CoordResult(
                float(hit["lat"]),
                float(hit["lon"]),
                "library",
                matched_name=hit.get("name"),
            )

    return CoordResult(None, None, "none")
