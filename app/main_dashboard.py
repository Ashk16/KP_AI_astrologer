import streamlit as st
import pandas as pd
import datetime
import pytz
import sys
import os
import json
from glob import glob
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np

# --- Path Correction ---
# Streamlit Cloud runs this file directly; ensure both repo root and app/ are importable.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
for _path in (_PROJECT_ROOT, _APP_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from app.api_config import get_cricapi_key as load_cricapi_key, get_sportmonks_key as load_sportmonks_key
    from app.fixtures_cache import cache_status, ensure_cache, get_cached_fixtures_for_date
    from app.venue_coords import (
        list_ground_options,
        resolve_venue_coordinates,
        search_ground_options,
    )
except ImportError:
    from api_config import get_cricapi_key as load_cricapi_key, get_sportmonks_key as load_sportmonks_key
    from fixtures_cache import cache_status, ensure_cache, get_cached_fixtures_for_date
    from venue_coords import (
        list_ground_options,
        resolve_venue_coordinates,
        search_ground_options,
    )

# --- Actual KP Core Imports ---
import swisseph as swe
from kp_core.kp_engine import KPEngine, PlanetNameUtils
from kp_core.timeline_generator import TimelineGenerator
from kp_core.analysis_engine import AnalysisEngine

def _styler_element_map(styler, func, **kwargs):
    """Compatibility wrapper: uses Styler.map (pandas>=2.1) or Styler.applymap (older)."""
    if hasattr(styler, 'map'):
        return styler.map(func, **kwargs)
    return styler.applymap(func, **kwargs)



# --- Constants ---
ARCHIVE_DIR = "match_archive"
MANUAL_FIXTURE_OPTION = "__manual__"
CUSTOM_VENUE_KEY = "__custom_venue__"


def get_sportmonks_key():
    """Read Sportmonks API key from Streamlit secrets, local files, or environment."""
    try:
        key = st.secrets["SPORTMONKS_API_KEY"]
        if key:
            return key
    except (KeyError, TypeError, AttributeError):
        pass
    return load_sportmonks_key()


def get_cricapi_key():
    """Read CricAPI key from Streamlit secrets, local files, or environment."""
    try:
        key = st.secrets["CRICAPI_KEY"]
        if key:
            return key
    except Exception:
        pass
    return load_cricapi_key()


def _init_match_form_state():
    defaults = {
        "team_a": "Team A",
        "team_b": "Team B",
        "location_query": "Wankhede Stadium, Mumbai",
        "time_str": "20:00:00",
        "lat": 19.0760,
        "lon": 72.8777,
        "selected_fixture_id": MANUAL_FIXTURE_OPTION,
        "last_autofill_fixture_id": None,
        "last_geocoded_location": None,
        "match_duration": 9.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _apply_fixture_autofill(fixture: dict, tz_name: str):
    st.session_state.team_a = fixture.get("team_a") or st.session_state.team_a
    st.session_state.team_b = fixture.get("team_b") or st.session_state.team_b

    venue = fixture.get("venue") or st.session_state.location_query
    st.session_state.location_query = venue
    _resolve_and_apply_location(venue)

    start_time_utc = fixture.get("start_time_utc")
    if start_time_utc:
        utc_dt = datetime.datetime.fromisoformat(start_time_utc)
        if utc_dt.tzinfo is None:
            utc_dt = utc_dt.replace(tzinfo=pytz.utc)
        local_dt = utc_dt.astimezone(pytz.timezone(tz_name))
        st.session_state.time_str = local_dt.strftime("%H:%M:%S")

    st.session_state.last_autofill_fixture_id = fixture.get("fixture_id")


def _load_sidebar_fixtures(cricapi_key, sportmonks_key, match_date):
    """Load fixture dropdown data from cache."""
    fixture_options = {MANUAL_FIXTURE_OPTION: "Manual entry"}
    fixtures_by_id = {}
    cache = None
    cache_caption = None
    no_fixtures_caption = None
    api_warning = None

    if not cricapi_key and not sportmonks_key:
        api_warning = (
            "Add CRICAPI_KEY and/or SPORTMONKS_API_KEY to `.streamlit/secrets.toml` "
            "(local) or host secrets (deployed). See `docs/API_KEY_SETUP.md`."
        )
        return fixture_options, fixtures_by_id, cache, cache_caption, no_fixtures_caption, api_warning

    try:
        cache = ensure_cache(cricapi_key=cricapi_key, sportmonks_key=sportmonks_key)
        fixtures = get_cached_fixtures_for_date(
            cricapi_key=cricapi_key,
            sportmonks_key=sportmonks_key,
            target_date=match_date,
        )
        for fixture in fixtures:
            fixture_options[fixture["fixture_id"]] = fixture["label"]
            fixtures_by_id[fixture["fixture_id"]] = fixture

        status = cache_status(cache)
        if status["fixture_count"]:
            source_label = "cloud bundle" if status.get("cache_source") == "bundled" else "live cache"
            provider_label = ", ".join(status.get("providers") or ["cricapi"])
            refresh_note = (
                f"Next API refresh after {status['next_refresh_utc']}."
                if status.get("last_refresh_utc")
                else "Will refresh from API when needed."
            )
            cache_caption = (
                f"Fixture cache ({source_label}, {provider_label}): {status['fixture_count']} matches "
                f"({status['window_start']} to {status['window_end']}). {refresh_note}"
            )
        if len(fixtures) == 0:
            no_fixtures_caption = "No fixtures found for this date in the local cache."
    except Exception as exc:
        api_warning = f"Could not load fixtures from cache/API: {exc}"

    return fixture_options, fixtures_by_id, cache, cache_caption, no_fixtures_caption, api_warning


def _resolve_match_date():
    match_date_val = st.session_state.get("match_date", datetime.date.today())
    if isinstance(match_date_val, str):
        return datetime.date.fromisoformat(match_date_val)
    return match_date_val


def apply_team_name_replacements(text, asc_team_name, desc_team_name):
    """
    Intelligently replaces generic team references with actual team names while preserving technical terms.
    
    Args:
        text: Text to process
        asc_team_name: Name of ascendant team
        desc_team_name: Name of descendant team
        
    Returns:
        str: Text with team names replaced
    """
    if not text or not asc_team_name or not desc_team_name:
        return text
    
    # Convert to string if not already
    text = str(text)
    
    # Define all replacement patterns - order matters for specificity
    replacements = [
        # Header replacements (most specific first)
        (f"🏏 **Asc** (Ascendant) vs **Desc** (Descendant)", f"🏏 **{asc_team_name}** (Ascendant) vs **{desc_team_name}** (Descendant)"),
        (f"🏏 Asc (Ascendant) vs Desc (Descendant)", f"🏏 {asc_team_name} (Ascendant) vs {desc_team_name} (Descendant)"),
        
        # Verdict patterns with various formatting
        ("Strong Advantage Asc", f"Strong Advantage {asc_team_name}"),
        ("Strong Advantage Desc", f"Strong Advantage {desc_team_name}"),
        ("Advantage Asc", f"Advantage {asc_team_name}"),
        ("Advantage Desc", f"Advantage {desc_team_name}"),
        ("Balanced (Slight Asc)", f"Balanced (Slight {asc_team_name})"),
        ("Balanced (Slight Desc)", f"Balanced (Slight {desc_team_name})"),
        ("Favor Asc", f"Favor {asc_team_name}"),
        ("Favor Desc", f"Favor {desc_team_name}"),
        
        # Analysis verdicts with markdown formatting
        ("✅ **Strong Favor Asc**", f"✅ **Strong Favor {asc_team_name}**"),
        ("✅ **Strong Favor Desc**", f"✅ **Strong Favor {desc_team_name}**"),
        ("✅ **Favors Asc**", f"✅ **Favors {asc_team_name}**"),
        ("✅ **Favors Desc**", f"✅ **Favors {desc_team_name}**"),
        ("✅ **Supports Asc**", f"✅ **Supports {asc_team_name}**"),
        ("✅ **Supports Desc**", f"✅ **Supports {desc_team_name}**"),
        ("✅ **Strongly Supports Asc**", f"✅ **Strongly Supports {asc_team_name}**"),
        ("✅ **Strongly Supports Desc**", f"✅ **Strongly Supports {desc_team_name}**"),
        
        ("❌ **Strong Favor Asc**", f"❌ **Strong Favor {asc_team_name}**"),
        ("❌ **Strong Favor Desc**", f"❌ **Strong Favor {desc_team_name}**"),
        ("❌ **Favors Asc**", f"❌ **Favors {asc_team_name}**"),
        ("❌ **Favors Desc**", f"❌ **Favors {desc_team_name}**"),
        ("❌ **Opposes Asc**", f"❌ **Opposes {asc_team_name}**"),
        ("❌ **Opposes Desc**", f"❌ **Opposes {desc_team_name}**"),
        ("❌ **Strongly Opposes Asc**", f"❌ **Strongly Opposes {asc_team_name}**"),
        ("❌ **Strongly Opposes Desc**", f"❌ **Strongly Opposes {desc_team_name}**"),
        
        # Victory and confirmation patterns
        ("✅ **Strong Victory Asc**", f"✅ **Strong Victory {asc_team_name}**"),
        ("✅ **Strong Victory Desc**", f"✅ **Strong Victory {desc_team_name}**"),
        ("✅ **Victory Asc**", f"✅ **Victory {asc_team_name}**"),
        ("✅ **Victory Desc**", f"✅ **Victory {desc_team_name}**"),
        ("✅ **Final Confirmation Asc**", f"✅ **Final Confirmation {asc_team_name}**"),
        ("✅ **Final Confirmation Desc**", f"✅ **Final Confirmation {desc_team_name}**"),
        ("✅ **Confirms Asc**", f"✅ **Confirms {asc_team_name}**"),
        ("✅ **Confirms Desc**", f"✅ **Confirms {desc_team_name}**"),
        ("✅ **Strongly Confirms Asc**", f"✅ **Strongly Confirms {asc_team_name}**"),
        ("✅ **Strongly Confirms Desc**", f"✅ **Strongly Confirms {desc_team_name}**"),
        
        ("❌ **Strong Victory Asc**", f"❌ **Strong Victory {asc_team_name}**"),
        ("❌ **Strong Victory Desc**", f"❌ **Strong Victory {desc_team_name}**"),
        ("❌ **Victory Asc**", f"❌ **Victory {asc_team_name}**"),
        ("❌ **Victory Desc**", f"❌ **Victory {desc_team_name}**"),
        ("❌ **Final Confirmation Asc**", f"❌ **Final Confirmation {asc_team_name}**"),
        ("❌ **Final Confirmation Desc**", f"❌ **Final Confirmation {desc_team_name}**"),
        ("❌ **Denies Asc**", f"❌ **Denies {asc_team_name}**"),
        ("❌ **Denies Desc**", f"❌ **Denies {desc_team_name}**"),
        ("❌ **Strongly Denies Asc**", f"❌ **Strongly Denies {asc_team_name}**"),
        ("❌ **Strongly Denies Desc**", f"❌ **Strongly Denies {desc_team_name}**"),
        
        # Summary sentence patterns
        ("indicates a general advantage for Asc", f"indicates a general advantage for {asc_team_name}"),
        ("indicates a general advantage for Desc", f"indicates a general advantage for {desc_team_name}"),
        ("This indicates a general advantage for Asc", f"This indicates a general advantage for {asc_team_name}"),
        ("This indicates a general advantage for Desc", f"This indicates a general advantage for {desc_team_name}"),
        ("Support: Asc", f"Support: {asc_team_name}"),
        ("Support: Desc", f"Support: {desc_team_name}"),
        ("✅ **Support Asc**", f"✅ **Support {asc_team_name}**"),
        ("✅ **Support Desc**", f"✅ **Support {desc_team_name}**"),
        ("❌ **Support Asc**", f"❌ **Support {asc_team_name}**"),
        ("❌ **Support Desc**", f"❌ **Support {desc_team_name}**"),
        
        # Win probability patterns
        ("Win Probability: Asc", f"Win Probability: {asc_team_name}"),
        ("Win Probability: Desc", f"Win Probability: {desc_team_name}"),
        ("Weighted Score: Asc", f"Weighted Score: {asc_team_name}"),
        ("Weighted Score: Desc", f"Weighted Score: {desc_team_name}"),
    ]
    
    # Apply all replacements
    for old_pattern, new_pattern in replacements:
        text = text.replace(old_pattern, new_pattern)
    
    return text

def apply_team_replacements_to_results(results, asc_team_name, desc_team_name):
    """
    Apply team name replacements to all relevant parts of the results dictionary.
    
    Args:
        results: Analysis results dictionary
        asc_team_name: Name of ascendant team  
        desc_team_name: Name of descendant team
        
    Returns:
        dict: Updated results with team names replaced
    """
    if not asc_team_name or not desc_team_name:
        return results
    
    # Create a copy to avoid modifying the original
    updated_results = results.copy()
    
    # Replace in muhurta analysis text (skip if it's a dictionary structure)
    if 'muhurta_analysis' in updated_results:
        muhurta_data = updated_results['muhurta_analysis']
        # Only apply replacements if it's a string, not a dictionary
        if isinstance(muhurta_data, str):
            updated_results['muhurta_analysis'] = apply_team_name_replacements(
                muhurta_data, asc_team_name, desc_team_name
            )
        elif isinstance(muhurta_data, dict) and 'final_verdict' in muhurta_data:
            # Apply team name replacements to the verdict text if present
            if 'verdict' in muhurta_data['final_verdict']:
                muhurta_data['final_verdict']['verdict'] = apply_team_name_replacements(
                    muhurta_data['final_verdict']['verdict'], asc_team_name, desc_team_name
                )
    
    # Replace in timeline analyses (Only Moon timeline retained)
    for timeline_key in ['moon_timeline_analysis']:
        if timeline_key in updated_results and 'summary' in updated_results[timeline_key]:
            updated_results[timeline_key]['summary'] = apply_team_name_replacements(
                updated_results[timeline_key]['summary'], asc_team_name, desc_team_name
            )
    
    # Replace in timeline DataFrames - Verdict and Comment columns (Only Moon timeline retained)
    for df_key in ['moon_timeline_df']:
        if df_key in updated_results:
            df = updated_results[df_key].copy()
            
            if 'Verdict' in df.columns:
                df['Verdict'] = df['Verdict'].apply(
                    lambda x: apply_team_name_replacements(x, asc_team_name, desc_team_name)
                )
            
            if 'Comment' in df.columns:
                df['Comment'] = df['Comment'].apply(
                    lambda x: apply_team_name_replacements(x, asc_team_name, desc_team_name)
                )
            
            updated_results[df_key] = df
    
    # Store team mapping in results for future reference
    updated_results['team_mapping'] = {
        'ascendant_team': asc_team_name,
        'descendant_team': desc_team_name
    }
    
    return updated_results

def color_planets(val):
    """
    Applies red/green color gradient based on score.
    - Positive scores: Green with varying intensity
    - Negative scores: Red with varying intensity
    """
    if pd.isna(val):
        return ''
    
    # Define base colors
    GREEN = "rgba(0, 128, 0, {opacity})"  # rgb(0, 128, 0) is pure green
    RED = "rgba(255, 0, 0, {opacity})"    # rgb(255, 0, 0) is pure red
    
    # Calculate opacity based on absolute value
    # Scores beyond ±2.0 will get maximum opacity of 1.0
    opacity = min(abs(val) / 2.0, 1.0)
    # Ensure minimum opacity of 0.1 for visibility
    opacity = max(opacity, 0.1)
    
    # Apply color based on score sign
    if val > 0:
        return f'background-color: {GREEN.format(opacity=opacity)}'
    else:
        return f'background-color: {RED.format(opacity=opacity)}'

def color_timeline_planets_by_score(planet_short_name, planet_scores):
    """
    Applies the exact same color logic as planets table based on actual scores.
    """
    if pd.isna(planet_short_name):
        return ''
    
    # Remove retrograde indicator if present for score lookup
    lookup_name = planet_short_name
    if planet_short_name.startswith('(R)'):
        lookup_name = planet_short_name[3:]  # Remove "(R)" prefix
    
    if lookup_name not in planet_scores:
        return ''
        
    score = planet_scores[lookup_name]
    return color_planets(score)

def color_verdict_cell(verdict_text, team_a_name="Team A", team_b_name="Team B"):
    """
    Applies color coding to verdict cells with normal colors and transparency levels:
    
    - Strong Advantage: Normal red/green (no transparency)
    - Advantage: 50% transparency  
    - Balanced: 80% transparency (color based on score sign)
    
    Args:
        verdict_text: The verdict text to color
        team_a_name: Name of Team A (Ascendant)
        team_b_name: Name of Team B (Descendant)
    """
    if pd.isna(verdict_text) or not verdict_text:
        return ''
    
    verdict_lower = verdict_text.lower()
    team_a_lower = team_a_name.lower()
    team_b_lower = team_b_name.lower()
    
    # === STRONG ADVANTAGE (Normal colors, no transparency) ===
    if "strong advantage" in verdict_lower:
        if team_a_lower in verdict_lower:
            return 'background-color: #008000; color: white; font-weight: bold'  # Normal green
        elif team_b_lower in verdict_lower:
            return 'background-color: #ff0000; color: white; font-weight: bold'  # Normal red
    
    # === ADVANTAGE (50% transparency) ===
    elif "advantage" in verdict_lower and "strong" not in verdict_lower:
        if team_a_lower in verdict_lower:
            return 'background-color: rgba(0, 128, 0, 0.5); color: #004000; font-weight: 500'  # Green 50% transparency
        elif team_b_lower in verdict_lower:
            return 'background-color: rgba(255, 0, 0, 0.5); color: #800000; font-weight: 500'  # Red 50% transparency
    
    # === BALANCED PERIODS (80% transparency, color based on sign) ===
    elif "balanced" in verdict_lower:
        if team_a_lower in verdict_lower:
            return 'background-color: rgba(0, 128, 0, 0.2); color: #006000; font-weight: 400'  # Green 80% transparency
        elif team_b_lower in verdict_lower:
            return 'background-color: rgba(255, 0, 0, 0.2); color: #800000; font-weight: 400'  # Red 80% transparency
        else:
            # Pure balanced period
            return 'background-color: #e0e0e0; color: #333; font-weight: 400'  # Light gray for neutral
    
    # === SPECIAL CASES ===
    # Challenging periods (Orange)
    elif 'challenging period' in verdict_lower:
        return 'background-color: #ff7043; color: white; font-weight: 400'  # Orange for challenges
    
    # === FALLBACK PATTERNS ===
    # Catch any remaining team-specific patterns with lightest shades (80% transparency)
    elif team_a_lower in verdict_lower:
        return 'background-color: rgba(0, 128, 0, 0.2); color: #006000; font-weight: 400'  # Green 80% transparency
    elif team_b_lower in verdict_lower:
        return 'background-color: rgba(255, 0, 0, 0.2); color: #800000; font-weight: 400'  # Red 80% transparency
    
    return ''  # No styling for unrecognized patterns

def _apply_selected_ground(ground: dict) -> None:
    st.session_state.location_query = ground["name"]
    st.session_state.lat = ground["lat"]
    st.session_state.lon = ground["lon"]
    st.session_state.last_geocoded_location = ground["name"]


def _resolve_and_apply_location(location_str: str) -> str:
    """Resolve lat/lon using the ground library first, then geocoding."""
    location_str = (location_str or "").strip()
    if not location_str:
        return "none"

    coord = resolve_venue_coordinates(location_str)
    if coord.resolved:
        st.session_state.lat = coord.lat
        st.session_state.lon = coord.lon
        st.session_state.last_geocoded_location = location_str
        return "library"

    lat_from_geo, lon_from_geo = get_lat_lon(location_str)
    if lat_from_geo is not None:
        st.session_state.lat = lat_from_geo
        st.session_state.lon = lon_from_geo
        st.session_state.last_geocoded_location = location_str
        return "geocoder"

    return "none"


def _render_manual_venue_inputs() -> None:
    """Venue picker for manual entry: library suggestions first, geocoder as fallback."""
    grounds = list_ground_options()
    ground_by_key = {ground["key"]: ground for ground in grounds}
    picker_options = [CUSTOM_VENUE_KEY] + [ground["key"] for ground in grounds]

    def _format_venue_option(key: str) -> str:
        if key == CUSTOM_VENUE_KEY:
            return "Enter custom location (not in list)"
        return ground_by_key[key]["label"]

    picked = st.selectbox(
        "Venue",
        options=picker_options,
        format_func=_format_venue_option,
        key="venue_picker",
        help="Type to search known cricket grounds. Listed venues use coordinates from the ground library.",
    )

    if picked != CUSTOM_VENUE_KEY:
        ground = ground_by_key[picked]
        _apply_selected_ground(ground)
        st.caption(
            f"Coordinates from ground library: {ground['lat']:.4f}, {ground['lon']:.4f}"
        )
        return

    location_query = st.text_input(
        "Enter Location (e.g., 'Wankhede Stadium, Mumbai')",
        key="location_query",
    )
    query = (location_query or "").strip()

    if query and query != st.session_state.get("last_geocoded_location"):
        coord = resolve_venue_coordinates(query)
        if coord.resolved:
            st.session_state.lat = coord.lat
            st.session_state.lon = coord.lon
            st.session_state.last_geocoded_location = query
            st.caption(f"Matched ground library entry: {coord.matched_name}")
        else:
            suggestions = search_ground_options(query, limit=8)
            if suggestions:
                st.caption("Pick a suggested ground below, or keep typing to refine your search.")
            else:
                source = _resolve_and_apply_location(query)
                if source == "geocoder":
                    st.caption("Coordinates from geocoder (venue not found in ground library).")
    elif query:
        coord = resolve_venue_coordinates(query)
        if coord.resolved:
            st.caption(f"Using ground library: {coord.matched_name}")
        elif st.session_state.get("last_geocoded_location") == query:
            st.caption("Using geocoded coordinates.")

    if len(query) >= 2:
        suggestions = search_ground_options(query, limit=8)
        if suggestions:
            suggestion_labels = {item["key"]: item["label"] for item in suggestions}

            def _apply_suggestion() -> None:
                selected_key = st.session_state.get("venue_suggestion_pick")
                if not selected_key or selected_key not in ground_by_key:
                    return
                _apply_selected_ground(ground_by_key[selected_key])
                st.session_state.venue_picker = selected_key

            st.selectbox(
                "Suggestions from ground library",
                options=[""] + list(suggestion_labels.keys()),
                format_func=lambda key: "Select a suggested ground..." if not key else suggestion_labels[key],
                key="venue_suggestion_pick",
                on_change=_apply_suggestion,
            )


@st.cache_data
def get_lat_lon(location_str):
    """Gets latitude and longitude from a location string using multiple geocoding services."""
    if not location_str:
        return None, None
    
    # Try geocode.xyz first (free service that often works on cloud platforms)
    try:
        import requests
        import time
        time.sleep(1)  # Be respectful
        
        url = f"https://geocode.xyz/{location_str}?json=1"
        headers = {'User-Agent': 'kp_ai_astrologer_streamlit_app'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'latt' in data and 'longt' in data:
                try:
                    lat = float(data['latt'])
                    lon = float(data['longt'])
                    if lat != 0 and lon != 0:  # Valid coordinates
                        return lat, lon
                except (ValueError, TypeError):
                    pass
    except Exception:
        pass  # Fall through to next service
    
    # Fallback to Nominatim with enhanced headers and rate limiting
    try:
        import time
        time.sleep(1.5)  # Respect rate limits
        geolocator = Nominatim(
            user_agent="kp_ai_astrologer_streamlit_v1.0",
            timeout=10
        )
        location = geolocator.geocode(location_str)
        if location:
            return location.latitude, location.longitude
    except Exception:
        pass
        
    st.warning("Geocoder service is unavailable. Please enter coordinates manually.")
    return None, None

def save_analysis(results):
    """Saves the complete analysis results to a JSON file using the standardized structure."""
    match_details = results['match_details']
    team_a = match_details['team_a'].replace(" ", "-")
    team_b = match_details['team_b'].replace(" ", "-")
    date_str = match_details['datetime_utc'].date().strftime('%Y-%m-%d')
    
    filename = f"{date_str}_{team_a}_vs_{team_b}.json"
    
    # Ensure the archive directory exists
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    
    filepath = os.path.join(ARCHIVE_DIR, filename)

    # Create a deep copy to modify for serialization
    data_to_save = {k: v for k, v in results.items() if not isinstance(v, pd.DataFrame)}
    data_to_save['match_details'] = results['match_details'].copy()

    # Convert dataframes and datetime to JSON serializable formats
    data_to_save['planets_df'] = results['planets_df'].to_json(orient='split')
    data_to_save['cusps_df'] = results['cusps_df'].to_json(orient='split')
    data_to_save['moon_timeline_df'] = results['moon_timeline_df'].to_json(orient='split')
    data_to_save['match_details']['datetime_utc'] = match_details['datetime_utc'].isoformat()
    
    # Preserve team mapping if present
    if 'team_mapping' in results:
        data_to_save['team_mapping'] = results['team_mapping']

    with open(filepath, 'w') as f:
        json.dump(data_to_save, f, indent=4)
    st.success(f"Analysis saved to {filename}")

def load_analysis(filename):
    """
    Loads an analysis file and restores data structures.
    Includes backward compatibility for old, flawed file structures.
    """
    filepath = os.path.join(ARCHIVE_DIR, filename)
    with open(filepath, 'r') as f:
        loaded_data = json.load(f)

    # --- Backward Compatibility Check ---
    # If 'match_details' is missing, it's an old file. Rebuild the structure.
    if 'match_details' not in loaded_data:
        # Reconstruct match_details from old flat keys
        match_details = {
            'team_a': loaded_data.get('team_a', 'Unknown'),
            'team_b': loaded_data.get('team_b', 'Unknown'),
            'lat': loaded_data.get('inputs', {}).get('lat', 0),
            'lon': loaded_data.get('inputs', {}).get('lon', 0),
            'duration_hours': loaded_data.get('inputs', {}).get('duration_hours', 0),
            'datetime_utc': loaded_data.get('inputs', {}).get('datetime_utc_str', '1970-01-01T00:00:00')
        }
        loaded_data['match_details'] = match_details
    
    # Restore dataframes from JSON
    loaded_data['planets_df'] = pd.read_json(loaded_data['planets_df'], orient='split')
    loaded_data['cusps_df'] = pd.read_json(loaded_data['cusps_df'], orient='split') if 'cusps_df' in loaded_data else KPEngine(match_details['datetime_utc'], match_details['lat'], match_details['lon']).get_all_cusps_df()
    loaded_data['moon_timeline_df'] = pd.read_json(loaded_data['moon_timeline_df'], orient='split')
    
    # Restore datetime
    loaded_data['match_details']['datetime_utc'] = datetime.datetime.fromisoformat(
        loaded_data['match_details']['datetime_utc']
    )
    
    # Restore team mapping if present (for newer saved files)
    if 'team_mapping' not in loaded_data:
        loaded_data['team_mapping'] = {}
    
    return loaded_data

def get_saved_matches():
    """Returns a sorted list of saved match files."""
    # Ensure the archive directory exists before trying to list files
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    files = glob(os.path.join(ARCHIVE_DIR, "*.json"))
    return sorted(files, reverse=True)

def get_ist_time(dt_utc):
    """Convert UTC datetime to IST"""
    return pd.to_datetime(dt_utc).tz_convert('Asia/Kolkata')

# Function removed as magnitude categorization is no longer used

def clean_comment_for_display(comment):
    """
    Clean comment by removing score and technical information.
    Keeps only the descriptive planetary and cricket context parts.
    
    Args:
        comment: Original comment string
        
    Returns:
        str: Cleaned comment
    """
    if pd.isna(comment):
        return ""
    
    # Split comment by separator
    parts = comment.split(" | ")
    
    # Filter out parts containing technical indicators
    filtered_parts = []
    for part in parts:
        # Skip parts that contain technical indicators
        skip_indicators = [
            "score:", "nl:", "sl:", "combined:", 
            "📊", "high", "medium", "low"
        ]
        
        # Keep the part if it doesn't contain any skip indicators
        if not any(indicator in part.lower() for indicator in skip_indicators):
            filtered_parts.append(part)
    
    return " | ".join(filtered_parts)

def prepare_timeline_for_display(timeline_df):
    """
    Prepare timeline DataFrame for user-friendly display by adding formatted columns.
    
    Args:
        timeline_df: Original timeline DataFrame
        
    Returns:
        DataFrame: Enhanced timeline with additional display columns
    """
    if timeline_df.empty:
        return timeline_df
    
    df_display = timeline_df.copy()
    
    # Add formatted Score column if it exists
    if 'Score' in df_display.columns:
        df_display['Score_Display'] = df_display['Score'].apply(lambda x: f"{x:+.3f}" if pd.notna(x) else "")
    else:
        df_display['Score_Display'] = ""
    
    # Clean comments
    if 'Comment' in df_display.columns:
        df_display['Comment_Clean'] = df_display['Comment'].apply(clean_comment_for_display)
    else:
        df_display['Comment_Clean'] = ""
    
    return df_display

def display_analysis(results):
    """Display the analysis results with enhanced formatting."""
    display_results = results
    
    # Display astrological settings information
    match_details = display_results.get('match_details', {})
    ayanamsa = match_details.get('ayanamsa', 'KRISHNAMURTI')
    
    # Create ayanamsa info box
    st.info(f"🔮 **Astrological Settings**: Using {ayanamsa} Ayanamsa for sidereal calculations (authentic KP astrology)")
    
    # Check if there are any errors first
    if results.get("error"):
        st.error("An error occurred during analysis:")
        st.exception(results.get("traceback", "No traceback available."))
        st.code(results.get("traceback", "No traceback available."))
        return # Stop execution if there was an error

    # Team Name Replacement UI
    st.subheader("🏏 Team Assignment")
    st.markdown("*Assign actual team names to Ascendant and Descendant positions based on match observation*")
    
    # Get existing team mapping if available
    existing_mapping = results.get('team_mapping', {})
    default_asc = existing_mapping.get('ascendant_team', '')
    default_desc = existing_mapping.get('descendant_team', '')
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        asc_team_input = st.text_input(
            "Ascendant Team",
            value=default_asc,
            placeholder="e.g., Mumbai Indians",
            help="Team performing like Ascendant position",
            key=f"asc_team_{id(results)}"
        )
    
    with col2:
        desc_team_input = st.text_input(
            "Descendant Team", 
            value=default_desc,
            placeholder="e.g., Chennai Super Kings",
            help="Team performing like Descendant position",
            key=f"desc_team_{id(results)}"
        )
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        apply_teams = st.button(
            "Apply Team Names",
            help="Replace generic 'Asc/Desc' with actual team names",
            key=f"apply_teams_{id(results)}"
        )
    
    # Apply team name replacements if requested
    if apply_teams and asc_team_input.strip() and desc_team_input.strip():
        display_results = apply_team_replacements_to_results(results, asc_team_input.strip(), desc_team_input.strip())
        st.success(f"✅ Analysis updated with team names: **{asc_team_input}** (Ascendant) vs **{desc_team_input}** (Descendant)")
    elif apply_teams:
        st.warning("⚠️ Please enter both team names to apply replacements")
    
    # Button to save the current analysis (with team names if applied)
    if st.button("Save Current Analysis", key=f"save_{id(results)}"):
         save_analysis(display_results)
         
    # Authentic KP Muhurta Chart Analysis
    st.header("Authentic KP Muhurta Chart Analysis")
    
    # Updated Muhurta Display
    if "muhurta_analysis" in display_results:
        cached_analysis = display_results["muhurta_analysis"]
        
        # Handle both dictionary and string formats
        if isinstance(cached_analysis, dict):
            st.markdown(f"**Method**: {cached_analysis.get('method', 'Unknown')}")
            st.markdown(f"**Timestamp**: {cached_analysis.get('timestamp', 'Unknown')}")
        else:
            # If it's a string (legacy format), display it directly
            st.markdown("**Analysis Results:**")
            st.markdown(str(cached_analysis))
        
        # House Groups (only for dictionary format)
        if isinstance(cached_analysis, dict):
            house_groups = cached_analysis.get('house_groups', {})
        else:
            house_groups = {}
        if house_groups:
            st.subheader("House Classification")
            col1, col2 = st.columns(2)
            col1.write("Strong Houses: " + str(house_groups['strong_houses']))
            col2.write("Weak Houses: " + str(house_groups['weak_houses']))
        
        # Primary Promise Test - Tabular (only for dictionary format)
        if isinstance(cached_analysis, dict):
            promise_test = cached_analysis.get('promise_test', {})
        else:
            promise_test = {}
        if promise_test:
            st.subheader("Primary Promise Test")
            analysis_data = []
            analysis = promise_test.get('analysis', {})
            for cusp_key, data in analysis.items():
                strength = data['strength_data']
                analysis_data.append({
                    'Cusp': cusp_key,
                    'Sub Lord': data['sub_lord'],
                    'Score': data['score'],
                    'Classification': strength['classification'],
                    'Net Strength': strength['net_strength'],
                    'Highest House': strength.get('highest_house', 'N/A'),
                    'Highest Rule': strength.get('highest_rule', 'N/A')
                })
            
            if analysis_data:
                df = pd.DataFrame(analysis_data)
                st.dataframe(df, use_container_width=True)
            
            total_score = promise_test.get('total_score', 0)
            st.metric("Total Promise Score", f"{total_score:.2f}")
        
        # Ruling Planets Verification - Tabular (only for dictionary format)
        if isinstance(cached_analysis, dict):
            rp_verification = cached_analysis.get('ruling_planets', {})
        else:
            rp_verification = {}
        if rp_verification:
            st.subheader("Ruling Planets Verification")
            rp_data = []
            for rp_type, data in rp_verification.items():
                rp_data.append({
                    'Type': rp_type.capitalize(),
                    'Planet': data['planet'],
                    'Strength': data['strength']
                })
            if rp_data:
                rp_df = pd.DataFrame(rp_data)
                st.dataframe(rp_df, use_container_width=True)
        
        # Additional Metrics: Win Percentages and Contest Type
        if promise_test:
            # Normalize score to percentage (assuming score range -5 to 5 for example)
            asc_pct = max(0, min(100, (total_score + 5) / 10 * 100))
            desc_pct = 100 - asc_pct
            
            # Contest Type
            abs_score = abs(total_score)
            if abs_score > 3:
                contest_type = "One-sided"
            elif abs_score > 1:
                contest_type = "Likely Decisive"
            else:
                contest_type = "Close Contest"
            
            st.subheader("Prediction Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Asc Win %", f"{asc_pct:.1f}%")
            col2.metric("Desc Win %", f"{desc_pct:.1f}%")
            col3.metric("Contest Type", contest_type)
        
        # Final Verdict (only for dictionary format)
        if isinstance(cached_analysis, dict):
            final_verdict = cached_analysis.get('final_verdict', {})
        else:
            final_verdict = {}
        if final_verdict:
            st.subheader("Final Verdict")
            st.success(final_verdict['verdict'])
            st.metric("11th Cusp Strength", f"{final_verdict['eleventh_cusp_strength']:.2f}")
    
    # --- New: Cusp Details Table (moved outside muhurta conditional) ---
    st.subheader("Cusp Details (All 12 Houses)")
    cusps_df = display_results.get('cusps_df')
    if cusps_df is not None and not cusps_df.empty:
        # Build a compact display table
        disp = cusps_df.copy()
        disp = disp[[
            'longitude', 'sign', 'sign_lord', 'nl', 'sl', 'ssl'
        ]].rename(columns={
            'longitude': 'Longitude',
            'sign': 'Sign',
            'sign_lord': 'Sign Lord',
            'nl': 'NL',
            'sl': 'SL',
            'ssl': 'SSL'
        })
        
        # Format longitude to show degrees and minutes
        disp['Longitude'] = disp['Longitude'].apply(lambda x: f"{int(x)}°{int((x % 1) * 60):02d}'" if pd.notna(x) else '')

        # Compute SL significators for each cusp's sub lord
        try:
            match_details = display_results['match_details']
            ayanamsa = match_details.get('ayanamsa', 'KRISHNAMURTI')
            engine = KPEngine(match_details['datetime_utc'], match_details['lat'], match_details['lon'], ayanamsa=ayanamsa)
            # Initialize timeline_weights if missing
            if 'timeline_weights' not in st.session_state:
                st.session_state.timeline_weights = {'ssl_timeline': {'NL': 0.1, 'SL': 0.2, 'SSL': 0.7}}
            ae_for_sig = AnalysisEngine(engine, match_details['team_a'], match_details['team_b'],
                                        house_weights=st.session_state.house_weights, timeline_weights=st.session_state.timeline_weights)
            sl_sigs = []
            for cusp_num in disp.index:
                sl_short = disp.loc[cusp_num, 'SL']
                sl_full = PlanetNameUtils.to_full_name(sl_short)
                try:
                    sigs = ae_for_sig.get_significators(sl_full)
                    # Group by rule for readable display
                    groups = {1: [], 2: [], 3: [], 4: []}
                    for h, r in sigs:
                        groups[r].append(h)
                    parts = []
                    if groups[1]: parts.append(f"R1:{','.join(str(h) for h in sorted(set(groups[1]))) }")
                    if groups[2]: parts.append(f"R2:{','.join(str(h) for h in sorted(set(groups[2]))) }")
                    if groups[3]: parts.append(f"R3:{','.join(str(h) for h in sorted(set(groups[3]))) }")
                    if groups[4]: parts.append(f"R4:{','.join(str(h) for h in sorted(set(groups[4]))) }")
                    sl_sigs.append(' | '.join(parts) if parts else '')
                except Exception:
                    sl_sigs.append('')
            disp['SL Significators'] = sl_sigs
        except Exception:
            disp['SL Significators'] = ''

        disp.index.name = 'Cusp'
        st.dataframe(disp, use_container_width=True)
    else:
        st.info("No cusp data available")

    # --- Duplicate muhurta analysis section removed ---
    # The old simplified muhurta analysis display has been removed
    # to avoid conflicts with the new authentic KP analysis above
    # --- End of removed duplicate section ---

    st.subheader("Planetary Positions & Scores")
    planets_df = display_results["planets_df"].copy()
    
    # Create planet scores mapping BEFORE modifying index for consistent coloring
    planet_scores = {}
    for planet in planets_df.index:
        planet_short = PlanetNameUtils.to_short_name(planet)
        planet_scores[planet_short] = planets_df.loc[planet, 'Score']
    
    # Add retrograde indicators to planet names in the index
    if 'is_retrograde' in planets_df.columns:
        new_index = []
        for planet in planets_df.index:
            is_retrograde = planets_df.loc[planet, 'is_retrograde']
            display_name = PlanetNameUtils.standardize_for_display(planet, is_retrograde)
            if is_retrograde and planet not in ['Rahu', 'Ketu']:  # Don't add (R) to Rahu/Ketu as they're always retrograde
                new_index.append(f"(R) {planet}")
            else:
                new_index.append(planet)
        planets_df.index = new_index
    
    # Specify column order to have Comment at the very end (remove is_retrograde from display)
    base_columns = [col for col in planets_df.columns if col not in ['Score', 'Significators', 'Comment', 'is_retrograde']]
    column_order = base_columns + ['Score', 'Significators', 'Comment']
    
    # 1. Reorder the DataFrame columns first
    reordered_df = planets_df.reindex(columns=column_order)

    # 2. Apply styling and formatting to both Planet name and Score columns
    styler = reordered_df.style.apply(lambda x: x.map(color_planets), subset=['Score'])
    
    # Apply the same color to the index (Planet names) based on their scores
    planet_colors = {planet: color_planets(score) for planet, score in reordered_df['Score'].items()}
    styler = styler.apply(lambda x: pd.Series([planet_colors.get(idx, '') for idx in x.index], index=x.index), axis=0)
    
    st.dataframe(styler.format({'Score': '{:.2f}'}), use_container_width=True)

    # Get team names for verdict coloring (use replaced names if available)
    team_mapping = display_results.get('team_mapping', {})
    team_a_name = team_mapping.get('ascendant_team', 'Asc')
    team_b_name = team_mapping.get('descendant_team', 'Desc')

    # Add retrograde indicators to planet names for display
    def add_retrograde_to_planet_name(planet_short_name):
        if pd.isna(planet_short_name):
            return planet_short_name
        # Convert short name to full name for lookup
        planet_full = PlanetNameUtils.to_full_name(planet_short_name)
        if planet_full in planets_df.index and 'is_retrograde' in planets_df.columns:
            is_retrograde = planets_df.loc[planet_full, 'is_retrograde']
            return PlanetNameUtils.standardize_for_display(planet_short_name, is_retrograde)
        return planet_short_name

    # === ASCENDANT TIMELINE SECTION (Collapsed by default) ===
    with st.expander("🔮 Ascendant Timeline (NL & SL Analysis)", expanded=False):
        st.markdown('<p class="timeline-description">Ascendant cusp timeline showing Nakshatra Lord (NL) and Sub Lord (SL) transitions. Ascendant represents the match start conditions and overall momentum.</p>', unsafe_allow_html=True)
        
        if "asc_timeline_df" in display_results and display_results["asc_timeline_df"] is not None:
            asc_timeline_df = display_results["asc_timeline_df"].copy()
            
            # Convert times to IST for display
            asc_timeline_df['Start Time'] = pd.to_datetime(asc_timeline_df['Start Time']).dt.tz_convert('Asia/Kolkata').dt.strftime('%H:%M:%S')
            asc_timeline_df['End Time'] = pd.to_datetime(asc_timeline_df['End Time']).dt.tz_convert('Asia/Kolkata').dt.strftime('%H:%M:%S')
            
            # Apply retrograde indicators to planet columns
            if 'NL_Planet' in asc_timeline_df.columns:
                asc_timeline_df['NL_Planet'] = asc_timeline_df['NL_Planet'].apply(add_retrograde_to_planet_name)
            if 'SL_Planet' in asc_timeline_df.columns:
                asc_timeline_df['SL_Planet'] = asc_timeline_df['SL_Planet'].apply(add_retrograde_to_planet_name)
            
            # Prepare display DataFrame with only NL and SL
            display_cols = ['Start Time', 'End Time', 'NL_Planet', 'SL_Planet', 'Score', 'Verdict', 'Comment']
            asc_display_df = asc_timeline_df[display_cols].copy()
            
            # Format Score column
            if 'Score' in asc_display_df.columns:
                asc_display_df['Score'] = asc_display_df['Score'].apply(lambda x: f"{x:+.3f}" if pd.notna(x) else "")
            
            # Apply team name replacements to Verdict and Comment
            asc_display_df['Verdict'] = asc_display_df['Verdict'].apply(lambda x: apply_team_name_replacements(x, team_a_name, team_b_name))
            asc_display_df['Comment'] = asc_display_df['Comment'].apply(lambda x: apply_team_name_replacements(x, team_a_name, team_b_name))
            
            # Apply color styling to planet columns and verdict
            styler_asc = _styler_element_map(
                _styler_element_map(
                    asc_display_df.style,
                    lambda x: color_timeline_planets_by_score(x, planet_scores),
                    subset=['NL_Planet', 'SL_Planet']
                ),
                lambda x: color_verdict_cell(x, team_a_name, team_b_name),
                subset=['Verdict']
            )
            
            st.dataframe(styler_asc, use_container_width=True, height=400)
            
            # Display summary
            if "asc_timeline_analysis" in display_results:
                st.write(display_results["asc_timeline_analysis"]["summary"])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Favorable Planets:**")
                    st.json(display_results['asc_timeline_analysis']['favorable_planets'])
                with col2:
                    st.write("**Unfavorable Planets:**")
                    st.json(display_results['asc_timeline_analysis']['unfavorable_planets'])
        else:
            st.info("Ascendant timeline data not available")

    st.subheader("Moon SSL Timeline - Enhanced Dynamic Full Granular Detail")
    st.markdown('<p class="timeline-description">Enhanced detailed timeline with dynamic layer analysis showing all Sub-Sub Lord periods. Includes planetary influence percentages for precise timing analysis.</p>', unsafe_allow_html=True)
    moon_timeline_df = display_results["moon_timeline_df"].copy()

    # Convert times to IST for display using the correct pandas method
    moon_timeline_df['Start Time'] = pd.to_datetime(moon_timeline_df['Start Time']).dt.tz_convert('Asia/Kolkata').dt.strftime('%H:%M:%S')
    moon_timeline_df['End Time'] = pd.to_datetime(moon_timeline_df['End Time']).dt.tz_convert('Asia/Kolkata').dt.strftime('%H:%M:%S')
    
    # Apply retrograde indicators to planet columns for Moon timeline
    if 'NL_Planet' in moon_timeline_df.columns:
        moon_timeline_df['NL_Planet'] = moon_timeline_df['NL_Planet'].apply(add_retrograde_to_planet_name)
    if 'SL_Planet' in moon_timeline_df.columns:
        moon_timeline_df['SL_Planet'] = moon_timeline_df['SL_Planet'].apply(add_retrograde_to_planet_name)
    if 'SSL_Planet' in moon_timeline_df.columns:
        moon_timeline_df['SSL_Planet'] = moon_timeline_df['SSL_Planet'].apply(add_retrograde_to_planet_name)

    # Display analysis method if available
    if 'method' in display_results["moon_timeline_analysis"]:
        method = display_results['moon_timeline_analysis'].get('method', 'standard')
        st.metric("Analysis Method", method.replace('_', ' ').title())

    # Create a view for display with enhanced user-friendly columns
    if 'NL_Influence' in moon_timeline_df.columns:
        # Prepare enhanced timeline with additional display columns
        moon_prepared_df = prepare_timeline_for_display(moon_timeline_df)
        
        # Define display columns in the requested order: Score, Verdict, Comment
        base_columns = ['Start Time', 'End Time', 'NL_Planet', 'SL_Planet', 'SSL_Planet']
        enhanced_columns = ['Score_Display', 'Verdict', 'Comment_Clean']
        
        display_columns = base_columns + enhanced_columns
        moon_display_df = moon_prepared_df[display_columns].copy()
        
        # Rename columns for better display
        moon_display_df = moon_display_df.rename(columns={
            'Score_Display': 'Score',
            'Comment_Clean': 'Comment'
        })
        
        # Add option to view comments separately for better readability
        if st.checkbox("📝 Show Detailed Comments Separately", key=f"moon_comments_separate_{id(results)}_{hash(str(moon_timeline_df.columns))}"):
            # Display table without comments
            moon_no_comments = moon_display_df.drop(columns=['Comment'])
            styler_moon_no_comments = _styler_element_map(
                _styler_element_map(
                    moon_no_comments.style,
                    lambda x: color_timeline_planets_by_score(x, planet_scores),
                    subset=['NL_Planet', 'SL_Planet', 'SSL_Planet']
                ),
                lambda x: color_verdict_cell(x, team_a_name, team_b_name),
                subset=['Verdict']
            )
            st.dataframe(styler_moon_no_comments, use_container_width=True, height=400)
            
            # Display comments in expandable sections
            with st.expander("📝 Detailed Period Comments", expanded=True):
                for idx, row in moon_display_df.iterrows():
                    st.markdown(f"**{row['Start Time']} - {row['End Time']}** ({row['NL_Planet']}-{row['SL_Planet']}-{row['SSL_Planet']}) - *{row['Verdict']}*")
                    st.markdown(f"<div style='padding-left: 20px; color: #444; line-height: 1.5;'>{row['Comment']}</div>", unsafe_allow_html=True)
                    st.markdown("---")
        else:
            # Display full table with comments - already prepared above
            pass
    else:
        # Fallback for older timeline format
        moon_display_df = moon_timeline_df.drop(columns=['Score'] if 'Score' in moon_timeline_df.columns else [])
    
    # Apply coloring to planet columns and verdict column
    # Only show the main dataframe if comments are not shown separately
    if 'NL_Influence' not in moon_timeline_df.columns or not st.session_state.get(f"moon_comments_separate_{id(results)}_{hash(str(moon_timeline_df.columns))}", False):
        styler_moon = _styler_element_map(
            _styler_element_map(
                moon_display_df.style,
                lambda x: color_timeline_planets_by_score(x, planet_scores),
                subset=['NL_Planet', 'SL_Planet', 'SSL_Planet']
            ),
            lambda x: color_verdict_cell(x, team_a_name, team_b_name),
            subset=['Verdict']
        )
        st.dataframe(styler_moon, use_container_width=True, height=400)
    st.write(display_results["moon_timeline_analysis"]["summary"])
    
    st.subheader("Favorable Planets")
    st.json(display_results['moon_timeline_analysis']['favorable_planets'])
        
    st.subheader("Unfavorable Planets")
    st.json(display_results['moon_timeline_analysis']['unfavorable_planets'])

    # Display Detailed SSSL Timeline if available
    if 'detailed_timelines' in display_results and display_results['detailed_timelines']:
        st.markdown("---")
        st.subheader("Moon SSSL Timeline - Full Detailed Analysis (NL-SL-SSL-SSSL)")
        st.markdown('<p class="timeline-description">Enhanced timeline with SSSL (Sub-Sub-Sub Lord) level for finest granularity. Shows all four KP levels: NL, SL, SSL, and SSSL.</p>', unsafe_allow_html=True)
        
        detailed_timelines = display_results['detailed_timelines']
        
        # Display only the SSSL-Level Timeline (finest granularity with all 4 columns)
        if 'sssl_timeline' in detailed_timelines and not detailed_timelines['sssl_timeline'].empty:
            sssl_df = detailed_timelines['sssl_timeline'].copy()
            sssl_df['Start Time'] = pd.to_datetime(sssl_df['Start Time']).dt.tz_convert('Asia/Kolkata').dt.strftime('%H:%M:%S')
            sssl_df['End Time'] = pd.to_datetime(sssl_df['End Time']).dt.tz_convert('Asia/Kolkata').dt.strftime('%H:%M:%S')
            
            # Apply retrograde indicators
            for col in ['NL_Planet', 'SL_Planet', 'SSL_Planet', 'SSSL_Planet']:
                if col in sssl_df.columns:
                    sssl_df[col] = sssl_df[col].apply(add_retrograde_to_planet_name)
            
            # Apply team name replacements
            sssl_df['Verdict'] = sssl_df['Verdict'].apply(lambda x: apply_team_name_replacements(x, team_a_name, team_b_name))
            sssl_df['Comment'] = sssl_df['Comment'].apply(lambda x: apply_team_name_replacements(x, team_a_name, team_b_name))
            
            # Apply color coding - same as the existing Moon SSL timeline
            display_cols = ['Start Time', 'End Time', 'NL_Planet', 'SL_Planet', 'SSL_Planet', 'SSSL_Planet', 'Score', 'Verdict', 'Comment']
            sssl_display_df = sssl_df[display_cols]
            
            # Apply styling with color coding for planet columns and verdict
            styler_sssl = _styler_element_map(
                _styler_element_map(
                    sssl_display_df.style,
                    lambda x: color_timeline_planets_by_score(x, planet_scores),
                    subset=['NL_Planet', 'SL_Planet', 'SSL_Planet', 'SSSL_Planet']
                ),
                lambda x: color_verdict_cell(x, team_a_name, team_b_name),
                subset=['Verdict']
            )
            
            st.dataframe(styler_sssl, use_container_width=True, height=400)
        else:
            st.info("No SSSL-level timeline data available")

def run_analysis(match_details, timeline_weights=None, house_weights=None):
    """
    Orchestrates the KP core analysis and returns a single, consistent dictionary.
    Accepts timeline_weights and house_weights for dynamic user control.
    """
    try:
        ayanamsa = match_details.get('ayanamsa', 'KRISHNAMURTI')
        engine = KPEngine(match_details['datetime_utc'], match_details['lat'], match_details['lon'], ayanamsa=ayanamsa)
        # Pass house_weights and timeline_weights to AnalysisEngine
        analysis_engine = AnalysisEngine(engine, match_details['team_a'], match_details['team_b'],
                                         house_weights=house_weights, timeline_weights=timeline_weights)
        muhurta_analysis = analysis_engine.analyze_muhurta_chart(scoring_method='authentic_kp')
        planets_df = analysis_engine.get_all_planet_details_df()
        cusps_df = engine.get_all_cusps_df()
        # Generate Ascendant timeline (aggregated - NL and SL only)
        asc_timeline_gen = TimelineGenerator(engine, 'Ascendant')
        asc_timeline_df = asc_timeline_gen.generate_aggregated_timeline_df(match_details['datetime_utc'], match_details['duration_hours'])
        asc_timeline_df, asc_timeline_analysis = analysis_engine.analyze_aggregated_timeline(asc_timeline_df, 'ascendant')
        
        # Generate Moon timeline
        moon_timeline_gen = TimelineGenerator(engine, 'Moon')
        moon_timeline_df = moon_timeline_gen.generate_timeline_df(match_details['datetime_utc'], match_details['duration_hours'])
        moon_timeline_df, moon_timeline_analysis = analysis_engine.analyze_timeline(moon_timeline_df, 'ascendant')
        
        # Generate detailed timelines at NL, SL, SSL, and SSSL levels
        detailed_timelines = moon_timeline_gen.generate_detailed_timeline_df(match_details['datetime_utc'], match_details['duration_hours'])
        analyzed_detailed_timelines = analysis_engine.analyze_detailed_timeline(detailed_timelines, 'ascendant')
        
        return {
            "muhurta_analysis": muhurta_analysis,
            "planets_df": planets_df,
            "cusps_df": cusps_df,
            "asc_timeline_df": asc_timeline_df,
            "asc_timeline_analysis": asc_timeline_analysis,
            "moon_timeline_df": moon_timeline_df,
            "moon_timeline_analysis": moon_timeline_analysis,
            "detailed_timelines": analyzed_detailed_timelines,
            "match_details": match_details,
            "error": None,
            "traceback": None
        }
    except Exception as e:
        import traceback
        return {
            "error": e,
            "traceback": traceback.format_exc()
        }

def main():
    st.set_page_config(page_title="KP AI Astrologer", layout="wide")
    st.title("KP AI Astrologer: Cricket Match Predictor")
    
    # Custom CSS for better dataframe display
    st.markdown("""
    <style>
    /* Simple and clean approach for dataframe display */
    .stDataFrame div[data-testid="stDataFrame"] {
        width: 100% !important;
    }
    
    /* Style for better text wrapping in cells */
    .stDataFrame div[data-testid="stDataFrame"] table {
        font-size: 12px;
        width: 100% !important;
        table-layout: fixed !important;
    }
    
    /* Timeline table specific styling */
    .stDataFrame div[data-testid="stDataFrame"] td {
        white-space: normal !important;
        word-wrap: break-word !important;
        padding: 8px !important;
        vertical-align: top !important;
        max-width: none !important;
        overflow-wrap: break-word !important;
    }
    
    /* Header styling */
    .stDataFrame div[data-testid="stDataFrame"] th {
        background-color: #f0f2f6 !important;
        font-weight: bold !important;
        text-align: center !important;
        padding: 10px 8px !important;
        border-bottom: 2px solid #ddd !important;
        white-space: normal !important;
    }
    
    /* Planet columns styling */
    .stDataFrame div[data-testid="stDataFrame"] td:nth-child(3),
    .stDataFrame div[data-testid="stDataFrame"] td:nth-child(4),
    .stDataFrame div[data-testid="stDataFrame"] td:nth-child(5) {
        text-align: center !important;
        font-weight: bold !important;
        width: 80px !important;
        min-width: 80px !important;
    }
    
    /* Time columns styling */
    .stDataFrame div[data-testid="stDataFrame"] td:nth-child(1),
    .stDataFrame div[data-testid="stDataFrame"] td:nth-child(2) {
        text-align: center !important;
        font-family: monospace !important;
        width: 90px !important;
        min-width: 90px !important;
    }
    
    /* Verdict column styling */
    .stDataFrame div[data-testid="stDataFrame"] td:nth-last-child(2) {
        text-align: center !important;
        font-weight: bold !important;
        white-space: normal !important;
        width: 150px !important;
        min-width: 150px !important;
    }
    
    /* Comment column styling - Make it flexible and visible */
    .stDataFrame div[data-testid="stDataFrame"] td:nth-last-child(1) {
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        text-align: left !important;
        padding: 8px 12px !important;
        line-height: 1.4 !important;
        min-width: 400px !important;
        max-width: 600px !important;
        width: auto !important;
    }
    
    /* Timeline section styling */
    .timeline-description {
        font-style: italic;
        color: #666;
        margin-bottom: 10px;
    }
    
    /* Ensure horizontal scrolling works */
    .stDataFrame div[data-testid="stDataFrame"] > div {
        overflow-x: auto !important;
        min-width: 100% !important;
    }
    
    /* Comment header styling */
    .stDataFrame div[data-testid="stDataFrame"] th:nth-last-child(1) {
        min-width: 400px !important;
        width: auto !important;
    }

    /* Highlight Generate Predictions button in sidebar */
    section[data-testid="stSidebar"] button[kind="primary"] {
        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%) !important;
        border: 1px solid #0d9488 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(15, 118, 110, 0.35) !important;
    }

    section[data-testid="stSidebar"] button[kind="primary"]:hover {
        background: linear-gradient(135deg, #115e59 0%, #0f766e 100%) !important;
        border-color: #0f766e !important;
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state to hold multiple analyses in tabs
    if 'analyses' not in st.session_state:
        st.session_state.analyses = []
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0

    # --- Input Sidebar ---
    with st.sidebar:
        st.header("Match Details")

        st.subheader("New Analysis")
        _init_match_form_state()

        cricapi_key = get_cricapi_key()
        sportmonks_key = get_sportmonks_key()
        pending_match_date = _resolve_match_date()
        fixture_options, fixtures_by_id, _, _, _, early_api_warning = _load_sidebar_fixtures(
            cricapi_key, sportmonks_key, pending_match_date
        )

        if st.session_state.selected_fixture_id not in fixture_options:
            st.session_state.selected_fixture_id = MANUAL_FIXTURE_OPTION

        pending_fixture_id = st.session_state.get("selected_fixture_id", MANUAL_FIXTURE_OPTION)
        if (
            pending_fixture_id != MANUAL_FIXTURE_OPTION
            and pending_fixture_id in fixtures_by_id
            and pending_fixture_id != st.session_state.get("last_autofill_fixture_id")
        ):
            _apply_fixture_autofill(
                fixtures_by_id[pending_fixture_id],
                st.session_state.get("tz_name", "Asia/Kolkata"),
            )

        match_date = st.date_input("Date of Match", datetime.date.today(), key="match_date")

        time_str = st.text_input(
            "Time of Match (HH:MM:SS)",
            key="time_str",
            help="Enter time in HH:MM:SS format for maximum accuracy",
        )

        if st.button("Generate Predictions", type="primary", use_container_width=True, key="generate_predictions_btn"):
            try:
                try:
                    match_time = datetime.datetime.strptime(st.session_state.time_str, "%H:%M:%S").time()
                except ValueError:
                    match_time = datetime.datetime.strptime(st.session_state.time_str, "%H:%M").time()
                    match_time = match_time.replace(second=0)

                local_datetime = datetime.datetime.combine(match_date, match_time)
                local_tz = pytz.timezone(st.session_state.get("tz_name", "Asia/Kolkata"))
                localized_dt = local_tz.localize(local_datetime)
                utc_dt = localized_dt.astimezone(pytz.utc)

                team_a_value = st.session_state.team_a
                team_b_value = st.session_state.team_b

                with st.spinner("Generating astrological analysis..."):
                    new_analysis = run_analysis({
                        "team_a": team_a_value,
                        "team_b": team_b_value,
                        "datetime_utc": utc_dt,
                        "lat": st.session_state.lat,
                        "lon": st.session_state.lon,
                        "duration_hours": st.session_state.get("match_duration", 9.0),
                        "ayanamsa": st.session_state.get("ayanamsa_choice", "KRISHNAMURTI"),
                    },
                    timeline_weights=st.session_state.timeline_weights,
                    house_weights=st.session_state.house_weights)

                    tab_name = f"{team_a_value} vs {team_b_value} - {match_date.strftime('%Y-%m-%d')}"
                    new_analysis['tab_name'] = tab_name

                    st.session_state.analyses.append(new_analysis)
                    st.session_state.active_tab = len(st.session_state.analyses) - 1
                    st.rerun()
            except ValueError:
                st.error("Invalid time format. Please use HH:MM:SS or HH:MM format.")

        fixture_options, fixtures_by_id, _, cache_caption, no_fixtures_caption, api_warning = _load_sidebar_fixtures(
            cricapi_key, sportmonks_key, match_date
        )

        if api_warning:
            if api_warning.startswith("Add CRICAPI_KEY"):
                st.info(api_warning)
            else:
                st.warning(api_warning)

        if st.session_state.selected_fixture_id not in fixture_options:
            st.session_state.selected_fixture_id = MANUAL_FIXTURE_OPTION

        selected_fixture_id = st.selectbox(
            "Select Match",
            options=list(fixture_options.keys()),
            format_func=lambda key: fixture_options[key],
            key="selected_fixture_id",
        )

        team_a = st.text_input("Team A (Ascendant)", key="team_a")
        team_b = st.text_input("Team B (Descendant)", key="team_b")

        if selected_fixture_id == MANUAL_FIXTURE_OPTION:
            _render_manual_venue_inputs()
        else:
            st.text_input(
                "Enter Location (e.g., 'Mumbai, India')",
                key="location_query",
            )

        lat = st.number_input("Latitude", key="lat", format="%.4f")
        lon = st.number_input("Longitude", key="lon", format="%.4f")

        match_duration = st.number_input(
            "Match Duration (hours)",
            min_value=1.0,
            max_value=12.0,
            value=9.0,
            step=0.5,
            key="match_duration",
        )

        timezones = pytz.all_timezones
        default_tz_index = timezones.index('Asia/Kolkata') if 'Asia/Kolkata' in timezones else 0
        tz_name = st.selectbox("Timezone", timezones, index=default_tz_index, key="tz_name")

        if cache_caption:
            st.caption(cache_caption)
        if no_fixtures_caption:
            st.caption(no_fixtures_caption)

        st.subheader("Astrological Settings")
        ayanamsa_options = ['KRISHNAMURTI', 'LAHIRI', 'RAMAN', 'TRUE_CITRA']
        ayanamsa_choice = st.selectbox(
            "Ayanamsa (Sidereal Correction)",
            ayanamsa_options,
            index=0,
            key="ayanamsa_choice",
            help="Krishnamurti Ayanamsa is recommended for KP astrology. Lahiri is official in India.",
        )
        
        st.divider()

        # --- Load Previous Match ---
        st.subheader("Load Analysis")
        saved_matches = get_saved_matches()
        # The selectbox is searchable by default
        match_to_load = st.selectbox(
            "Select a saved match", 
            options=[""] + [os.path.basename(f) for f in saved_matches],
            index=0,
            help="Select from the last 10 matches or start typing to search for older ones."
        )
        if st.button("Load Match") and match_to_load:
            with st.spinner("Loading analysis..."):
                loaded_analysis = load_analysis(match_to_load)
                
                # Add tab name to the loaded analysis
                match_details = loaded_analysis['match_details']
                tab_name = f"{match_details['team_a']} vs {match_details['team_b']} - {match_details['datetime_utc'].strftime('%Y-%m-%d')}"
                loaded_analysis['tab_name'] = tab_name
                
                # Add to analyses list and set as active tab
                st.session_state.analyses.append(loaded_analysis)
                st.session_state.active_tab = len(st.session_state.analyses) - 1
            st.rerun()

        # --- Timeline Layer Weights ---
        st.subheader("Timeline Layer Weights")
        st.markdown("*Adjust the influence of NL, SL, SSL/Context in timeline verdicts*")
        if 'timeline_weights' not in st.session_state:
            st.session_state.timeline_weights = {
                'ssl_timeline': {'NL': 0.1, 'SL': 0.2, 'SSL': 0.7}
            }
        ssl_nl = st.slider("SSL Timeline: NL (%)", 0, 100, int(st.session_state.timeline_weights['ssl_timeline']['NL']*100), 5)
        ssl_sl = st.slider("SSL Timeline: SL (%)", 0, 100, int(st.session_state.timeline_weights['ssl_timeline']['SL']*100), 5)
        ssl_ssl = st.slider("SSL Timeline: SSL (%)", 0, 100, int(st.session_state.timeline_weights['ssl_timeline']['SSL']*100), 5)
        total_ssl = ssl_nl + ssl_sl + ssl_ssl
        if total_ssl != 100:
            st.warning("SSL Timeline weights must sum to 100%.")
        st.session_state.timeline_weights['ssl_timeline'] = {
            'NL': ssl_nl/100, 'SL': ssl_sl/100, 'SSL': ssl_ssl/100
        }
        st.divider()

    # --- House Weights Controls ---
    st.subheader("House Weights")
    st.markdown("*Adjust the weight for each house (1-12). Changes apply in real time.*")
    if 'house_weights' not in st.session_state:
        st.session_state.house_weights = {
            1: 0.5, 2: 0.2, 3: 0.3, 4: -0.2, 5: -0.8, 6: 1.0, 7: -0.6, 8: -1.0, 9: 0.3, 10: 0.7, 11: 0.9, 12: -0.9
        }
    cols = st.columns(6)
    for i in range(1, 13):
        with cols[(i-1)%6]:
            st.session_state.house_weights[i] = st.number_input(f"House {i}", value=st.session_state.house_weights[i], key=f"house_{i}", step=0.05, format="%.2f")
    st.divider()

    # --- Display Area with Multiple Tabs ---
    if st.session_state.analyses:
        # Create tab names with close buttons
        tab_names = []
        for i, analysis in enumerate(st.session_state.analyses):
            tab_name = analysis.get('tab_name', f"Match {i+1}")
            # Truncate long tab names
            if len(tab_name) > 30:
                tab_name = tab_name[:27] + "..."
            tab_names.append(tab_name)
        
        # Create tabs
        tabs = st.tabs(tab_names)
        
        # Display each analysis in its respective tab
        for i, (tab, analysis) in enumerate(zip(tabs, st.session_state.analyses)):
            with tab:
                # Add close button at the top of each tab
                col1, col2 = st.columns([6, 1])
                with col2:
                    if st.button("❌ Close Tab", key=f"close_{i}", help="Close this tab"):
                        # Remove this analysis from the list
                        st.session_state.analyses.pop(i)
                        # Adjust active tab if necessary
                        if st.session_state.active_tab >= len(st.session_state.analyses):
                            st.session_state.active_tab = max(0, len(st.session_state.analyses) - 1)
                        st.rerun()
                
                # Display the analysis
                display_analysis(analysis)

if __name__ == "__main__":
    main() 