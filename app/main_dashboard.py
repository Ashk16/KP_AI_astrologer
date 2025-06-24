import streamlit as st
import pandas as pd
import datetime
import pytz
import sys
import os
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

# --- Path Correction ---
# Add the root directory of the project to the Python path
# This allows us to import modules from 'kp_core'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Actual KP Core Imports ---
import swisseph as swe
from kp_core.kp_engine import KPEngine
from kp_core.timeline_generator import TimelineGenerator
from kp_core.analysis_engine import AnalysisEngine

# Placeholder for future imports from kp_core
# from kp_core.kp_engine import ...
# from kp_core.timeline_generator import ...
# from kp_core.analysis_engine import ...

@st.cache_data
def get_lat_lon(location_str):
    """Gets latitude and longitude from a location string using geopy."""
    if not location_str:
        return None, None
    try:
        geolocator = Nominatim(user_agent="kp_ai_astrologer")
        location = geolocator.geocode(location_str)
        if location:
            return location.latitude, location.longitude
    except (GeocoderTimedOut, GeocoderUnavailable):
        st.warning("Geocoder service is unavailable. Please enter coordinates manually.")
    except Exception as e:
        st.error(f"An error occurred during geocoding: {e}")
    return None, None

def run_analysis(utc_dt, lat, lon, duration, team_a, team_b):
    """
    Orchestrates the calls to the KP-core modules and returns the results.
    This function no longer uses Streamlit commands directly.
    """
    try:
        # 1. Muhurta Chart
        engine = KPEngine(dt=utc_dt, lat=lat, lon=lon)
        planets_df = engine.get_all_planets_df().drop(columns=['sign_lord']).rename(columns={'nl': 'NL', 'sl': 'SL', 'ssl': 'SSL'})
        
        analysis_engine = AnalysisEngine(engine, team_a, team_b)
        muhurta_analysis = analysis_engine.analyze_muhurta_chart()

        # 2. Ascendant Timeline
        asc_gen = TimelineGenerator(utc_dt, lat, lon, swe.ASC, duration)
        asc_timeline_df = asc_gen.generate_timeline()
        analyzed_asc_df = analysis_engine.analyze_timeline(asc_timeline_df)
        analyzed_asc_df['Start Time'] = analyzed_asc_df['Start Time'].apply(lambda x: x.strftime('%H:%M:%S'))
        analyzed_asc_df['End Time'] = analyzed_asc_df['End Time'].apply(lambda x: x.strftime('%H:%M:%S'))
        
        # 3. Moon Timeline
        moon_gen = TimelineGenerator(utc_dt, lat, lon, swe.MOON, duration)
        moon_timeline_df = moon_gen.generate_timeline()
        analyzed_moon_df = analysis_engine.analyze_timeline(moon_timeline_df)
        analyzed_moon_df['Start Time'] = analyzed_moon_df['Start Time'].apply(lambda x: x.strftime('%H:%M:%S'))
        analyzed_moon_df['End Time'] = analyzed_moon_df['End Time'].apply(lambda x: x.strftime('%H:%M:%S'))
        
        return {
            "error": None,
            "muhurta_analysis": muhurta_analysis,
            "planets_df": planets_df,
            "asc_timeline_df": analyzed_asc_df,
            "moon_timeline_df": analyzed_moon_df,
            "team_a": team_a,
            "team_b": team_b
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

    # Initialize session state to hold results
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    # --- Input Sidebar ---
    with st.sidebar:
        st.header("Match Details")

        team_a = st.text_input("Team A (Ascendant)", "Team A")
        team_b = st.text_input("Team B (Descendant)", "Team B")

        # --- Location Input ---
        location_query = st.text_input("Enter Location (e.g., 'Mumbai, India')", "Wankhede Stadium, Mumbai")
        
        # Initialize lat/lon
        lat_val, lon_val = 19.0760, 72.8777 # Default to Mumbai

        if location_query:
            lat_from_geo, lon_from_geo = get_lat_lon(location_query)
            if lat_from_geo is not None:
                lat_val = lat_from_geo
                lon_val = lon_from_geo

        match_date = st.date_input("Date of Match", datetime.date.today())
        match_time = st.time_input("Time of Match", datetime.time(20, 0))
        
        # Timezone selection
        timezones = pytz.all_timezones
        default_tz_index = timezones.index('Asia/Kolkata') if 'Asia/Kolkata' in timezones else 0
        tz_name = st.selectbox("Timezone", timezones, index=default_tz_index)

        lat = st.number_input("Latitude", value=lat_val, format="%.4f")
        lon = st.number_input("Longitude", value=lon_val, format="%.4f")
        
        match_duration = st.number_input("Match Duration (hours)", min_value=1.0, max_value=8.0, value=3.5, step=0.5)

        if st.button("Generate Predictions"):
            # Combine date and time
            local_datetime = datetime.datetime.combine(match_date, match_time)
            # Localize the datetime
            local_tz = pytz.timezone(tz_name)
            localized_dt = local_tz.localize(local_datetime)
            # Convert to UTC
            utc_dt = localized_dt.astimezone(pytz.utc)

            with st.spinner("Generating astrological analysis..."):
                st.session_state.analysis_results = run_analysis(utc_dt, lat, lon, match_duration, team_a, team_b)

    # --- Display Area (checks session state) ---
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        if results.get("error"):
            st.error(f"An error occurred during analysis: {results['error']}")
            st.error("Please ensure the Swiss Ephemeris path (SWEP_PATH) is set correctly as an environment variable.")
            st.code(results['traceback'])
        else:
            team_a = results['team_a']
            team_b = results['team_b']
            
            # 1. Muhurta Chart Analysis
            st.header(f"Muhurta Chart Analysis: {team_a} vs {team_b}")
            st.write(results['muhurta_analysis'])
            st.table(results['planets_df'].style.format({'longitude': "{:.2f}"}))

            # 2. Ascendant CSSL Timeline
            st.header(f"Ascendant ({team_a}) CSSL Timeline")
            st.table(results['asc_timeline_df'][['Start Time', 'End Time', 'NL', 'SL', 'SSL', 'Verdict', 'Comment']])
            
            # 3. Moon SSL Timeline
            st.header(f"Moon SSL Timeline")
            st.table(results['moon_timeline_df'][['Start Time', 'End Time', 'NL', 'SL', 'SSL', 'Verdict', 'Comment']])


if __name__ == "__main__":
    main() 