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

# --- Path Correction ---
# Add the root directory of the project to the Python path
# This allows us to import modules from 'kp_core'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Actual KP Core Imports ---
import swisseph as swe
from kp_core.kp_engine import KPEngine, PlanetNameUtils
from kp_core.timeline_generator import TimelineGenerator
from kp_core.analysis_engine import AnalysisEngine



# --- Constants ---
ARCHIVE_DIR = "match_archive"

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
    
    # Replace in muhurta analysis text
    if 'muhurta_analysis' in updated_results:
        updated_results['muhurta_analysis'] = apply_team_name_replacements(
            updated_results['muhurta_analysis'], asc_team_name, desc_team_name
        )
    
    # Replace in timeline analyses
    for timeline_key in ['asc_timeline_analysis', 'desc_timeline_analysis', 'moon_timeline_analysis']:
        if timeline_key in updated_results and 'summary' in updated_results[timeline_key]:
            updated_results[timeline_key]['summary'] = apply_team_name_replacements(
                updated_results[timeline_key]['summary'], asc_team_name, desc_team_name
            )
    
    # Replace in timeline DataFrames - Verdict and Comment columns
    for df_key in ['asc_timeline_df', 'desc_timeline_df', 'moon_timeline_df']:
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
    Applies subtle green/red color based on score.
    Uses a perceptually uniform colormap with adjusted ranges for better visualization.
    """
    if pd.isna(val):
        return ''
        
    # Using a perceptually uniform colormap 'PiYG' (Pinkish/Green)
    cmap = cm.get_cmap('PiYG')
    
    # Normalize the score to a reasonable range
    # Scores beyond ±2.0 will get the maximum color intensity
    norm = colors.Normalize(vmin=-2.0, vmax=2.0)
    normalized_val = norm(val)
    
    # Create more subtle colors by adjusting the color mapping
    if val < 0:
        # For negative values (red shades)
        # Map -2.0 to 0.0 to 0.35 to 0.45 (subtle reds)
        color_val = 0.45 - (abs(normalized_val) * 0.1)
    else:
        # For positive values (green shades)
        # Map 0.0 to 2.0 to 0.55 to 0.65 (subtle greens)
        color_val = 0.55 + (normalized_val * 0.1)
    
    # Ensure color_val stays within bounds
    color_val = max(0.35, min(0.65, color_val))
    
    rgba_color = cmap(color_val)
    return f'background-color: {colors.to_hex(rgba_color, keep_alpha=True)}'

def color_timeline_planets_by_score(planet_short_name, planet_scores):
    """
    Applies the exact same color logic as planets table based on actual scores.
    """
    if pd.isna(planet_short_name) or planet_short_name not in planet_scores:
        return ''
        
    score = planet_scores[planet_short_name]
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
    data_to_save['asc_timeline_df'] = results['asc_timeline_df'].to_json(orient='split')
    data_to_save['desc_timeline_df'] = results['desc_timeline_df'].to_json(orient='split')
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
    loaded_data['asc_timeline_df'] = pd.read_json(loaded_data['asc_timeline_df'], orient='split')
    loaded_data['desc_timeline_df'] = pd.read_json(loaded_data['desc_timeline_df'], orient='split')
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
    return dt_utc.astimezone(pytz.timezone('Asia/Kolkata'))

def display_analysis(results):
    """Display the analysis results for a single match."""
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
    display_results = results
    if apply_teams and asc_team_input.strip() and desc_team_input.strip():
        display_results = apply_team_replacements_to_results(results, asc_team_input.strip(), desc_team_input.strip())
        st.success(f"✅ Analysis updated with team names: **{asc_team_input}** (Ascendant) vs **{desc_team_input}** (Descendant)")
    elif apply_teams:
        st.warning("⚠️ Please enter both team names to apply replacements")
    
    # Button to save the current analysis (with team names if applied)
    if st.button("Save Current Analysis", key=f"save_{id(results)}"):
         save_analysis(display_results)
         
    # Muhurta Chart Analysis with toggle for scoring methods
    col_header, col_toggle = st.columns([3, 1])
    
    with col_header:
        st.header("Muhurta Chart Analysis")
    
    with col_toggle:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        scoring_method = st.radio(
            "Scoring Method:",
            options=["Proportional", "Binary"],
            horizontal=True,
            key=f"muhurta_method_{id(results)}",
            help="Toggle between proportional (score-based) and binary (verdict-based) synthesis methods"
        )
    
    # Display analysis based on selected method
    if "muhurta_analysis" in display_results:
        method_key = scoring_method.lower()
        
        # Check if we need to regenerate analysis with different method
        if not hasattr(st.session_state, f'muhurta_cache_{id(results)}'):
            st.session_state[f'muhurta_cache_{id(results)}'] = {}
        
        cache = st.session_state[f'muhurta_cache_{id(results)}']
        
        if method_key not in cache:
            # Generate analysis for the selected method
            with st.spinner(f"Generating {scoring_method} analysis..."):
                try:
                    # Get the analysis engine from session state or recreate
                    match_details = display_results['match_details']
                    engine = KPEngine(match_details['datetime_utc'], match_details['lat'], match_details['lon'])
                    analysis_engine = AnalysisEngine(engine, match_details['team_a'], match_details['team_b'])
                    
                    # Generate analysis with the selected method
                    muhurta_analysis = analysis_engine.analyze_muhurta_chart(scoring_method=method_key)
                    
                    # Apply team name replacements if they exist
                    team_mapping = display_results.get('team_mapping', {})
                    if team_mapping.get('ascendant_team') and team_mapping.get('descendant_team'):
                        muhurta_analysis = apply_team_name_replacements(
                            muhurta_analysis, 
                            team_mapping['ascendant_team'], 
                            team_mapping['descendant_team']
                        )
                    
                    cache[method_key] = muhurta_analysis
                    
                except Exception as e:
                    st.error(f"Error generating {scoring_method} analysis: {str(e)}")
                    cache[method_key] = display_results["muhurta_analysis"]  # Fallback to existing
        
        # Display the cached analysis
        st.write(cache[method_key])
        
        # Add a small note about the difference
        if scoring_method == "Proportional":
            st.info("💡 **Proportional Method**: Uses actual score magnitudes to weight each factor, providing nuanced probability calculations.")
        else:
            st.info("💡 **Binary Method**: Uses traditional fixed points based on verdict categories, following classical approach.")

    st.subheader("Planetary Positions & Scores")
    planets_df = display_results["planets_df"]
    
    # Specify column order to have Score at the end
    column_order = [col for col in planets_df.columns if col not in ['Score', 'Significators']] + ['Significators', 'Score']
    
    # 1. Reorder the DataFrame columns first
    reordered_df = planets_df.reindex(columns=column_order)

    # 2. Apply styling and formatting to the reordered DataFrame
    styler = reordered_df.style.apply(lambda x: x.map(color_planets), subset=['Score'])
    st.dataframe(styler.format({'Score': '{:.2f}'}))

    # Get team names for verdict coloring (use replaced names if available)
    team_mapping = display_results.get('team_mapping', {})
    team_a_name = team_mapping.get('ascendant_team', 'Asc')
    team_b_name = team_mapping.get('descendant_team', 'Desc')

    st.subheader(f"Ascendant Based Timeline (Asc) - Star Lord + Sub Lord Level")
    st.markdown('<p class="timeline-description">Aggregated timeline showing periods at Star Lord and Sub Lord level for practical match analysis. Each period represents longer, more actionable time segments.</p>', unsafe_allow_html=True)
    asc_timeline_df = display_results["asc_timeline_df"].copy() # Use a copy to avoid modifying session state
    
    # Convert times to IST for display using the correct pandas method
    asc_timeline_df['Start Time'] = pd.to_datetime(asc_timeline_df['Start Time']).dt.tz_convert('Asia/Kolkata').dt.strftime('%H:%M:%S')
    asc_timeline_df['End Time'] = pd.to_datetime(asc_timeline_df['End Time']).dt.tz_convert('Asia/Kolkata').dt.strftime('%H:%M:%S')

    # Create planet scores mapping for consistent coloring
    planet_scores = {}
    for planet in planets_df.index:
        planet_short = PlanetNameUtils.to_short_name(planet)
        planet_scores[planet_short] = planets_df.loc[planet, 'Score']
    
    # Create a view for display, dropping the score column but keeping Verdict and Comment
    asc_display_df = asc_timeline_df.drop(columns=['Score'])
    
    # Apply coloring to planet columns and verdict column  
    # Check if SSL_Planet column exists (for granular timelines) or not (for aggregated timelines)
    planet_columns = ['NL_Planet', 'SL_Planet']
    if 'SSL_Planet' in asc_display_df.columns:
        planet_columns.append('SSL_Planet')
    
    styler_asc = asc_display_df.style.applymap(
        lambda x: color_timeline_planets_by_score(x, planet_scores),
        subset=planet_columns
    ).applymap(
        lambda x: color_verdict_cell(x, team_a_name, team_b_name),
        subset=['Verdict']
    )
    st.dataframe(styler_asc, use_container_width=True, height=400)
    st.write(display_results["asc_timeline_analysis"]["summary"])

    st.subheader(f"Descendant Based Timeline (Desc) - Star Lord + Sub Lord Level")
    st.markdown('<p class="timeline-description">Aggregated timeline showing periods at Star Lord and Sub Lord level for practical match analysis. Each period represents longer, more actionable time segments.</p>', unsafe_allow_html=True)
    desc_timeline_df = display_results["desc_timeline_df"].copy() # Use a copy

    # Convert times to IST for display using the correct pandas method
    desc_timeline_df['Start Time'] = pd.to_datetime(desc_timeline_df['Start Time']).dt.tz_convert('Asia/Kolkata').dt.strftime('%H:%M:%S')
    desc_timeline_df['End Time'] = pd.to_datetime(desc_timeline_df['End Time']).dt.tz_convert('Asia/Kolkata').dt.strftime('%H:%M:%S')

    # Create a view for display, dropping the score column but keeping Verdict and Comment
    desc_display_df = desc_timeline_df.drop(columns=['Score'])
    
    # Apply coloring to planet columns and verdict column
    # Check if SSL_Planet column exists (for granular timelines) or not (for aggregated timelines)
    planet_columns_desc = ['NL_Planet', 'SL_Planet']
    if 'SSL_Planet' in desc_display_df.columns:
        planet_columns_desc.append('SSL_Planet')
    
    styler_desc = desc_display_df.style.applymap(
        lambda x: color_timeline_planets_by_score(x, planet_scores),
        subset=planet_columns_desc
    ).applymap(
        lambda x: color_verdict_cell(x, team_a_name, team_b_name),
        subset=['Verdict']
    )
    st.dataframe(styler_desc, use_container_width=True, height=400)
    st.write(display_results["desc_timeline_analysis"]["summary"])
    
    st.subheader("Moon SSL Timeline - Full Granular Detail")
    st.markdown('<p class="timeline-description">Detailed timeline showing all Sub-Sub Lord periods for precise timing analysis. Useful for identifying exact moments of significant events.</p>', unsafe_allow_html=True)
    moon_timeline_df = display_results["moon_timeline_df"].copy()

    # Convert times to IST for display using the correct pandas method
    moon_timeline_df['Start Time'] = pd.to_datetime(moon_timeline_df['Start Time']).dt.tz_convert('Asia/Kolkata').dt.strftime('%H:%M:%S')
    moon_timeline_df['End Time'] = pd.to_datetime(moon_timeline_df['End Time']).dt.tz_convert('Asia/Kolkata').dt.strftime('%H:%M:%S')

    # Create a view for display, dropping the score column but keeping Verdict and Comment
    moon_display_df = moon_timeline_df.drop(columns=['Score'])
    
    # Apply coloring to planet columns and verdict column
    styler_moon = moon_display_df.style.applymap(
        lambda x: color_timeline_planets_by_score(x, planet_scores),
        subset=['NL_Planet', 'SL_Planet', 'SSL_Planet']
    ).applymap(
        lambda x: color_verdict_cell(x, team_a_name, team_b_name),
        subset=['Verdict']
    )
    st.dataframe(styler_moon, use_container_width=True, height=400)
    st.write(display_results["moon_timeline_analysis"]["summary"])
    
    st.subheader("Favorable Planets")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**For {team_a_name} (Ascendant):**")
        st.json(display_results['asc_timeline_analysis']['favorable_planets'])
    with col2:
        st.write(f"**For {team_b_name} (Descendant):**")
        st.json(display_results['desc_timeline_analysis']['favorable_planets'])
    with col3:
        st.write("**For Moon SSL:**")
        st.json(display_results['moon_timeline_analysis']['favorable_planets'])
        
    st.subheader("Unfavorable Planets")
    col4, col5, col6 = st.columns(3)
    with col4:
        st.write(f"**For {team_a_name} (Ascendant):**")
        st.json(display_results['asc_timeline_analysis']['unfavorable_planets'])
    with col5:
        st.write(f"**For {team_b_name} (Descendant):**")
        st.json(display_results['desc_timeline_analysis']['unfavorable_planets'])
    with col6:
        st.write("**For Moon SSL:**")
        st.json(display_results['moon_timeline_analysis']['unfavorable_planets'])

def run_analysis(match_details):
    """
    Orchestrates the KP core analysis and returns a single, consistent dictionary.
    """
    try:
        engine = KPEngine(match_details['datetime_utc'], match_details['lat'], match_details['lon'])
        analysis_engine = AnalysisEngine(engine, match_details['team_a'], match_details['team_b'])
        
        muhurta_analysis = analysis_engine.analyze_muhurta_chart(scoring_method='proportional')
        planets_df = analysis_engine.get_all_planet_details_df()

        asc_timeline_gen = TimelineGenerator(engine, 'Ascendant')
        # Use aggregated timeline for Ascendant (Star Lord + Sub Lord level)
        asc_timeline_df = asc_timeline_gen.generate_aggregated_timeline_df(match_details['datetime_utc'], match_details['duration_hours'])
        asc_timeline_df, asc_timeline_analysis = analysis_engine.analyze_aggregated_timeline(asc_timeline_df, 'ascendant')
        
        desc_timeline_gen = TimelineGenerator(engine, 'Descendant')
        # Use aggregated timeline for Descendant (Star Lord + Sub Lord level)  
        desc_timeline_df = desc_timeline_gen.generate_aggregated_timeline_df(match_details['datetime_utc'], match_details['duration_hours'])
        desc_timeline_df, desc_timeline_analysis = analysis_engine.analyze_aggregated_timeline(desc_timeline_df, 'descendant')
        
        moon_timeline_gen = TimelineGenerator(engine, 'Moon')
        # Use granular timeline for Moon (full SSL level)
        moon_timeline_df = moon_timeline_gen.generate_timeline_df(match_details['datetime_utc'], match_details['duration_hours'])
        moon_timeline_df, moon_timeline_analysis = analysis_engine.analyze_timeline(moon_timeline_df, 'ascendant')
        
        # Return a single, consistently structured dictionary
        return {
            "muhurta_analysis": muhurta_analysis,
            "planets_df": planets_df,
            "asc_timeline_df": asc_timeline_df,
            "desc_timeline_df": desc_timeline_df,
            "moon_timeline_df": moon_timeline_df,
            "asc_timeline_analysis": asc_timeline_analysis,
            "desc_timeline_analysis": desc_timeline_analysis,
            "moon_timeline_analysis": moon_timeline_analysis,
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
    }
    
    /* Timeline table specific styling */
    .stDataFrame div[data-testid="stDataFrame"] td {
        white-space: normal !important;
        word-wrap: break-word !important;
        padding: 8px !important;
        vertical-align: top !important;
        max-width: none !important;
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
    }
    
    /* Time columns styling */
    .stDataFrame div[data-testid="stDataFrame"] td:nth-child(1),
    .stDataFrame div[data-testid="stDataFrame"] td:nth-child(2) {
        text-align: center !important;
        font-family: monospace !important;
    }
    
    /* Verdict column styling */
    .stDataFrame div[data-testid="stDataFrame"] td:nth-last-child(2) {
        text-align: center !important;
        font-weight: bold !important;
        white-space: normal !important;
    }
    
    /* Timeline section styling */
    .timeline-description {
        font-style: italic;
        color: #666;
        margin-bottom: 10px;
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
        
        # Time input as a text field for flexibility
        time_str = st.text_input("Time of Match (HH:MM)", "20:00")
        
        # Timezone selection
        timezones = pytz.all_timezones
        default_tz_index = timezones.index('Asia/Kolkata') if 'Asia/Kolkata' in timezones else 0
        tz_name = st.selectbox("Timezone", timezones, index=default_tz_index)

        lat = st.number_input("Latitude", value=lat_val, format="%.4f")
        lon = st.number_input("Longitude", value=lon_val, format="%.4f")
        
        match_duration = st.number_input("Match Duration (hours)", min_value=1.0, max_value=8.0, value=3.5, step=0.5)

        if st.button("Generate Predictions"):
            try:
                # Parse the time string
                match_time = datetime.datetime.strptime(time_str, "%H:%M").time()
                
                # Combine date and time
                local_datetime = datetime.datetime.combine(match_date, match_time)
                # Localize the datetime
                local_tz = pytz.timezone(tz_name)
                localized_dt = local_tz.localize(local_datetime)
                # Convert to UTC
                utc_dt = localized_dt.astimezone(pytz.utc)

                with st.spinner("Generating astrological analysis..."):
                    new_analysis = run_analysis({
                        'datetime_utc': utc_dt,
                        'lat': lat,
                        'lon': lon,
                        'duration_hours': match_duration,
                        'team_a': team_a,
                        'team_b': team_b
                    })
                    
                    # Add tab name to the analysis
                    tab_name = f"{team_a} vs {team_b} - {match_date.strftime('%Y-%m-%d')}"
                    new_analysis['tab_name'] = tab_name
                    
                    # Add to analyses list and set as active tab
                    st.session_state.analyses.append(new_analysis)
                    st.session_state.active_tab = len(st.session_state.analyses) - 1
                    st.rerun()
            except ValueError:
                st.error("Invalid time format. Please use HH:MM.")
        
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