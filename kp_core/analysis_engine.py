import pandas as pd
import os
import sys
from collections import defaultdict

# --- Path Correction ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from kp_core.kp_engine import KPEngine

# --- Weighting System ---

SIGNIFICATOR_RULE_WEIGHTS = {
    1: 1.0,   # Strongest
    2: 0.85,
    3: 0.70,
    4: 0.60   # Weakest
}

HOUSE_WEIGHTS = {
    # Favorable for Ascendant (Team A)
    6: 1.0,   # Victory over opponents
    11: 0.9,  # Gains and success
    1: 0.7,   # The self, vitality
    10: 0.6,  # Performance, status
    3: 0.5,   # Courage, effort
    2: 0.4,   # Resources, runs
    9: 0.2,   # Luck
    # Unfavorable for Ascendant (Team A)
    12: -1.0, # Loss, self-undoing
    8: -0.9,  # Sudden obstacles, defeat
    5: -0.8,  # Opponent's gains
    7: -0.6,  # The opponent
    4: -0.3,  # End of the matter, home comfort
}


class AnalysisEngine:
    """
    Generates astrological analysis and predictions based on a weighted, hierarchical system.
    """
    def __init__(self, engine: KPEngine, team_a_name: str, team_b_name: str):
        self.engine = engine
        self.team_a = team_a_name
        self.team_b = team_b_name
        self.planets = engine.get_all_planets_df()
        self.cusps = engine.get_all_cusps_df()

        # --- Pre-computation for efficiency ---
        self._precompute_chart_data()

    def _precompute_chart_data(self):
        """Pre-calculates essential chart data for quick lookups."""
        self.planet_house_map = {p: self._get_house_occupancy(p_info['longitude']) for p, p_info in self.planets.iterrows()}
        
        self.house_occupants_map = defaultdict(list)
        for planet, house in self.planet_house_map.items():
            if house:
                self.house_occupants_map[house].append(planet)
        
        self.vacant_houses = [h for h in range(1, 13) if not self.house_occupants_map[h]]

    def _get_house_occupancy(self, longitude):
        """Finds which house a given longitude falls into."""
        for i in range(1, 13):
            cusp_start = self.cusps.loc[i]['longitude']
            next_cusp_num = i % 12 + 1
            cusp_end = self.cusps.loc[next_cusp_num]['longitude']
            
            normalized_lon = (longitude - cusp_start + 360) % 360
            normalized_end = (cusp_end - cusp_start + 360) % 360

            if normalized_lon < normalized_end:
                return i
        return None # Should not happen

    def get_significators(self, planet_name):
        """
        Calculates significators for a single planet using the 4-step hierarchical method.
        This version includes robust checks to prevent crashes from invalid data.
        """
        significations = []
        planet_info = self.planets.loc[planet_name]
        star_lord_short_name = planet_info['nl']
        
        # --- Safely resolve Star Lord's full name ---
        star_lord_full_name = None
        if isinstance(star_lord_short_name, str) and star_lord_short_name:
            try:
                star_lord_full_name = [p for p in self.planets.index if p.startswith(star_lord_short_name)][0]
            except IndexError:
                # This handles cases where nl is an invalid short name.
                pass 

        # --- Rule 1: Planet in the Star of Occupants ---
        if star_lord_full_name:
            for house, occupants in self.house_occupants_map.items():
                if star_lord_full_name in occupants:
                    significations.append((house, 1))

        # --- Rule 2: Occupants of a House ---
        house_occupied = self.planet_house_map.get(planet_name)
        if house_occupied:
            significations.append((house_occupied, 2))

        # --- Rule 3: Planet in the Star of the Lord of a House ---
        if star_lord_short_name: # Check short name is valid before using
            for house in self.vacant_houses:
                house_lord_name = self.cusps.loc[house]['sign_lord']
                if star_lord_short_name == house_lord_name:
                    significations.append((house, 3))

        # --- Rule 4: Owners of a House ---
        for house in self.vacant_houses:
            house_lord_name = self.cusps.loc[house]['sign_lord']
            if planet_info.name[:2] == house_lord_name:
                significations.append((house, 4))
        
        # --- Rahu & Ketu Special Logic ---
        if planet_name in ['Rahu', 'Ketu']:
            sign_lord_short = planet_info['sign_lord']
            
            # Defensive check for sign lord
            if isinstance(sign_lord_short, str) and sign_lord_short:
                try:
                    sign_lord_full = [p for p in self.planets.index if p.startswith(sign_lord_short)][0]
                    # Recursively get significators of the agent, but prevent infinite loops
                    if sign_lord_full not in ['Rahu', 'Ketu']:
                         significations.extend(self.get_significators(sign_lord_full))
                except IndexError:
                    pass
        
        return sorted(list(set(significations)), key=lambda x: x[0])
    
    def calculate_planet_score(self, planet_name):
        """Calculates a normalized score for a planet based on its weighted significations."""
        significations = self.get_significators(planet_name)
        if not significations:
            return 0.0

        total_score = 0
        for house, rule in significations:
            rule_weight = SIGNIFICATOR_RULE_WEIGHTS.get(rule, 0)
            house_weight = HOUSE_WEIGHTS.get(house, 0)
            total_score += (rule_weight * house_weight)
            
        unique_houses = set([s[0] for s in significations])
        if not unique_houses:
            return 0.0
            
        return total_score / len(unique_houses)

    def get_all_planet_scores_df(self):
        """Calculates scores for all planets and adds them to the planets DataFrame."""
        scores = {planet: self.calculate_planet_score(planet) for planet in self.planets.index}
        df = self.engine.get_all_planets_df() # Get fresh df
        df['Score'] = df.index.map(scores)
        return df

    def analyze_muhurta_chart(self):
        """Provides a high-level analysis based on Ascendant/Descendant lord scores."""
        asc_lord_name = self.cusps.loc[1]['sign_lord']
        desc_lord_name = self.cusps.loc[7]['sign_lord']
        
        asc_lord_full_name = [p for p in self.planets.index if p.startswith(asc_lord_name)][0]
        desc_lord_full_name = [p for p in self.planets.index if p.startswith(desc_lord_name)][0]

        asc_score = self.calculate_planet_score(asc_lord_full_name)
        desc_score = self.calculate_planet_score(desc_lord_full_name)

        synopsis = (
            f"Ascendant Lord ({asc_lord_name}) Score: {asc_score:.2f}. "
            f"Descendant Lord ({desc_lord_name}) Score: {desc_score:.2f}."
        )

        if asc_score > desc_score:
            verdict = f"Overall chart favors {self.team_a}"
        elif desc_score > asc_score:
            verdict = f"Overall chart favors {self.team_b}"
        else:
            verdict = "Chart is neutral or tightly contested."
            
        return f"{synopsis}\n\n**Verdict:** {verdict}"

    def analyze_timeline(self, timeline_df: pd.DataFrame):
        """Analyzes a timeline using the pre-calculated planet scores."""
        verdicts = []
        comments = []
        scores = []

        planet_scores = {planet: self.calculate_planet_score(planet) for planet in self.planets.index}

        for _, row in timeline_df.iterrows():
            ssl_short_name = row['SSL']
            ssl_full_name = [p for p in self.planets.index if p.startswith(ssl_short_name)][0]
            ssl_score = planet_scores.get(ssl_full_name, 0)
            scores.append(ssl_score)
            
            verdict = "Neutral"
            comment = f"SSL {ssl_short_name} has a score of {ssl_score:.2f}."

            if ssl_score > 0.4:
                verdict = f"Strongly favors {self.team_a}"
                comment += " High scoring or wicket-taking period for Team A."
            elif ssl_score > 0.15:
                verdict = f"Slightly favors {self.team_a}"
                comment += " Period of steady progress for Team A."
            elif ssl_score < -0.4:
                verdict = f"Strongly favors {self.team_b}"
                comment += " Difficult period for Team A, potential for losses."
            elif ssl_score < -0.15:
                 verdict = f"Slightly favors {self.team_b}"
                 comment += " Some pressure on Team A."
            
            verdicts.append(verdict)
            comments.append(comment)
            
        timeline_df['Verdict'] = verdicts
        timeline_df['Comment'] = comments
        timeline_df['Score'] = scores
        return timeline_df 