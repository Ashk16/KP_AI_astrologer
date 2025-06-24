import pandas as pd
import os
import sys

# --- Path Correction ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from kp_core.kp_engine import KPEngine

class AnalysisEngine:
    """
    Generates astrological analysis and predictions based on a calculated chart.
    """
    def __init__(self, engine: KPEngine, team_a_name: str, team_b_name: str):
        self.engine = engine
        self.team_a = team_a_name
        self.team_b = team_b_name
        self.planets = engine.get_all_planets_df()
        self.cusps = engine.get_all_cusps_df()

    def get_significators(self, planet_name):
        """Finds houses signified by a planet (robust implementation)."""
        significators = set()

        # --- 1. Find the planet's full details ---
        if not isinstance(planet_name, str) or not planet_name:
            return []  # Invalid input

        try:
            # Find the full name (e.g., 'Sun') from the short name ('Su')
            planet_full_name = [p for p in self.planets.index if p.startswith(planet_name)][0]
            planet_info = self.planets.loc[planet_full_name]
        except IndexError:
            return []  # Planet short name not found in index

        # --- 2. Find house the planet occupies ---
        for i in range(1, 13):
            cusp_start = self.cusps.loc[i]['longitude']
            normalized_planet_lon = (planet_info['longitude'] - cusp_start + 360) % 360
            
            next_cusp_num = i % 12 + 1
            cusp_end = self.cusps.loc[next_cusp_num]['longitude']
            normalized_cusp_end = (cusp_end - cusp_start + 360) % 360

            if normalized_planet_lon < normalized_cusp_end:
                significators.add(i)
                break  # A planet can only be in one house

        # --- 3. Find houses ruled by the planet itself ---
        planet_short_name = planet_info.name[:2]
        for cusp, cusp_data in self.cusps.iterrows():
            if cusp_data['sign_lord'] == planet_short_name:
                significators.add(cusp)
        
        # --- 4. Find houses ruled AND occupied by the planet's Nakshatra Lord ---
        planet_nl_short = planet_info['nl']
        
        if isinstance(planet_nl_short, str) and planet_nl_short:
            try:
                nl_full_name = [p for p in self.planets.index if p.startswith(planet_nl_short)][0]
                nl_info = self.planets.loc[nl_full_name]
                nl_short_name = nl_full_name[:2]

                # Add houses ruled by the Nakshatra Lord
                for cusp, cusp_data in self.cusps.iterrows():
                    if cusp_data['sign_lord'] == nl_short_name:
                        significators.add(cusp)

                # Add the house OCCUPIED by the Nakshatra Lord (This is the crucial addition)
                for i in range(1, 13):
                    cusp_start = self.cusps.loc[i]['longitude']
                    normalized_nl_lon = (nl_info['longitude'] - cusp_start + 360) % 360
                    
                    next_cusp_num = i % 12 + 1
                    cusp_end = self.cusps.loc[next_cusp_num]['longitude']
                    normalized_cusp_end = (cusp_end - cusp_start + 360) % 360

                    if normalized_nl_lon < normalized_cusp_end:
                        significators.add(i)
                        break

            except IndexError:
                # Defensively skip if the Nakshatra Lord short name is invalid or not found.
                pass

        return sorted(list(significators))

    def get_all_planet_significators_df(self):
        """
        Calculates significators for all planets and adds them to the planets DataFrame.
        """
        significators_list = []
        # Ensure we iterate in the same order as the DataFrame index
        for planet_name in self.planets.index:
            sigs = self.get_significators(planet_name)
            # Convert list to a comma-separated string for clean display
            significators_list.append(", ".join(map(str, sigs)))

        planets_with_sigs_df = self.planets.copy()
        planets_with_sigs_df['Significators'] = significators_list
        return planets_with_sigs_df

    def analyze_muhurta_chart(self):
        """
        Provides a high-level analysis of the Muhurta chart.
        Rule: Favor the team whose ascendant lord is stronger and signifies favorable houses.
        """
        asc_lord_name = self.cusps.loc[1]['sign_lord']
        desc_lord_name = self.cusps.loc[7]['sign_lord']

        asc_sigs = self.get_significators(asc_lord_name)
        desc_sigs = self.get_significators(desc_lord_name)

        # Favorable houses for victory: 1, 2, 3, 6, 10, 11
        # Unfavorable: 5, 8, 12
        fav_houses = {1, 2, 3, 6, 10, 11}
        unfav_houses = {5, 8, 12}

        asc_score = len(fav_houses.intersection(asc_sigs)) - len(unfav_houses.intersection(desc_sigs))
        desc_score = len(fav_houses.intersection(desc_sigs)) - len(unfav_houses.intersection(asc_sigs))

        synopsis = (
            f"Ascendant Lord ({asc_lord_name}) signifies houses: {asc_sigs}. "
            f"Descendant Lord ({desc_lord_name}) signifies houses: {desc_sigs}. "
            f"Favorable houses for victory for Ascendant are 1, 2, 3, 6, 10, 11. "
            f"Favorable for Descendant are 7, 8, 9, 12, 4, 5."
        )

        if asc_score > desc_score:
            verdict = f"Favors {self.team_a}"
        elif desc_score > asc_score:
            verdict = f"Favors {self.team_b}"
        else:
            verdict = "Neutral / Tightly Contested"
            
        return f"{synopsis}\n\n**Verdict:** {verdict}"

    def analyze_timeline(self, timeline_df: pd.DataFrame):
        """Analyzes a timeline and adds verdict/comment columns."""
        verdicts = []
        comments = []

        for _, row in timeline_df.iterrows():
            ssl_sigs = self.get_significators(row['SSL'])
            
            # Simplified rules for timeline events
            fav_houses = {2, 3, 6, 10, 11} # Strong hitting, victory
            unfav_houses = {5, 8, 12} # Wickets, losses, poor performance

            score = len(fav_houses.intersection(ssl_sigs)) - len(unfav_houses.intersection(ssl_sigs))
            
            verdict = "Neutral"
            comment = f"SSL {row['SSL']} signifies {ssl_sigs}."

            if score > 1:
                verdict = f"Favors {self.team_a}"
                comment += " Very strong period for runs/wickets."
            elif score == 1:
                verdict = f"Favors {self.team_a}"
                comment += " Gentle support for the batting side."
            elif score < -1:
                verdict = f"Favors {self.team_b}"
                comment += " Very difficult period, high chance of losses/wickets."
            elif score == -1:
                verdict = f"Favors {self.team_b}"
                comment += " Some pressure on the batting side."
            
            verdicts.append(verdict)
            comments.append(comment)
            
        timeline_df['Verdict'] = verdicts
        timeline_df['Comment'] = comments
        return timeline_df 