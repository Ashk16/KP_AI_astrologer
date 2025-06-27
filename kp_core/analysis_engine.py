import pandas as pd
import os
import sys
from collections import defaultdict

# --- Path Correction ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from kp_core.kp_engine import KPEngine, PlanetNameUtils

# --- Weighting System ---

SIGNIFICATOR_RULE_WEIGHTS = {
    1: 1.0,   # Strongest - Planet in star of occupants
    2: 0.5,   # Second - Occupants of a house
    3: 0.3,   # Third - Planet in star of house lord
    4: 0.1    # Weakest - Owners of a house
}

HOUSE_WEIGHTS = {
    # Victory Houses for Ascendant Team (as per KP principles)
    6: 1.0,   # Victory over opponents/enemies (strongest for winning)
    11: 0.9,  # Gains, profits, fulfillment of desires
    1: 0.8,   # Self, strength, health, overall well-being
    10: 0.7,  # Achievements, public recognition, success, performance
    
    # Defeat Houses for Ascendant Team (as per KP principles)
    12: -1.0, # Losses, expenditure, self-undoing (strongest for losing)
    8: -0.9,  # Obstacles, sudden events, crises, "death"/wickets
    7: -0.8,  # The opponent (direct opposition)
    5: -0.7,  # 11th from 7th (opponent's gains), speculation
    9: -0.6,  # 3rd from 7th (opponent's courage), opponent's fortune
    4: -0.5,  # End of activity, change of field, downfall
    
    # Neutral/Secondary Houses
    3: 0.3,   # Courage, effort (generally favorable but not primary)
    2: 0.2,   # Resources, runs (helpful but secondary)
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
        Calculates significators for a single planet using the correct 4-step KP method.
        Now sorted by rule priority (Rule 1 first) and includes conjoint logic for Rahu/Ketu.
        """
        # Standardize planet name for index lookup
        planet_name = PlanetNameUtils.standardize_for_index(planet_name)
        
        significations = []
        planet_info = self.planets.loc[planet_name]
        planet_star_lord = planet_info['nl']  # Nakshatra Lord of this planet
        
        # --- Rule 1: Planet in the Star of Occupants of a House (Strongest) ---
        # FIXED: Check if this planet's star lord is the same as any house occupant
        for house, occupants in self.house_occupants_map.items():
            for occupant in occupants:
                # Convert planet's star lord to full name for comparison
                planet_star_lord_full = PlanetNameUtils.to_full_name(planet_star_lord)
                # Check if planet's star lord matches the occupant itself (not occupant's star lord)
                if planet_star_lord_full == occupant:
                    significations.append((house, 1))
        
        # --- Rule 2: Occupants of a House (Second Strongest) ---
        # Check if this planet itself occupies a house
        house_occupied = self.planet_house_map.get(planet_name)
        if house_occupied:
            significations.append((house_occupied, 2))
        
        # --- Rule 3: Planet in the Star of the Lord of a House (Third Strongest) ---
        # Check if this planet is in the star of any house lord
        # This applies to ALL houses, not just vacant ones
        for house_num in range(1, 13):
            house_lord_short = self.cusps.loc[house_num]['sign_lord']
            if planet_star_lord == house_lord_short:
                # Only add if house is vacant (no occupants)
                if house_num not in self.house_occupants_map or len(self.house_occupants_map[house_num]) == 0:
                    significations.append((house_num, 3))
        
        # --- Rule 4: Owners of a House (Weakest) ---
        # Check if this planet owns any house
        planet_short_name = PlanetNameUtils.to_short_name(planet_name)
        for house_num in range(1, 13):
            house_lord_short = self.cusps.loc[house_num]['sign_lord']
            if planet_short_name == house_lord_short:
                # Only add if house is vacant (no occupants)
                if house_num not in self.house_occupants_map or len(self.house_occupants_map[house_num]) == 0:
                    significations.append((house_num, 4))
        
        # --- Enhanced Rahu & Ketu Agent Logic ---
        if planet_name in ['Rahu', 'Ketu']:
            # Get the house where Rahu/Ketu is positioned
            rahu_ketu_house = self.planet_house_map.get(planet_name)
            
            if rahu_ketu_house:
                # Get all planets in the same house as Rahu/Ketu
                conjoint_planets = [p for p in self.house_occupants_map.get(rahu_ketu_house, []) 
                                  if p != planet_name and p not in ['Rahu', 'Ketu']]
                
                # PRIORITY 1: Conjoint planets (higher importance than sign lord)
                agent_significations = []
                for conjoint_planet in conjoint_planets:
                    try:
                        conjoint_sigs = self.get_significators(conjoint_planet)
                        agent_significations.extend(conjoint_sigs)
                    except (KeyError, RecursionError):
                        pass
                
                # PRIORITY 2: Sign lord (only if no conjoint planets or as additional agent)
                sign_lord_short = planet_info['sign_lord']
                try:
                    sign_lord_full = PlanetNameUtils.to_full_name(sign_lord_short)
                    if (sign_lord_full not in ['Rahu', 'Ketu'] and  # Prevent infinite recursion
                        sign_lord_full not in conjoint_planets):      # Avoid duplication if sign lord is conjoint
                        sign_lord_sigs = self.get_significators(sign_lord_full)
                        agent_significations.extend(sign_lord_sigs)
                except (StopIteration, IndexError, KeyError, RecursionError):
                    pass
                
                # Add all agent significators (preserve original rule strength)
                for house, rule in agent_significations:
                    significations.append((house, rule))
        
        # FIXED: Sort by rule priority first (Rule 1, Rule 2, etc.), then by house number
        return sorted(significations, key=lambda x: (x[1], x[0]))
    
    def _get_day_lord(self) -> str:
        """
        Calculates the day lord based on the weekday.
        Returns the short name of the planet ruling the day.
        """
        # Day lords: Sunday=Sun, Monday=Moon, Tuesday=Mars, Wednesday=Mercury,
        # Thursday=Jupiter, Friday=Venus, Saturday=Saturn
        day_lords = {
            0: "Mo",  # Monday
            1: "Ma",  # Tuesday  
            2: "Me",  # Wednesday
            3: "Ju",  # Thursday
            4: "Ve",  # Friday
            5: "Sa",  # Saturday
            6: "Su"   # Sunday
        }
        
        weekday = self.engine.utc_dt.weekday()  # 0=Monday, 6=Sunday
        return day_lords[weekday]

    def _get_house_weight(self, house_num: int, perspective: str = 'ascendant') -> float:
        """
        Gets the weight for a house based on the perspective (ascendant or descendant).
        For descendant perspective, the weights are reversed (e.g., house 1 becomes house 7).
        
        Args:
            house_num: The house number (1-12)
            perspective: Either 'ascendant' or 'descendant'
            
        Returns:
            float: The weight for the house from the given perspective
        """
        if perspective.lower() == 'descendant':
            # For descendant perspective, we need to:
            # 1. Map the house to its opposite house (e.g., 1->7, 2->8, etc.)
            # 2. Reverse the sign of the weight
            opposite_house = (house_num + 6) if house_num <= 6 else (house_num - 6)
            return -HOUSE_WEIGHTS.get(opposite_house, 0)
        return HOUSE_WEIGHTS.get(house_num, 0)

    def calculate_planet_score(self, planet_name: str, perspective: str = 'ascendant') -> float:
        """
        Calculates a score for a planet based on its weighted significations.
        
        Args:
            planet_name: Name of the planet (can be either short or full name)
            perspective: Either 'ascendant' or 'descendant'
            
        Returns:
            float: The calculated score from the given perspective
        """
        # Standardize planet name for processing
        planet_name = PlanetNameUtils.standardize_for_index(planet_name)

        significations = self.get_significators(planet_name)
        if not significations:
            return 0.0

        total_score = 0
        unique_houses = set()
        for house, rule in significations:
            rule_weight = SIGNIFICATOR_RULE_WEIGHTS.get(rule, 0)
            house_weight = self._get_house_weight(house, perspective)
            total_score += (rule_weight * house_weight)
            unique_houses.add(house)
            
        # Return total score normalized by number of unique houses
        return total_score / len(unique_houses) if unique_houses else 0.0

    def get_all_planet_scores_df(self):
        """Calculates scores for all planets and adds them to the planets DataFrame."""
        scores = {planet: self.calculate_planet_score(planet) for planet in self.planets.index}
        df = self.engine.get_all_planets_df() # Get fresh df
        df['Score'] = df.index.map(scores)
        return df

    def get_all_planet_details_df(self):
        """
        Calculates scores and significators for all planets and returns a comprehensive DataFrame.
        """
        df = self.engine.get_all_planets_df().copy()
        
        scores = {planet: self.calculate_planet_score(planet) for planet in df.index}
        significators_str = {
            planet: ", ".join(map(str, [s[0] for s in self.get_significators(planet)])) 
            for planet in df.index
        }

        df['Score'] = df.index.map(scores)
        df['Significators'] = df.index.map(significators_str)
        return df

    def analyze_muhurta_chart(self):
        """
        Enhanced KP Muhurta Chart Analysis with CSSL scores and match competitiveness analysis.
        """
        analysis_parts = []
        
        # --- Quick Team Setup ---
        analysis_parts.append(f"🏏 **{self.team_a}** (Ascendant) vs **{self.team_b}** (Descendant)")
        analysis_parts.append("")
        
        # --- CSSL 1 Analysis (Most Critical) ---
        cssl_1 = self.cusps.loc[1]['ssl']
        cssl_1_full = PlanetNameUtils.to_full_name(cssl_1)
        cssl_1_sigs = self.get_significators(cssl_1_full) if cssl_1_full in self.planets.index else []
        
        # Calculate CSSL 1 numerical score
        cssl_1_score = self.calculate_planet_score(cssl_1_full, 'ascendant') if cssl_1_full in self.planets.index else 0.0
        
        analysis_parts.append("**📊 CSSL 1 (Primary Indicator):**")
        analysis_parts.append(f"• Planet: **{cssl_1}** | Score: **{cssl_1_score:+.2f}**")
        
        if cssl_1_sigs:
            victory_houses = [h for h, r in cssl_1_sigs if h in [1, 6, 10, 11]]
            defeat_houses = [h for h, r in cssl_1_sigs if h in [4, 5, 7, 8, 9, 12]]
            
            sig_summary = []
            for house, rule in cssl_1_sigs:
                sig_summary.append(f"H{house}(R{rule})")
            
            analysis_parts.append(f"• Houses: {', '.join(sig_summary)}")
            analysis_parts.append(f"• Victory Houses: {victory_houses} | Defeat Houses: {defeat_houses}")
            
            # Enhanced verdict based on both house count and numerical score
            if cssl_1_score > 0.3:
                cssl_1_verdict = f"✅ **Strong Favor {self.team_a}**"
            elif cssl_1_score > 0:
                cssl_1_verdict = f"✅ **Favors {self.team_a}**"
            elif cssl_1_score < -0.3:
                cssl_1_verdict = f"❌ **Strong Favor {self.team_b}**"
            elif cssl_1_score < 0:
                cssl_1_verdict = f"❌ **Favors {self.team_b}**"
            else:
                cssl_1_verdict = "⚖️ **Close Contest**"
            
            analysis_parts.append(f"• Result: {cssl_1_verdict}")
        else:
            analysis_parts.append("• Result: ⚪ **Neutral**")
            cssl_1_verdict = "NEUTRAL"
        
        analysis_parts.append("")
        
        # --- CSSL 6 Analysis ---
        cssl_6 = self.cusps.loc[6]['ssl']
        cssl_6_full = PlanetNameUtils.to_full_name(cssl_6)
        cssl_6_sigs = self.get_significators(cssl_6_full) if cssl_6_full in self.planets.index else []
        
        # Calculate CSSL 6 numerical score
        cssl_6_score = self.calculate_planet_score(cssl_6_full, 'ascendant') if cssl_6_full in self.planets.index else 0.0
        
        analysis_parts.append("**🎯 CSSL 6 (Victory Confirmation):**")
        analysis_parts.append(f"• Planet: **{cssl_6}** | Score: **{cssl_6_score:+.2f}**")
        
        if cssl_6_sigs:
            victory_houses = [h for h, r in cssl_6_sigs if h in [1, 6, 10, 11]]
            defeat_houses = [h for h, r in cssl_6_sigs if h in [4, 5, 7, 8, 9, 12]]
            
            sig_summary = []
            for house, rule in cssl_6_sigs:
                sig_summary.append(f"H{house}(R{rule})")
            
            analysis_parts.append(f"• Houses: {', '.join(sig_summary)}")
            
            # Enhanced verdict based on both house count and numerical score
            if cssl_6_score > 0.3:
                cssl_6_verdict = f"✅ **Strong Confirm {self.team_a}**"
            elif cssl_6_score > 0:
                cssl_6_verdict = f"✅ **Confirms {self.team_a}**"
            elif cssl_6_score < -0.3:
                cssl_6_verdict = f"❌ **Strong Confirm {self.team_b}**"
            elif cssl_6_score < 0:
                cssl_6_verdict = f"❌ **Confirms {self.team_b}**"
            else:
                cssl_6_verdict = "⚖️ **Mixed**"
            
            analysis_parts.append(f"• Result: {cssl_6_verdict}")
        else:
            analysis_parts.append("• Result: ⚪ **Neutral**")
            cssl_6_verdict = "NEUTRAL"
        
        analysis_parts.append("")
        
        # --- Ruling Planets Analysis ---
        asc_sign_lord = self.cusps.loc[1]['sign_lord']
        asc_star_lord = self.cusps.loc[1]['nl']
        moon_sign_lord = self.planets.loc['Moon']['sign_lord'] if 'Moon' in self.planets.index else None
        moon_star_lord = self.planets.loc['Moon']['nl'] if 'Moon' in self.planets.index else None
        day_lord = self._get_day_lord()
        
        ruling_planets = list(set([rp for rp in [asc_sign_lord, asc_star_lord, moon_sign_lord, moon_star_lord, day_lord] if rp is not None]))
        
        analysis_parts.append("**🌟 Ruling Planets:**")
        analysis_parts.append(f"• Planets: {', '.join(ruling_planets)}")
        
        rp_team_a_count = 0
        rp_team_b_count = 0
        
        for rp in ruling_planets:
            rp_full = PlanetNameUtils.to_full_name(rp)
            if rp_full in self.planets.index:
                rp_sigs = self.get_significators(rp_full)
                if rp_sigs:
                    rp_victory_houses = [h for h, r in rp_sigs if h in [1, 6, 10, 11]]
                    rp_defeat_houses = [h for h, r in rp_sigs if h in [4, 5, 7, 8, 9, 12]]
                    
                    if len(rp_victory_houses) > len(rp_defeat_houses):
                        rp_team_a_count += 1
                    elif len(rp_defeat_houses) > len(rp_victory_houses):
                        rp_team_b_count += 1
        
        analysis_parts.append(f"• Support: {self.team_a}({rp_team_a_count}) vs {self.team_b}({rp_team_b_count})")
        
        if rp_team_a_count > rp_team_b_count:
            rp_verdict = f"✅ **Support {self.team_a}**"
        elif rp_team_b_count > rp_team_a_count:
            rp_verdict = f"❌ **Support {self.team_b}**"
        else:
            rp_verdict = "⚖️ **Mixed Support**"
        
        analysis_parts.append(f"• Result: {rp_verdict}")
        analysis_parts.append("")
        
        # --- Match Competitiveness Analysis ---
        analysis_parts.append("**⚖️ MATCH COMPETITIVENESS:**")
        
        # Calculate total CSSL score difference
        cssl_score_diff = abs(cssl_1_score - cssl_6_score)
        avg_cssl_score = abs((cssl_1_score + cssl_6_score) / 2)
        
        analysis_parts.append(f"• CSSL 1 Score: **{cssl_1_score:+.2f}** | CSSL 6 Score: **{cssl_6_score:+.2f}**")
        analysis_parts.append(f"• Average Score: **{avg_cssl_score:.2f}** | Score Difference: **{cssl_score_diff:.2f}**")
        
        # Determine match competitiveness
        if avg_cssl_score > 0.6:
            competitiveness = "🔥 **ONE-SIDED MATCH**"
            competitiveness_desc = "Clear dominance indicated"
        elif avg_cssl_score > 0.3:
            competitiveness = "⚡ **MODERATE ADVANTAGE**"
            competitiveness_desc = "One team has clear edge"
        elif avg_cssl_score > 0.15:
            competitiveness = "🎯 **CLOSE CONTEST**"
            competitiveness_desc = "Fairly balanced match"
        else:
            competitiveness = "🎲 **VERY CLOSE MATCH**"
            competitiveness_desc = "Extremely tight contest"
        
        analysis_parts.append(f"• Type: {competitiveness}")
        analysis_parts.append(f"• Description: {competitiveness_desc}")
        analysis_parts.append("")
        
        # --- Final Prediction ---
        analysis_parts.append("**🏆 FINAL PREDICTION:**")
        
        team_a_scores = 0
        team_b_scores = 0
        total_weight = 0
        
        # Count indicators with weighted scoring
        if self.team_a in cssl_1_verdict:
            team_a_scores += 2  # CSSL 1 gets double weight
            total_weight += 2
        elif self.team_b in cssl_1_verdict:
            team_b_scores += 2
            total_weight += 2
        else:
            total_weight += 2
            
        if self.team_a in cssl_6_verdict:
            team_a_scores += 1.5  # CSSL 6 gets 1.5x weight
            total_weight += 1.5
        elif self.team_b in cssl_6_verdict:
            team_b_scores += 1.5
            total_weight += 1.5
        else:
            total_weight += 1.5
            
        if self.team_a in rp_verdict:
            team_a_scores += 1  # Ruling planets get normal weight
            total_weight += 1
        elif self.team_b in rp_verdict:
            team_b_scores += 1
            total_weight += 1
        else:
            total_weight += 1
        
        # Calculate weighted percentages
        team_a_percentage = (team_a_scores / total_weight) * 100 if total_weight > 0 else 0
        team_b_percentage = (team_b_scores / total_weight) * 100 if total_weight > 0 else 0
        
        analysis_parts.append(f"• Weighted Score: {self.team_a}({team_a_scores:.1f}) vs {self.team_b}({team_b_scores:.1f})")
        analysis_parts.append(f"• Win Probability: {self.team_a}({team_a_percentage:.0f}%) vs {self.team_b}({team_b_percentage:.0f}%)")
        
        # Enhanced final verdict
        score_difference = abs(team_a_scores - team_b_scores)
        
        if team_a_scores > team_b_scores:
            if score_difference >= 2:
                final_verdict = f"🏆 **{self.team_a} STRONGLY FAVORED**"
                confidence = "Very High"
            elif score_difference >= 1:
                final_verdict = f"🏆 **{self.team_a} PREDICTED TO WIN**"
                confidence = "High"
            else:
                final_verdict = f"🏆 **{self.team_a} SLIGHT EDGE**"
                confidence = "Medium"
        elif team_b_scores > team_a_scores:
            if score_difference >= 2:
                final_verdict = f"🏆 **{self.team_b} STRONGLY FAVORED**"
                confidence = "Very High"
            elif score_difference >= 1:
                final_verdict = f"🏆 **{self.team_b} PREDICTED TO WIN**"
                confidence = "High"
            else:
                final_verdict = f"🏆 **{self.team_b} SLIGHT EDGE**"
                confidence = "Medium"
        else:
            final_verdict = "⚖️ **EXTREMELY CLOSE MATCH**"
            confidence = "Low"
        
        analysis_parts.append(f"• Confidence: **{confidence}**")
        analysis_parts.append(f"• {final_verdict}")
        
        return "\n".join(analysis_parts)

    def _generate_verdict_and_comment(self, timeline_row: pd.Series, perspective: str = 'ascendant') -> tuple:
        """
        Generates verdict and comment using multi-layered KP analysis:
        Star Lord promises → Sub Lord modifies → Sub-Sub Lord delivers
        
        Args:
            timeline_row: Row from timeline DataFrame with NL_Planet, SL_Planet, SSL_Planet
            perspective: Either 'ascendant' or 'descendant'
            
        Returns:
            tuple: (verdict, comment)
        """
        nl_planet = timeline_row.get('NL_Planet')
        sl_planet = timeline_row.get('SL_Planet') 
        ssl_planet = timeline_row.get('SSL_Planet')
        
        # Handle missing data
        if pd.isna(nl_planet) or pd.isna(sl_planet) or pd.isna(ssl_planet):
            return "Neutral", "Insufficient planetary data for analysis"
        
        # Determine team names based on perspective
        team_name = self.team_a if perspective == 'ascendant' else self.team_b
        opponent_name = self.team_b if perspective == 'ascendant' else self.team_a
        
        # === LAYER 1: STAR LORD ANALYSIS (The Promise) ===
        nl_standardized = PlanetNameUtils.standardize_for_index(nl_planet)
        nl_significators = self.get_significators(nl_standardized) if nl_standardized in self.planets.index else []
        
        nl_victory_houses = [h for h, r in nl_significators if h in [1, 6, 10, 11]]
        nl_defeat_houses = [h for h, r in nl_significators if h in [4, 5, 7, 8, 9, 12]]
        
        if len(nl_victory_houses) > len(nl_defeat_houses) and nl_victory_houses:
            nl_promise = "VICTORY"
            nl_promise_desc = f"promises victory (H{','.join(map(str, nl_victory_houses))})"
        elif len(nl_defeat_houses) > len(nl_victory_houses) and nl_defeat_houses:
            nl_promise = "DEFEAT" 
            nl_promise_desc = f"promises challenges (H{','.join(map(str, nl_defeat_houses))})"
        elif nl_victory_houses and nl_defeat_houses:
            nl_promise = "MIXED"
            nl_promise_desc = f"promises mixed results (V:{','.join(map(str, nl_victory_houses))} D:{','.join(map(str, nl_defeat_houses))})"
        else:
            nl_promise = "NEUTRAL"
            nl_promise_desc = "promises neutral period"
        
        # === LAYER 2: SUB LORD ANALYSIS (The Modifier) ===
        sl_standardized = PlanetNameUtils.standardize_for_index(sl_planet)
        sl_significators = self.get_significators(sl_standardized) if sl_standardized in self.planets.index else []
        
        sl_victory_houses = [h for h, r in sl_significators if h in [1, 6, 10, 11]]
        sl_defeat_houses = [h for h, r in sl_significators if h in [4, 5, 7, 8, 9, 12]]
        
        # Determine how Sub Lord modifies the promise
        if nl_promise == "VICTORY":
            if len(sl_victory_houses) > len(sl_defeat_houses):
                sl_modification = "SUPPORTS"
                sl_mod_desc = f"supports promise (H{','.join(map(str, sl_victory_houses))})"
            elif len(sl_defeat_houses) > len(sl_victory_houses):
                sl_modification = "OPPOSES"
                sl_mod_desc = f"opposes promise (H{','.join(map(str, sl_defeat_houses))})"
            else:
                sl_modification = "NEUTRAL"
                sl_mod_desc = "neutral on promise"
        elif nl_promise == "DEFEAT":
            if len(sl_defeat_houses) > len(sl_victory_houses):
                sl_modification = "SUPPORTS"
                sl_mod_desc = f"supports promise (H{','.join(map(str, sl_defeat_houses))})"
            elif len(sl_victory_houses) > len(sl_defeat_houses):
                sl_modification = "OPPOSES"
                sl_mod_desc = f"opposes promise (H{','.join(map(str, sl_victory_houses))})"
            else:
                sl_modification = "NEUTRAL"
                sl_mod_desc = "neutral on promise"
        else:  # MIXED or NEUTRAL promise
            if len(sl_victory_houses) > len(sl_defeat_houses):
                sl_modification = "CLARIFIES_VICTORY"
                sl_mod_desc = f"clarifies toward victory (H{','.join(map(str, sl_victory_houses))})"
            elif len(sl_defeat_houses) > len(sl_victory_houses):
                sl_modification = "CLARIFIES_DEFEAT"
                sl_mod_desc = f"clarifies toward challenges (H{','.join(map(str, sl_defeat_houses))})"
            else:
                sl_modification = "MAINTAINS"
                sl_mod_desc = "maintains mixed signals"
        
        # === LAYER 3: SUB-SUB LORD ANALYSIS (The Deliverer) ===
        ssl_standardized = PlanetNameUtils.standardize_for_index(ssl_planet)
        ssl_significators = self.get_significators(ssl_standardized) if ssl_standardized in self.planets.index else []
        
        ssl_victory_houses = [h for h, r in ssl_significators if h in [1, 6, 10, 11]]
        ssl_defeat_houses = [h for h, r in ssl_significators if h in [4, 5, 7, 8, 9, 12]]
        
        if len(ssl_victory_houses) > len(ssl_defeat_houses) and ssl_victory_houses:
            ssl_delivery = "DELIVERS_VICTORY"
            ssl_del_desc = f"delivers victory (H{','.join(map(str, ssl_victory_houses))})"
        elif len(ssl_defeat_houses) > len(ssl_victory_houses) and ssl_defeat_houses:
            ssl_delivery = "DELIVERS_DEFEAT"
            ssl_del_desc = f"delivers challenges (H{','.join(map(str, ssl_defeat_houses))})"
        elif ssl_victory_houses and ssl_defeat_houses:
            ssl_delivery = "PARTIAL_DELIVERY"
            ssl_del_desc = f"partial delivery (V:{','.join(map(str, ssl_victory_houses))} D:{','.join(map(str, ssl_defeat_houses))})"
        else:
            ssl_delivery = "NEUTRAL_DELIVERY"
            ssl_del_desc = "neutral delivery"
        
        # === SYNTHESIS: COMBINE ALL LAYERS FOR FINAL VERDICT ===
        confidence_level = "MEDIUM"
        
        # Determine final verdict based on layer combinations
        if nl_promise == "VICTORY":
            if sl_modification == "SUPPORTS" and ssl_delivery == "DELIVERS_VICTORY":
                verdict = f"Strong Advantage {team_name}"
                cricket_context = "Excellent period for building partnerships and dominating opponents"
                confidence_level = "HIGH"
            elif sl_modification == "OPPOSES" and ssl_delivery == "DELIVERS_DEFEAT":
                verdict = f"Advantage {opponent_name}"
                cricket_context = "Promised advantage turns into setback - wickets or pressure likely"
                confidence_level = "HIGH"
            elif ssl_delivery == "DELIVERS_VICTORY":
                verdict = f"Advantage {team_name}"
                cricket_context = "Good period despite some obstacles"
                confidence_level = "MEDIUM"
            elif ssl_delivery == "DELIVERS_DEFEAT":
                verdict = f"Challenging Period {team_name}"
                cricket_context = "Tough phase with unexpected difficulties"
                confidence_level = "MEDIUM"
            else:
                verdict = "Balanced with Slight Edge"
                cricket_context = "Mixed signals, marginal advantage varies"
                confidence_level = "LOW"
                
        elif nl_promise == "DEFEAT":
            if sl_modification == "SUPPORTS" and ssl_delivery == "DELIVERS_DEFEAT":
                verdict = f"Strong Advantage {opponent_name}"
                cricket_context = "Consistent pressure phase - wickets and obstacles likely"
                confidence_level = "HIGH"
            elif sl_modification == "OPPOSES" and ssl_delivery == "DELIVERS_VICTORY":
                verdict = f"Advantage {team_name}"
                cricket_context = "Unexpected turnaround - recovery from difficult start"
                confidence_level = "HIGH"
            elif ssl_delivery == "DELIVERS_VICTORY":
                verdict = f"Advantage {team_name}"
                cricket_context = "Tough start but eventual breakthrough"
                confidence_level = "MEDIUM"
            elif ssl_delivery == "DELIVERS_DEFEAT":
                verdict = f"Challenging Period {team_name}"
                cricket_context = "Difficult phase with mounting pressure"
                confidence_level = "MEDIUM"
            else:
                verdict = "Balanced with Slight Edge"
                cricket_context = "Uncertain period with variable momentum"
                confidence_level = "LOW"
                
        else:  # MIXED or NEUTRAL promise
            if ssl_delivery == "DELIVERS_VICTORY":
                verdict = f"Advantage {team_name}"
                cricket_context = "Uncertain start but eventual team dominance"
                confidence_level = "MEDIUM"
            elif ssl_delivery == "DELIVERS_DEFEAT":
                verdict = f"Advantage {opponent_name}"
                cricket_context = "Mixed signals resolve into opposition advantage"
                confidence_level = "MEDIUM"
            elif ssl_delivery == "PARTIAL_DELIVERY":
                verdict = "Highly Unpredictable"
                cricket_context = "Momentum shift likely - either team could break through"
                confidence_level = "LOW"
            else:
                verdict = "Balanced Period"
                cricket_context = "Evenly matched phase with gradual developments"
                confidence_level = "LOW"
        
        # === GENERATE DETAILED COMMENT ===
        comment_parts = []
        comment_parts.append(f"🌟 {nl_planet} {nl_promise_desc}")
        comment_parts.append(f"⚖️ {sl_planet} {sl_mod_desc}")
        comment_parts.append(f"🎯 {ssl_planet} {ssl_del_desc}")
        comment_parts.append(f"🏏 {cricket_context}")
        comment_parts.append(f"📊 Confidence: {confidence_level}")
        
        detailed_comment = " | ".join(comment_parts)
        
        return verdict, detailed_comment

    def analyze_timeline(self, timeline_df, perspective='ascendant'):
        """
        Analyzes a timeline DataFrame and returns both the scored DataFrame and analysis.
        
        Args:
            timeline_df: DataFrame with timeline data
            perspective: Either 'ascendant' or 'descendant'
            
        Returns:
            tuple: (scored_timeline_df, analysis_dict)
        """
        # Add score column based on the SSL planet
        timeline_df['Score'] = timeline_df['SSL_Planet'].apply(
            lambda x: self.calculate_planet_score(x, perspective) if pd.notna(x) else 0.0
        )
        
        # Add Verdict and Comment columns
        verdict_comment_data = timeline_df.apply(
            lambda row: self._generate_verdict_and_comment(row, perspective), axis=1
        )
        timeline_df['Verdict'] = [vc[0] for vc in verdict_comment_data]
        timeline_df['Comment'] = [vc[1] for vc in verdict_comment_data]
        
        # Calculate average score
        avg_score = timeline_df['Score'].mean()
        
        # Identify favorable and unfavorable planets
        favorable_planets = []
        unfavorable_planets = []
        
        # Get unique planets from NL, SL, and SSL columns
        planet_columns = ['NL_Planet', 'SL_Planet', 'SSL_Planet']
        unique_planets = pd.unique(timeline_df[planet_columns].values.ravel())
        unique_planets = [p for p in unique_planets if pd.notna(p)]
        
        for planet in unique_planets:
            score = self.calculate_planet_score(planet, perspective)
            if score > 0:
                favorable_planets.append(planet)
            elif score < 0:
                unfavorable_planets.append(planet)
        
        # Generate analysis summary with team-specific context
        team_name = self.team_a if perspective == 'ascendant' else self.team_b
        
        if abs(avg_score) < 0.1:
            summary = f"The average score for this timeline is {avg_score:.2f}. The timeline appears balanced, suggesting a tightly contested match."
        elif avg_score > 0:
            summary = f"The average score for this timeline is {avg_score:.2f}. This indicates a general advantage for {team_name}."
        else:
            opponent_name = self.team_b if perspective == 'ascendant' else self.team_a
            summary = f"The average score for this timeline is {avg_score:.2f}. This indicates a general advantage for {opponent_name}."
        
        analysis = {
            "summary": summary,
            "favorable_planets": sorted(favorable_planets),
            "unfavorable_planets": sorted(unfavorable_planets)
        }
        
        return timeline_df, analysis 