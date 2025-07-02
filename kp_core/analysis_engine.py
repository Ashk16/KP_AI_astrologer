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
    
    # === EXALTATION MAPPING (Classical KP) ===
    EXALTATION_MAPPING = {
        'Sun': ('Aries', 10.0),      # Sun exalted in Aries at 10°
        'Moon': ('Taurus', 3.0),     # Moon exalted in Taurus at 3°
        'Mars': ('Capricorn', 28.0), # Mars exalted in Capricorn at 28°
        'Mercury': ('Virgo', 15.0),  # Mercury exalted in Virgo at 15°
        'Jupiter': ('Cancer', 5.0),  # Jupiter exalted in Cancer at 5°
        'Venus': ('Pisces', 27.0),   # Venus exalted in Pisces at 27°
        'Saturn': ('Libra', 20.0),   # Saturn exalted in Libra at 20°
        'Rahu': ('Taurus', 20.0),    # Rahu exalted in Taurus at 20°
        'Ketu': ('Scorpio', 15.0),   # Ketu exalted in Scorpio at 15°
    }
    
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
        For descendant perspective, victory/defeat meanings are reversed.
        
        Args:
            house_num: The house number (1-12)
            perspective: Either 'ascendant' or 'descendant'
            
        Returns:
            float: The weight for the house from the given perspective
        """
        base_weight = HOUSE_WEIGHTS.get(house_num, 0)
        
        if perspective.lower() == 'descendant':
            # For descendant perspective, simply reverse the sign of the weight
            # What's good for ascendant becomes bad for descendant, and vice versa
            # H8 (defeat for ascendant) becomes victory for descendant = +0.9
            # H6 (victory for ascendant) becomes defeat for descendant = -1.0
            return -base_weight
        
        return base_weight

    def calculate_planet_score(self, planet_name: str, perspective: str = 'ascendant') -> float:
        """
        Calculates a score for a planet based on its weighted significations.
        Now includes Classical KP Debilitation and Exaltation Rules for accurate predictions.
        
        Args:
            planet_name: Name of the planet (can be either short or full name)
            perspective: Either 'ascendant' or 'descendant'
            
        Returns:
            float: The calculated score from the given perspective (with all corrections)
        """
        # Standardize planet name for processing
        planet_name = PlanetNameUtils.standardize_for_index(planet_name)

        # === CLASSICAL KP AGENCY RULE FOR DEBILITATED PLANETS ===
        if self._is_planet_debilitated(planet_name):
            return self._calculate_debilitated_planet_score(planet_name, perspective)
        
        # === NORMAL PLANET CALCULATION ===
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
            
        # Calculate base score normalized by number of unique houses
        base_score = total_score / len(unique_houses) if unique_houses else 0.0
        
        # === CLASSICAL KP CORRECTIONS (FOR NON-DEBILITATED PLANETS) ===
        exaltation_enhancements = self._apply_exaltation_rules(planet_name, base_score, perspective)
        
        # Return corrected score
        final_score = base_score + exaltation_enhancements
        return final_score

    def _is_planet_debilitated(self, planet_name: str) -> bool:
        """
        Check if a planet is debilitated in its current sign.
        
        Args:
            planet_name: Standardized planet name
            
        Returns:
            bool: True if planet is debilitated, False otherwise
        """
        if planet_name not in self.planets.index:
            return False
            
        planet_info = self.planets.loc[planet_name]
        planet_sign = planet_info['sign']
        
        # === DEBILITATION MAPPING (Classical KP) ===
        DEBILITATION_MAPPING = {
            'Sun': 'Libra',      # Sun debilitated in Libra
            'Moon': 'Scorpio',   # Moon debilitated in Scorpio
            'Mars': 'Cancer',    # Mars debilitated in Cancer
            'Mercury': 'Pisces', # Mercury debilitated in Pisces
            'Jupiter': 'Capricorn', # Jupiter debilitated in Capricorn
            'Venus': 'Virgo',    # Venus debilitated in Virgo
            'Saturn': 'Aries',   # Saturn debilitated in Aries
        }
        
        return planet_name in DEBILITATION_MAPPING and planet_sign == DEBILITATION_MAPPING[planet_name]

    def _calculate_debilitated_planet_score(self, planet_name: str, perspective: str = 'ascendant') -> float:
        """
        Calculate score for a debilitated planet using the Classical KP Agency Rule.
        
        The debilitated planet acts as an agent of its sign lord and gives results
        according to the sign lord's direction and house significators.
        
        Args:
            planet_name: Standardized debilitated planet name
            perspective: Either 'ascendant' or 'descendant'
            
        Returns:
            float: Score based on sign lord's influence and direction
        """
        if planet_name not in self.planets.index:
            return 0.0
            
        planet_info = self.planets.loc[planet_name]
        planet_sign = planet_info['sign']
        
        # === SIGN LORD MAPPING ===
        SIGN_LORD_MAPPING = {
            'Aries': 'Mars', 'Taurus': 'Venus', 'Gemini': 'Mercury', 'Cancer': 'Moon',
            'Leo': 'Sun', 'Virgo': 'Mercury', 'Libra': 'Venus', 'Scorpio': 'Mars',
            'Sagittarius': 'Jupiter', 'Capricorn': 'Saturn', 'Aquarius': 'Saturn', 'Pisces': 'Jupiter'
        }
        
        # Get the sign lord
        sign_lord_name = SIGN_LORD_MAPPING.get(planet_sign)
        if not sign_lord_name or sign_lord_name not in self.planets.index:
            return 0.0  # Fallback if sign lord not found
        
        # === CLASSICAL KP AGENCY RULE ===
        # Step 1: Calculate sign lord's base score and direction
        sign_lord_base_score = self._calculate_base_score(sign_lord_name, perspective)
        
        # Step 2: Get sign lord's house significators (what the debilitated planet will deliver)
        sign_lord_significators = self.get_significators(sign_lord_name)
        if not sign_lord_significators:
            return 0.0
        
        # Step 3: Calculate agency score using sign lord's houses but with modifications
        total_score = 0
        unique_houses = set()
        for house, rule in sign_lord_significators:
            rule_weight = SIGNIFICATOR_RULE_WEIGHTS.get(rule, 0)
            house_weight = self._get_house_weight(house, perspective)
            total_score += (rule_weight * house_weight)
            unique_houses.add(house)
            
        agency_base_score = total_score / len(unique_houses) if unique_houses else 0.0
        
        # Step 4: Apply sign lord's directional influence
        # If sign lord favors ascendant (positive), debilitated planet should also favor ascendant
        # If sign lord favors descendant (negative), debilitated planet should also favor descendant
        if sign_lord_base_score > 0:
            # Sign lord favors ascendant - apply positive influence
            agency_score = abs(agency_base_score)
        elif sign_lord_base_score < 0:
            # Sign lord favors descendant - apply negative influence  
            agency_score = -abs(agency_base_score)
        else:
            # Sign lord is neutral
            agency_score = agency_base_score
        
        # Step 5: Apply debilitation corrections if sign lord is strong enough
        if abs(sign_lord_base_score) > 0.03:  # Sign lord strong enough for Neecha Bhanga
            # Calculate Neecha Bhanga correction
            neecha_bhanga_strength = abs(sign_lord_base_score) * 0.6
            
            # Enhanced strength if sign lord significates victory houses
            victory_houses = [h for h, r in sign_lord_significators if h in [1, 6, 10, 11]]
            if len(victory_houses) >= 2:
                neecha_bhanga_strength *= 1.2
            
            # Apply correction in the same direction as sign lord
            if sign_lord_base_score > 0:
                agency_score += neecha_bhanga_strength
            else:
                agency_score -= neecha_bhanga_strength
        
        return agency_score

    def _apply_exaltation_rules(self, planet_name: str, base_score: float, perspective: str = 'ascendant') -> float:
        """
        Applies Classical KP Exaltation Enhancement Rules:
        
        CORE PRINCIPLE: "An exalted planet increases the INTENSITY of what it is supposed to do."
        
        - If base score is positive (+0.60), exaltation makes it MORE positive (e.g., +0.90)
        - If base score is negative (-0.45), exaltation makes it MORE negative (e.g., -0.67)
        
        Exaltation amplifies the planet's natural indications based on house significators.
        It does NOT reverse negative influences - it intensifies them!
        
        Args:
            planet_name: Standardized planet name
            base_score: Original calculated score based on house significators
            perspective: Either 'ascendant' or 'descendant'
            
        Returns:
            float: Enhancement amount to be added to base score (can be positive or negative)
        """
        if planet_name not in self.planets.index:
            return 0.0
            
        planet_info = self.planets.loc[planet_name]
        planet_sign = planet_info['sign']
        planet_longitude = planet_info['longitude']
        
        # Check if planet is exalted
        if planet_name not in self.EXALTATION_MAPPING:
            return 0.0  # No exaltation rules for some bodies
            
        exalt_sign, exalt_degree = self.EXALTATION_MAPPING[planet_name]
        is_exalted = planet_sign == exalt_sign
        
        if not is_exalted:
            return 0.0  # Planet not exalted, no enhancement needed
            
        # Calculate degree proximity to exact exaltation
        degree_in_sign = planet_longitude % 30  # Get degree within the sign
        distance_from_exact = abs(degree_in_sign - exalt_degree)
        
        # Proximity factor: closer to exact degree = stronger exaltation
        if distance_from_exact <= 3.0:  # Within 3 degrees of exact (very close)
            proximity_factor = 1.0  # 100% strength
        elif distance_from_exact <= 8.0:  # Within 8 degrees (close)
            proximity_factor = 0.9 - ((distance_from_exact - 3.0) / 5.0) * 0.3  # 90% to 60%
        else:  # Beyond 8 degrees (still exalted but weaker)
            proximity_factor = 0.6 - ((distance_from_exact - 8.0) / 22.0) * 0.2  # 60% to 40%
        
        proximity_factor = max(proximity_factor, 0.4)  # Minimum 40% strength
        
        # === CLASSICAL KP INTENSITY AMPLIFICATION ===
        # Base amplification: 40-80% increase in intensity based on proximity
        base_amplification_rate = 0.4 + (0.4 * proximity_factor)  # 40% to 80%
        
        # Calculate the amplification (maintaining the same direction as base score)
        intensity_amplification = base_score * base_amplification_rate
        
        # === ADDITIONAL ENHANCEMENT FACTORS ===
        additional_enhancement = 0.0
        
        # 1. Natural Authority Enhancement (planet-specific)
        if planet_name in ['Sun', 'Mars', 'Jupiter']:  # Natural authority planets
            authority_factor = 0.15 * proximity_factor  # Up to 15% additional
            additional_enhancement += abs(base_score) * authority_factor
        elif planet_name in ['Moon', 'Venus']:  # Natural grace/benefic planets
            grace_factor = 0.12 * proximity_factor  # Up to 12% additional
            additional_enhancement += abs(base_score) * grace_factor
        elif planet_name in ['Mercury', 'Saturn']:  # Natural intelligence/discipline planets
            wisdom_factor = 0.10 * proximity_factor  # Up to 10% additional
            additional_enhancement += abs(base_score) * wisdom_factor
        
        # 2. House Significator Enhancement
        planet_significators = self.get_significators(planet_name)
        planet_houses = [h for h, r in planet_significators]
        
        # Count strong house significators (Rule 1 and 2)
        strong_significators = [r for h, r in planet_significators if r in [1, 2]]
        if strong_significators:
            significator_factor = len(strong_significators) * 0.05 * proximity_factor  # 5% per strong significator
            significator_factor = min(significator_factor, 0.20)  # Cap at 20%
            additional_enhancement += abs(base_score) * significator_factor
        
        # === APPLY ENHANCEMENTS IN THE SAME DIRECTION AS BASE SCORE ===
        if base_score >= 0:
            # Positive base score: add positive enhancements
            total_enhancement = intensity_amplification + additional_enhancement
        else:
            # Negative base score: add negative enhancements (make more negative)
            total_enhancement = intensity_amplification - additional_enhancement
        
        # === MAXIMUM ENHANCEMENT CAP ===
        # Limit enhancement to prevent unrealistic values
        max_enhancement_magnitude = abs(base_score) * 1.5  # Maximum 150% amplification
        if abs(total_enhancement) > max_enhancement_magnitude:
            total_enhancement = max_enhancement_magnitude * (1 if total_enhancement >= 0 else -1)
        
        return total_enhancement

    def _calculate_base_score(self, planet_name: str, perspective: str = 'ascendant') -> float:
        """
        Calculates base score without debilitation corrections (to avoid recursion).
        
        Args:
            planet_name: Name of the planet
            perspective: Either 'ascendant' or 'descendant'
            
        Returns:
            float: Base score without corrections
        """
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
        Now includes Comment column explaining debilitation/exaltation effects.
        """
        df = self.engine.get_all_planets_df().copy()
        
        scores = {planet: self.calculate_planet_score(planet) for planet in df.index}
        significators_str = {
            planet: ", ".join(map(str, [s[0] for s in self.get_significators(planet)])) 
            for planet in df.index
        }
        
        # Generate comments explaining debilitation/exaltation effects
        comments = {}
        for planet in df.index:
            base_score = self._calculate_base_score(planet)
            final_score = scores[planet]
            
            comment_parts = []
            
            # Check for debilitation explanation
            debil_explanation = self._get_debilitation_explanation(planet, base_score, final_score)
            if debil_explanation:
                comment_parts.append(debil_explanation.strip())
            
            # Check for exaltation explanation  
            exalt_explanation = self._get_exaltation_explanation(planet, base_score, final_score)
            if exalt_explanation:
                comment_parts.append(exalt_explanation.strip())
            
            # Calculate impact on base score
            total_impact = final_score - base_score
            if abs(total_impact) >= 0.1:
                if total_impact > 0:
                    impact_desc = f"Score enhanced by +{total_impact:.3f}"
                else:
                    impact_desc = f"Score reduced by {total_impact:.3f}"
                comment_parts.append(impact_desc)
            
            # Combine all comment parts
            if comment_parts:
                comments[planet] = " | ".join(comment_parts)
            else:
                comments[planet] = "No special conditions"

        df['Score'] = df.index.map(scores)
        df['Significators'] = df.index.map(significators_str)
        df['Comment'] = df.index.map(comments)
        return df

    def analyze_muhurta_chart(self, scoring_method='proportional'):
        """
        Authentic KP Muhurta Chart Analysis following traditional methodology:
        1. Star Lord of 1st Cusp (Primary Indicator)
        2. Sub Lord of 1st Cusp (Modification Factor)
        3. Star Lord of 6th Cusp (Victory Indicator)
        4. Sub Lord of 6th Cusp (Victory Modification)
        5. Sub-Sub Lords for final confirmation
        6. Ruling Planets support
        
        Args:
            scoring_method: 'proportional' or 'binary'
        """
        analysis_parts = []
        
        # --- Quick Team Setup ---
        analysis_parts.append(f"🏏 **Asc** (Ascendant) vs **Desc** (Descendant)")
        analysis_parts.append("")
        
        # --- STAR LORD OF 1ST CUSP (Primary Indicator) ---
        cusp_1_star_lord = self.cusps.loc[1]['nl']
        c1sl_full = PlanetNameUtils.to_full_name(cusp_1_star_lord)
        c1sl_sigs = self.get_significators(c1sl_full) if c1sl_full in self.planets.index else []
        c1sl_score = self.calculate_planet_score(c1sl_full, 'ascendant') if c1sl_full in self.planets.index else 0.0
        
        analysis_parts.append("**🌟 STAR LORD OF 1ST CUSP (Primary Indicator):**")
        analysis_parts.append(f"• Planet: **{cusp_1_star_lord}** | Score: **{c1sl_score:+.2f}**")
        
        if c1sl_sigs:
            victory_houses = [h for h, r in c1sl_sigs if h in [1, 6, 10, 11]]
            defeat_houses = [h for h, r in c1sl_sigs if h in [4, 5, 7, 8, 9, 12]]
            
            sig_summary = []
            for house, rule in c1sl_sigs:
                sig_summary.append(f"H{house}(R{rule})")
            
            analysis_parts.append(f"• Houses: {', '.join(sig_summary)}")
            analysis_parts.append(f"• Victory Houses: {victory_houses} | Defeat Houses: {defeat_houses}")
            
            if c1sl_score > 0.3:
                c1sl_verdict = f"✅ **Strong Favor Asc**"
            elif c1sl_score > 0:
                c1sl_verdict = f"✅ **Favors Asc**"
            elif c1sl_score < -0.3:
                c1sl_verdict = f"❌ **Strong Favor Desc**"
            elif c1sl_score < 0:
                c1sl_verdict = f"❌ **Favors Desc**"
            else:
                c1sl_verdict = "⚖️ **Close Contest**"
            
            analysis_parts.append(f"• Result: {c1sl_verdict}")
        else:
            analysis_parts.append("• Result: ⚪ **Neutral**")
            c1sl_verdict = "NEUTRAL"
        
        analysis_parts.append("")
        
        # --- SUB LORD OF 1ST CUSP (Modification Factor) ---
        cusp_1_sub_lord = self.cusps.loc[1]['sl']
        c1subl_full = PlanetNameUtils.to_full_name(cusp_1_sub_lord)
        c1subl_sigs = self.get_significators(c1subl_full) if c1subl_full in self.planets.index else []
        c1subl_score = self.calculate_planet_score(c1subl_full, 'ascendant') if c1subl_full in self.planets.index else 0.0
        
        analysis_parts.append("**⚖️ SUB LORD OF 1ST CUSP (Modification Factor):**")
        analysis_parts.append(f"• Planet: **{cusp_1_sub_lord}** | Score: **{c1subl_score:+.2f}**")
        
        if c1subl_sigs:
            victory_houses = [h for h, r in c1subl_sigs if h in [1, 6, 10, 11]]
            defeat_houses = [h for h, r in c1subl_sigs if h in [4, 5, 7, 8, 9, 12]]
            
            sig_summary = []
            for house, rule in c1subl_sigs:
                sig_summary.append(f"H{house}(R{rule})")
            
            analysis_parts.append(f"• Houses: {', '.join(sig_summary)}")
            
            if c1subl_score > 0.3:
                c1subl_verdict = f"✅ **Strongly Supports Asc**"
            elif c1subl_score > 0:
                c1subl_verdict = f"✅ **Supports Asc**"
            elif c1subl_score < -0.3:
                c1subl_verdict = f"❌ **Strongly Opposes Asc**"
            elif c1subl_score < 0:
                c1subl_verdict = f"❌ **Opposes Asc**"
            else:
                c1subl_verdict = "⚖️ **Neutral Modification**"
            
            analysis_parts.append(f"• Result: {c1subl_verdict}")
        else:
            analysis_parts.append("• Result: ⚪ **Neutral**")
            c1subl_verdict = "NEUTRAL"
        
        analysis_parts.append("")
        
        # --- STAR LORD OF 6TH CUSP (Victory Indicator) ---
        cusp_6_star_lord = self.cusps.loc[6]['nl']
        c6sl_full = PlanetNameUtils.to_full_name(cusp_6_star_lord)
        c6sl_sigs = self.get_significators(c6sl_full) if c6sl_full in self.planets.index else []
        c6sl_score = self.calculate_planet_score(c6sl_full, 'ascendant') if c6sl_full in self.planets.index else 0.0
        
        analysis_parts.append("**🎯 STAR LORD OF 6TH CUSP (Victory Indicator):**")
        analysis_parts.append(f"• Planet: **{cusp_6_star_lord}** | Score: **{c6sl_score:+.2f}**")
        
        if c6sl_sigs:
            victory_houses = [h for h, r in c6sl_sigs if h in [1, 6, 10, 11]]
            defeat_houses = [h for h, r in c6sl_sigs if h in [4, 5, 7, 8, 9, 12]]
            
            sig_summary = []
            for house, rule in c6sl_sigs:
                sig_summary.append(f"H{house}(R{rule})")
            
            analysis_parts.append(f"• Houses: {', '.join(sig_summary)}")
            
            if c6sl_score > 0.3:
                c6sl_verdict = f"✅ **Strong Victory Asc**"
            elif c6sl_score > 0:
                c6sl_verdict = f"✅ **Victory Asc**"
            elif c6sl_score < -0.3:
                c6sl_verdict = f"❌ **Strong Victory Desc**"
            elif c6sl_score < 0:
                c6sl_verdict = f"❌ **Victory Desc**"
            else:
                c6sl_verdict = "⚖️ **Competitive Victory**"
            
            analysis_parts.append(f"• Result: {c6sl_verdict}")
        else:
            analysis_parts.append("• Result: ⚪ **Neutral**")
            c6sl_verdict = "NEUTRAL"
        
        analysis_parts.append("")
        
        # --- SUB LORD OF 6TH CUSP (Victory Modification) ---
        cusp_6_sub_lord = self.cusps.loc[6]['sl']
        c6subl_full = PlanetNameUtils.to_full_name(cusp_6_sub_lord)
        c6subl_sigs = self.get_significators(c6subl_full) if c6subl_full in self.planets.index else []
        c6subl_score = self.calculate_planet_score(c6subl_full, 'ascendant') if c6subl_full in self.planets.index else 0.0
        
        analysis_parts.append("**🏹 SUB LORD OF 6TH CUSP (Victory Modification):**")
        analysis_parts.append(f"• Planet: **{cusp_6_sub_lord}** | Score: **{c6subl_score:+.2f}**")
        
        if c6subl_sigs:
            victory_houses = [h for h, r in c6subl_sigs if h in [1, 6, 10, 11]]
            defeat_houses = [h for h, r in c6subl_sigs if h in [4, 5, 7, 8, 9, 12]]
            
            sig_summary = []
            for house, rule in c6subl_sigs:
                sig_summary.append(f"H{house}(R{rule})")
            
            analysis_parts.append(f"• Houses: {', '.join(sig_summary)}")
            
            if c6subl_score > 0.3:
                c6subl_verdict = f"✅ **Strongly Confirms Asc**"
            elif c6subl_score > 0:
                c6subl_verdict = f"✅ **Confirms Asc**"
            elif c6subl_score < -0.3:
                c6subl_verdict = f"❌ **Strongly Denies Asc**"
            elif c6subl_score < 0:
                c6subl_verdict = f"❌ **Denies Asc**"
            else:
                c6subl_verdict = "⚖️ **Mixed Signals**"
            
            analysis_parts.append(f"• Result: {c6subl_verdict}")
        else:
            analysis_parts.append("• Result: ⚪ **Neutral**")
            c6subl_verdict = "NEUTRAL"
        
        analysis_parts.append("")
        
        # --- SUB-SUB LORDS (Final Confirmation) ---
        cssl_1 = self.cusps.loc[1]['ssl']
        cssl_1_full = PlanetNameUtils.to_full_name(cssl_1)
        cssl_1_sigs = self.get_significators(cssl_1_full) if cssl_1_full in self.planets.index else []
        cssl_1_score = self.calculate_planet_score(cssl_1_full, 'ascendant') if cssl_1_full in self.planets.index else 0.0
        
        cssl_6 = self.cusps.loc[6]['ssl']
        cssl_6_full = PlanetNameUtils.to_full_name(cssl_6)
        cssl_6_sigs = self.get_significators(cssl_6_full) if cssl_6_full in self.planets.index else []
        cssl_6_score = self.calculate_planet_score(cssl_6_full, 'ascendant') if cssl_6_full in self.planets.index else 0.0
        
        analysis_parts.append("**📊 SUB-SUB LORDS (Final Confirmation):**")
        analysis_parts.append(f"• CSSL 1: **{cssl_1}** (Score: {cssl_1_score:+.2f}) | CSSL 6: **{cssl_6}** (Score: {cssl_6_score:+.2f})")
        
        # Combined CSSL verdict
        avg_cssl_score = (cssl_1_score + cssl_6_score) / 2
        if avg_cssl_score > 0.25:
            cssl_verdict = f"✅ **Final Confirmation Asc**"
        elif avg_cssl_score < -0.25:
            cssl_verdict = f"❌ **Final Confirmation Desc**"
        else:
            cssl_verdict = "⚖️ **Mixed Final Signals**"
        
        analysis_parts.append(f"• Combined Result: {cssl_verdict}")
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
        
        analysis_parts.append(f"• Support: Asc({rp_team_a_count}) vs Desc({rp_team_b_count})")
        
        if rp_team_a_count > rp_team_b_count:
            rp_verdict = f"✅ **Support Asc**"
        elif rp_team_b_count > rp_team_a_count:
            rp_verdict = f"❌ **Support Desc**"
        else:
            rp_verdict = "⚖️ **Mixed Support**"
        
        analysis_parts.append(f"• Result: {rp_verdict}")
        analysis_parts.append("")
        
        # --- Match Competitiveness Analysis ---
        analysis_parts.append("**⚖️ MATCH COMPETITIVENESS:**")
        
        # Calculate comprehensive score difference
        all_scores = [c1sl_score, c1subl_score, c6sl_score, c6subl_score, cssl_1_score, cssl_6_score]
        avg_all_scores = sum([abs(s) for s in all_scores]) / len(all_scores)
        score_variance = max(all_scores) - min(all_scores)
        
        analysis_parts.append(f"• Average Indicator Strength: **{avg_all_scores:.2f}**")
        analysis_parts.append(f"• Score Variance: **{score_variance:.2f}**")
        
        # Determine match competitiveness
        if avg_all_scores > 0.6:
            competitiveness = "🔥 **ONE-SIDED MATCH**"
            competitiveness_desc = "Clear dominance indicated"
        elif avg_all_scores > 0.3:
            competitiveness = "⚡ **MODERATE ADVANTAGE**"
            competitiveness_desc = "One team has clear edge"
        elif avg_all_scores > 0.15:
            competitiveness = "🎯 **CLOSE CONTEST**"
            competitiveness_desc = "Fairly balanced match"
        else:
            competitiveness = "🎲 **VERY CLOSE MATCH**"
            competitiveness_desc = "Extremely tight contest"
        
        analysis_parts.append(f"• Type: {competitiveness}")
        analysis_parts.append(f"• Description: {competitiveness_desc}")
        analysis_parts.append("")
        
        # --- SCORING METHOD SELECTION ---
        if scoring_method == 'proportional':
            synthesis_result = self._calculate_proportional_synthesis(
                c1sl_score, c1subl_score, c6sl_score, c6subl_score, avg_cssl_score,
                rp_team_a_count, rp_team_b_count, analysis_parts
            )
        else:  # binary method
            synthesis_result = self._calculate_binary_synthesis(
                c1sl_verdict, c1subl_verdict, c6sl_verdict, c6subl_verdict, cssl_verdict,
                rp_verdict, analysis_parts
            )
        
        return synthesis_result
    
    def _calculate_proportional_synthesis(self, c1sl_score, c1subl_score, c6sl_score, c6subl_score, avg_cssl_score, rp_team_a_count, rp_team_b_count, analysis_parts):
        """Calculate proportional weighted synthesis."""
        analysis_parts.append("**🏆 PROPORTIONAL WEIGHTED SYNTHESIS:**")
        
        # === SCORE-PROPORTIONAL WEIGHTED SYSTEM ===
        # Instead of binary weights, use actual score magnitudes with base weights
        
        base_weights = {
            'c1sl': 3.0,    # Star Lord of 1st Cusp (Highest Priority)
            'c1subl': 2.0,  # Sub Lord of 1st Cusp (High Priority)  
            'c6sl': 2.0,    # Star Lord of 6th Cusp (High Priority)
            'c6subl': 1.5,  # Sub Lord of 6th Cusp (Medium Priority)
            'cssl': 1.5,    # Combined Sub-Sub Lords (Medium Priority)
            'rp': 1.0       # Ruling Planets (Lower Priority)
        }
        
        # Calculate proportional weights based on actual scores
        team_a_weights = 0
        team_b_weights = 0
        total_possible_weights = sum(base_weights.values())
        
        # Star Lord of 1st Cusp - Proportional to score magnitude
        c1sl_magnitude = abs(c1sl_score)
        c1sl_weight = base_weights['c1sl'] * min(1.0, c1sl_magnitude)  # Cap at base weight
        if c1sl_score > 0:
            team_a_weights += c1sl_weight
        elif c1sl_score < 0:
            team_b_weights += c1sl_weight
        
        # Sub Lord of 1st Cusp - Proportional to score magnitude  
        c1subl_magnitude = abs(c1subl_score)
        c1subl_weight = base_weights['c1subl'] * min(1.0, c1subl_magnitude)
        if c1subl_score > 0:
            team_a_weights += c1subl_weight
        elif c1subl_score < 0:
            team_b_weights += c1subl_weight
        
        # Star Lord of 6th Cusp - Proportional to score magnitude
        c6sl_magnitude = abs(c6sl_score) 
        c6sl_weight = base_weights['c6sl'] * min(1.0, c6sl_magnitude)
        if c6sl_score > 0:
            team_a_weights += c6sl_weight
        elif c6sl_score < 0:
            team_b_weights += c6sl_weight
        
        # Sub Lord of 6th Cusp - Proportional to score magnitude
        c6subl_magnitude = abs(c6subl_score)
        c6subl_weight = base_weights['c6subl'] * min(1.0, c6subl_magnitude)
        if c6subl_score > 0:
            team_a_weights += c6subl_weight
        elif c6subl_score < 0:
            team_b_weights += c6subl_weight
        
        # Combined Sub-Sub Lords - Proportional to average score magnitude
        cssl_magnitude = abs(avg_cssl_score)
        cssl_weight = base_weights['cssl'] * min(1.0, cssl_magnitude)
        if avg_cssl_score > 0:
            team_a_weights += cssl_weight
        elif avg_cssl_score < 0:
            team_b_weights += cssl_weight
        
        # Ruling Planets - Binary for now (can be improved with individual planet scores)
        rp_weight = base_weights['rp']
        if rp_team_a_count > rp_team_b_count:
            team_a_weights += rp_weight
        elif rp_team_b_count > rp_team_a_count:
            team_b_weights += rp_weight
        
        # Calculate percentages based on actual weights received
        total_weights_assigned = team_a_weights + team_b_weights
        
        if total_weights_assigned > 0:
            team_a_percentage = (team_a_weights / total_weights_assigned) * 100
            team_b_percentage = (team_b_weights / total_weights_assigned) * 100
        else:
            team_a_percentage = 50.0
            team_b_percentage = 50.0
        
        analysis_parts.append(f"• Proportional Weights: Asc({team_a_weights:.2f}) vs Desc({team_b_weights:.2f})")
        analysis_parts.append(f"• Win Probability: Asc({team_a_percentage:.1f}%) vs Desc({team_b_percentage:.1f}%)")
        
        # Enhanced final verdict with confidence based on weight difference AND total strength
        weight_difference = abs(team_a_weights - team_b_weights)
        total_strength = team_a_weights + team_b_weights
        
        # Adjust confidence based on both difference and total strength
        if total_strength >= 7.0:  # Strong overall indications
            strength_modifier = "Strong"
        elif total_strength >= 4.0:  # Moderate overall indications
            strength_modifier = "Moderate" 
        else:  # Weak overall indications
            strength_modifier = "Weak"
        
        if team_a_weights > team_b_weights:
            if weight_difference >= 3.0:
                final_verdict = f"🏆 **Asc DECISIVELY FAVORED**"
                confidence = f"Very High ({strength_modifier} Signals)"
            elif weight_difference >= 1.5:
                final_verdict = f"🏆 **Asc STRONGLY FAVORED**"
                confidence = f"High ({strength_modifier} Signals)"
            elif weight_difference >= 0.5:
                final_verdict = f"🏆 **Asc FAVORED**"
                confidence = f"Medium ({strength_modifier} Signals)"
            else:
                final_verdict = f"🏆 **Asc SLIGHT EDGE**"
                confidence = f"Low ({strength_modifier} Signals)"
        elif team_b_weights > team_a_weights:
            if weight_difference >= 3.0:
                final_verdict = f"🏆 **Desc DECISIVELY FAVORED**"
                confidence = f"Very High ({strength_modifier} Signals)"
            elif weight_difference >= 1.5:
                final_verdict = f"🏆 **Desc STRONGLY FAVORED**"
                confidence = f"High ({strength_modifier} Signals)"
            elif weight_difference >= 0.5:
                final_verdict = f"🏆 **Desc FAVORED**"
                confidence = f"Medium ({strength_modifier} Signals)"
            else:
                final_verdict = f"🏆 **Desc SLIGHT EDGE**"
                confidence = f"Low ({strength_modifier} Signals)"
        else:
            final_verdict = "⚖️ **PERFECTLY BALANCED MATCH**"
            confidence = f"Uncertain ({strength_modifier} Signals)"
        
        analysis_parts.append(f"• Total Signal Strength: **{total_strength:.2f}** out of **{total_possible_weights:.1f}**")
        analysis_parts.append(f"• Confidence: **{confidence}**")
        analysis_parts.append(f"• {final_verdict}")
        
        return "\n".join(analysis_parts)
    
    def _calculate_binary_synthesis(self, c1sl_verdict, c1subl_verdict, c6sl_verdict, c6subl_verdict, cssl_verdict, rp_verdict, analysis_parts):
        """Calculate binary weighted synthesis."""
        analysis_parts.append("**🏆 BINARY WEIGHTED SYNTHESIS:**")
        
        # === TRADITIONAL BINARY SYSTEM ===
        # Each factor gets fixed points based on verdict
        
        weights = {
            'c1sl': 3,      # Star Lord of 1st Cusp (Highest Priority)
            'c1subl': 2,    # Sub Lord of 1st Cusp (High Priority)  
            'c6sl': 2,      # Star Lord of 6th Cusp (High Priority)
            'c6subl': 1.5,  # Sub Lord of 6th Cusp (Medium Priority)
            'cssl': 1.5,    # Combined Sub-Sub Lords (Medium Priority)
            'rp': 1         # Ruling Planets (Lower Priority)
        }
        
        team_a_points = 0
        team_b_points = 0
        total_possible_points = sum(weights.values())
        
        # Analyze each verdict and assign points
        verdicts = {
            'c1sl': c1sl_verdict,
            'c1subl': c1subl_verdict,
            'c6sl': c6sl_verdict,
            'c6subl': c6subl_verdict,
            'cssl': cssl_verdict,
            'rp': rp_verdict
        }
        
        for factor, verdict in verdicts.items():
            if 'Asc' in verdict or 'Supports Asc' in verdict or 'Confirms Asc' in verdict or 'Victory Asc' in verdict or 'Support Asc' in verdict or 'Confirmation Asc' in verdict:
                team_a_points += weights[factor]
            elif 'Desc' in verdict or 'Opposes Asc' in verdict or 'Denies Asc' in verdict or 'Victory Desc' in verdict or 'Support Desc' in verdict or 'Confirmation Desc' in verdict:
                team_b_points += weights[factor]
            # Neutral verdicts get no points
        
        # Calculate percentages
        total_points_assigned = team_a_points + team_b_points
        if total_points_assigned > 0:
            team_a_percentage = (team_a_points / total_points_assigned) * 100
            team_b_percentage = (team_b_points / total_points_assigned) * 100
        else:
            team_a_percentage = 50.0
            team_b_percentage = 50.0
        
        analysis_parts.append(f"• Binary Points: Asc({team_a_points}) vs Desc({team_b_points})")
        analysis_parts.append(f"• Win Probability: Asc({team_a_percentage:.1f}%) vs Desc({team_b_percentage:.1f}%)")
        
        # Final verdict based on point difference
        point_difference = abs(team_a_points - team_b_points)
        
        if team_a_points > team_b_points:
            if point_difference >= 5:
                final_verdict = f"🏆 **Asc DECISIVELY FAVORED**"
                confidence = "Very High"
            elif point_difference >= 3:
                final_verdict = f"🏆 **Asc STRONGLY FAVORED**"
                confidence = "High"
            elif point_difference >= 1:
                final_verdict = f"🏆 **Asc FAVORED**"
                confidence = "Medium"
            else:
                final_verdict = f"🏆 **Asc SLIGHT EDGE**"
                confidence = "Low"
        elif team_b_points > team_a_points:
            if point_difference >= 5:
                final_verdict = f"🏆 **Desc DECISIVELY FAVORED**"
                confidence = "Very High"
            elif point_difference >= 3:
                final_verdict = f"🏆 **Desc STRONGLY FAVORED**"
                confidence = "High"
            elif point_difference >= 1:
                final_verdict = f"🏆 **Desc FAVORED**"
                confidence = "Medium"
            else:
                final_verdict = f"🏆 **Desc SLIGHT EDGE**"
                confidence = "Low"
        else:
            final_verdict = "⚖️ **PERFECTLY BALANCED MATCH**"
            confidence = "Uncertain"
        
        analysis_parts.append(f"• Total Points Used: **{total_points_assigned}** out of **{total_possible_points}**")
        analysis_parts.append(f"• Confidence: **{confidence}**")
        analysis_parts.append(f"• {final_verdict}")
        
        return "\n".join(analysis_parts)

    def _generate_nl_sl_verdict_and_comment(self, timeline_row: pd.Series, perspective: str = 'ascendant') -> tuple:
        """
        Generates verdict and comment using only NL (Star Lord) and SL (Sub Lord) analysis:
        Star Lord promises → Sub Lord modifies → Combined verdict
        
        This method is used for aggregated timelines where SSL is not considered.
        
        Args:
            timeline_row: Row from timeline DataFrame with NL_Planet, SL_Planet (no SSL_Planet)
            perspective: Either 'ascendant' or 'descendant'
            
        Returns:
            tuple: (verdict, comment, combined_score)
        """
        nl_planet = timeline_row.get('NL_Planet')
        sl_planet = timeline_row.get('SL_Planet')
        
        # Handle missing data
        if pd.isna(nl_planet) or pd.isna(sl_planet):
            return "Neutral", "Insufficient planetary data for analysis", 0.0
        
        # Determine team names based on perspective
        team_name = "Asc" if perspective == 'ascendant' else "Desc"
        opponent_name = "Desc" if perspective == 'ascendant' else "Asc"
        
        # === LAYER 1: STAR LORD ANALYSIS (The Promise) ===
        nl_standardized = PlanetNameUtils.standardize_for_index(nl_planet)
        nl_score = self.calculate_planet_score(nl_standardized, perspective) if nl_standardized in self.planets.index else 0.0
        nl_significators = self.get_significators(nl_standardized) if nl_standardized in self.planets.index else []
        
        nl_victory_houses = [h for h, r in nl_significators if h in [1, 6, 10, 11]]
        nl_defeat_houses = [h for h, r in nl_significators if h in [4, 5, 7, 8, 9, 12]]
        
        if len(nl_victory_houses) > len(nl_defeat_houses) and nl_victory_houses:
            nl_promise = "VICTORY"
            nl_promise_desc = f"promises victory (V:{','.join(map(str, nl_victory_houses))})"
        elif len(nl_defeat_houses) > len(nl_victory_houses) and nl_defeat_houses:
            nl_promise = "DEFEAT" 
            nl_promise_desc = f"promises challenges (D:{','.join(map(str, nl_defeat_houses))})"
        elif nl_victory_houses and nl_defeat_houses:
            nl_promise = "MIXED"
            nl_promise_desc = f"mixed signals (V:{','.join(map(str, nl_victory_houses))} D:{','.join(map(str, nl_defeat_houses))})"
        else:
            nl_promise = "NEUTRAL"
            nl_promise_desc = "neutral period"
        
        # === LAYER 2: SUB LORD ANALYSIS (The Modifier) ===
        sl_standardized = PlanetNameUtils.standardize_for_index(sl_planet)
        sl_score = self.calculate_planet_score(sl_standardized, perspective) if sl_standardized in self.planets.index else 0.0
        sl_significators = self.get_significators(sl_standardized) if sl_standardized in self.planets.index else []
        
        sl_victory_houses = [h for h, r in sl_significators if h in [1, 6, 10, 11]]
        sl_defeat_houses = [h for h, r in sl_significators if h in [4, 5, 7, 8, 9, 12]]
        
        if len(sl_victory_houses) > len(sl_defeat_houses) and sl_victory_houses:
            sl_modification = "SUPPORTS"
            sl_mod_desc = f"supports victory (V:{','.join(map(str, sl_victory_houses))})"
        elif len(sl_defeat_houses) > len(sl_victory_houses) and sl_defeat_houses:
            sl_modification = "OPPOSES" 
            sl_mod_desc = f"supports challenges (D:{','.join(map(str, sl_defeat_houses))})"
        elif sl_victory_houses and sl_defeat_houses:
            sl_modification = "MIXED"
            sl_mod_desc = f"mixed modification (V:{','.join(map(str, sl_victory_houses))} D:{','.join(map(str, sl_defeat_houses))})"
        else:
            sl_modification = "NEUTRAL"
            sl_mod_desc = "neutral modification"
        
        # === COMBINED NL + SL SCORE CALCULATION ===
        # In KP, Star Lord has more weight than Sub Lord
        # Star Lord: 60% weight, Sub Lord: 40% weight
        combined_score = (nl_score * 0.6) + (sl_score * 0.4)
        
        # === GENERATE VERDICT BASED ON COMBINED SCORE ===
        if combined_score >= 0.25:
            verdict = f"Strong Advantage {team_name}"
            cricket_context = "Excellent period for building partnerships and dominating opponents"
            confidence_level = "HIGH"
        elif combined_score >= 0.12:
            verdict = f"Advantage {team_name}"
            cricket_context = "Good period for consolidation and steady progress"
            confidence_level = "MEDIUM"
        elif combined_score > 0.05:
            verdict = f"Balanced (Slight {team_name})"
            cricket_context = "Marginal advantage - gradual progress expected"
            confidence_level = "LOW"
        elif combined_score <= -0.25:
            verdict = f"Strong Advantage {opponent_name}"
            cricket_context = "Challenging period - wickets or pressure likely"
            confidence_level = "HIGH"
        elif combined_score <= -0.12:
            verdict = f"Advantage {opponent_name}"
            cricket_context = "Opposition builds pressure and momentum"
            confidence_level = "MEDIUM"
        elif combined_score < -0.05:
            verdict = f"Balanced (Slight {opponent_name})"
            cricket_context = "Slight opposition edge - careful play needed"
            confidence_level = "LOW"
        else:
            verdict = "Balanced Period"
            cricket_context = "Evenly matched phase with gradual developments"
            confidence_level = "LOW"
        
        # === GENERATE DETAILED COMMENT ===
        comment_parts = []
        
        # Add debilitation and exaltation explanations if applicable  
        nl_debil_explanation = self._get_debilitation_explanation(nl_standardized, 0.0, nl_score)
        nl_base_score = self._calculate_base_score(nl_standardized, perspective) if nl_standardized in self.planets.index else 0.0
        nl_exalt_explanation = self._get_exaltation_explanation(nl_standardized, nl_base_score, nl_score)
        
        sl_base_score = self._calculate_base_score(sl_standardized, perspective) if sl_standardized in self.planets.index else 0.0
        sl_debil_explanation = self._get_debilitation_explanation(sl_standardized, sl_base_score, sl_score)
        sl_exalt_explanation = self._get_exaltation_explanation(sl_standardized, sl_base_score, sl_score)
        
        # Combine explanations for each planet
        nl_combined_explanation = (nl_debil_explanation + nl_exalt_explanation).strip()
        sl_combined_explanation = (sl_debil_explanation + sl_exalt_explanation).strip()
        
        comment_parts.append(f"🌟 {nl_planet} {nl_promise_desc}{' ' + nl_combined_explanation if nl_combined_explanation else ''}")
        comment_parts.append(f"⚖️ {sl_planet} {sl_mod_desc}{' ' + sl_combined_explanation if sl_combined_explanation else ''}")
        comment_parts.append(f"🏏 {cricket_context}")
        comment_parts.append(f"📊 NL:{nl_score:+.2f} SL:{sl_score:+.2f} Combined:{combined_score:+.3f} | {confidence_level}")
        
        detailed_comment = " | ".join(comment_parts)
        
        return verdict, detailed_comment, combined_score

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
        
        # Calculate the hierarchical score for this period
        nl_standardized = PlanetNameUtils.standardize_for_index(nl_planet)
        sl_standardized = PlanetNameUtils.standardize_for_index(sl_planet)
        ssl_standardized = PlanetNameUtils.standardize_for_index(ssl_planet)
        
        nl_score = self.calculate_planet_score(nl_standardized, perspective) if nl_standardized in self.planets.index else 0.0
        sl_score = self.calculate_planet_score(sl_standardized, perspective) if sl_standardized in self.planets.index else 0.0
        ssl_base_score = self.calculate_planet_score(ssl_standardized, perspective) if ssl_standardized in self.planets.index else 0.0
        
        # Use hierarchical scoring instead of just SSL score
        ssl_score = self._calculate_ssl_hierarchical_score(ssl_base_score, sl_score, nl_score)
        
        # Determine team names based on perspective
        team_name = "Asc" if perspective == 'ascendant' else "Desc"
        opponent_name = "Desc" if perspective == 'ascendant' else "Asc"
        
        # === LAYER 1: STAR LORD ANALYSIS (The Promise) ===
        # nl_standardized already calculated above
        nl_significators = self.get_significators(nl_standardized) if nl_standardized in self.planets.index else []
        
        nl_victory_houses = [h for h, r in nl_significators if h in [1, 6, 10, 11]]
        nl_defeat_houses = [h for h, r in nl_significators if h in [4, 5, 7, 8, 9, 12]]
        
        if len(nl_victory_houses) > len(nl_defeat_houses) and nl_victory_houses:
            nl_promise = "VICTORY"
            nl_promise_desc = f"promises victory (V:{','.join(map(str, nl_victory_houses))} D:{','.join(map(str, nl_defeat_houses))})"
        elif len(nl_defeat_houses) > len(nl_victory_houses) and nl_defeat_houses:
            nl_promise = "DEFEAT" 
            nl_promise_desc = f"promises challenges (V:{','.join(map(str, nl_victory_houses))} D:{','.join(map(str, nl_defeat_houses))})"
        elif nl_victory_houses and nl_defeat_houses:
            nl_promise = "MIXED"
            nl_promise_desc = f"promises mixed results (V:{','.join(map(str, nl_victory_houses))} D:{','.join(map(str, nl_defeat_houses))})"
        else:
            nl_promise = "NEUTRAL"
            nl_promise_desc = "promises neutral period"
        
        # === LAYER 2: SUB LORD ANALYSIS (The Modifier) ===
        # sl_standardized already calculated above
        sl_significators = self.get_significators(sl_standardized) if sl_standardized in self.planets.index else []
        
        sl_victory_houses = [h for h, r in sl_significators if h in [1, 6, 10, 11]]
        sl_defeat_houses = [h for h, r in sl_significators if h in [4, 5, 7, 8, 9, 12]]
        
        # Determine how Sub Lord modifies the promise (simplified)
        if len(sl_victory_houses) > len(sl_defeat_houses):
            sl_modification = "SUPPORTS"
            sl_mod_desc = f"supports victory (H{','.join(map(str, sl_victory_houses))})"
        elif len(sl_defeat_houses) > len(sl_victory_houses):
            sl_modification = "OPPOSES" 
            sl_mod_desc = f"supports challenges (H{','.join(map(str, sl_defeat_houses))})"
        else:
            sl_modification = "NEUTRAL"
            sl_mod_desc = "maintains balance"
        
        # === LAYER 3: SUB-SUB LORD ANALYSIS (The Deliverer) ===
        # ssl_standardized already calculated above
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
        
        # === SIMPLIFIED VERDICT BASED ON ACTUAL SCORE ===
        # Trust the scoring system more than complex layer combinations
        
        if ssl_score >= 0.3:
            verdict = f"Strong Advantage {team_name}"
            cricket_context = "Excellent period for building partnerships and dominating opponents"
            confidence_level = "HIGH"
        elif ssl_score >= 0.15:
            verdict = f"Advantage {team_name}"
            cricket_context = "Good period for consolidation and steady progress"
            confidence_level = "MEDIUM"
        elif ssl_score > 0.05:
            verdict = f"Balanced (Slight {team_name})"
            cricket_context = "Marginal advantage - gradual progress expected"
            confidence_level = "LOW"
        elif ssl_score <= -0.3:
            verdict = f"Strong Advantage {opponent_name}"
            cricket_context = "Challenging period - wickets or pressure likely"
            confidence_level = "HIGH"
        elif ssl_score <= -0.15:
            verdict = f"Advantage {opponent_name}"
            cricket_context = "Opposition builds pressure and momentum"
            confidence_level = "MEDIUM"
        elif ssl_score < -0.05:
            verdict = f"Balanced (Slight {opponent_name})"
            cricket_context = "Slight opposition edge - careful play needed"
            confidence_level = "LOW"
        else:
            verdict = "Balanced Period"
            cricket_context = "Evenly matched phase with gradual developments"
            confidence_level = "LOW"
        
        # === GENERATE DETAILED COMMENT ===
        comment_parts = []
        
        # Add debilitation and exaltation explanations if applicable
        nl_base_score = self._calculate_base_score(nl_standardized, perspective) if nl_standardized in self.planets.index else 0.0
        nl_final_score = nl_score  # Use the calculated nl_score from hierarchical calculation
        nl_debil_explanation = self._get_debilitation_explanation(nl_standardized, nl_base_score, nl_final_score)
        nl_exalt_explanation = self._get_exaltation_explanation(nl_standardized, nl_base_score, nl_final_score)
        
        sl_base_score = self._calculate_base_score(sl_standardized, perspective) if sl_standardized in self.planets.index else 0.0
        sl_final_score = sl_score  # Use the calculated sl_score from hierarchical calculation
        sl_debil_explanation = self._get_debilitation_explanation(sl_standardized, sl_base_score, sl_final_score)
        sl_exalt_explanation = self._get_exaltation_explanation(sl_standardized, sl_base_score, sl_final_score)
        
        ssl_calculated_base_score = self._calculate_base_score(ssl_standardized, perspective) if ssl_standardized in self.planets.index else 0.0
        ssl_debil_explanation = self._get_debilitation_explanation(ssl_standardized, ssl_calculated_base_score, ssl_score)
        ssl_exalt_explanation = self._get_exaltation_explanation(ssl_standardized, ssl_calculated_base_score, ssl_score)
        
        # Combine explanations for each planet
        nl_combined_explanation = (nl_debil_explanation + nl_exalt_explanation).strip()
        sl_combined_explanation = (sl_debil_explanation + sl_exalt_explanation).strip()
        ssl_combined_explanation = (ssl_debil_explanation + ssl_exalt_explanation).strip()
        
        comment_parts.append(f"🌟 {nl_planet} {nl_promise_desc}{' ' + nl_combined_explanation if nl_combined_explanation else ''}")
        comment_parts.append(f"⚖️ {sl_planet} {sl_mod_desc}{' ' + sl_combined_explanation if sl_combined_explanation else ''}")
        comment_parts.append(f"🎯 {ssl_planet} {ssl_del_desc}{' ' + ssl_combined_explanation if ssl_combined_explanation else ''}")
        comment_parts.append(f"🏏 {cricket_context}")
        comment_parts.append(f"📊 Score: {ssl_score:+.3f} | Confidence: {confidence_level}")
        
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
        # Add score column using SSL-centric hierarchical method
        def calculate_hierarchical_score(row):
            nl_planet = row.get('NL_Planet')
            sl_planet = row.get('SL_Planet') 
            ssl_planet = row.get('SSL_Planet')
            
            if pd.isna(ssl_planet):
                return 0.0
                
            # === SSL-CENTRIC HIERARCHICAL SCORING ===
            # SSL is the primary delivery agent, but its expression is modified by NL and SL
            
            # 1. Get base SSL score (this is the core delivery potential)
            ssl_score = self.calculate_planet_score(ssl_planet, perspective)
            
            # 2. Calculate NL and SL influences as pathway modifiers
            nl_score = self.calculate_planet_score(nl_planet, perspective) if pd.notna(nl_planet) else 0.0
            sl_score = self.calculate_planet_score(sl_planet, perspective) if pd.notna(sl_planet) else 0.0
            
            # 3. Apply hierarchical modification method
            return self._calculate_ssl_hierarchical_score(ssl_score, sl_score, nl_score)
        
        timeline_df['Score'] = timeline_df.apply(calculate_hierarchical_score, axis=1)
        
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
        team_name = "Asc" if perspective == 'ascendant' else "Desc"
        
        if abs(avg_score) < 0.1:
            summary = f"The average score for this timeline is {avg_score:.2f}. The timeline appears balanced, suggesting a tightly contested match."
        elif avg_score > 0:
            summary = f"The average score for this timeline is {avg_score:.2f}. This indicates a general advantage for {team_name}."
        else:
            opponent_name = "Desc" if perspective == 'ascendant' else "Asc"
            summary = f"The average score for this timeline is {avg_score:.2f}. This indicates a general advantage for {opponent_name}."
        
        analysis = {
            "summary": summary,
            "favorable_planets": sorted(favorable_planets),
            "unfavorable_planets": sorted(unfavorable_planets)
        }
        
        return timeline_df, analysis

    def analyze_aggregated_timeline(self, timeline_df, perspective='ascendant'):
        """
        Analyzes an aggregated timeline DataFrame using only NL and SL (no SSL).
        
        Args:
            timeline_df: DataFrame with timeline data (NL_Planet, SL_Planet only)
            perspective: Either 'ascendant' or 'descendant'
            
        Returns:
            tuple: (scored_timeline_df, analysis_dict)
        """
        # Add score column based on combined NL + SL analysis
        verdict_comment_score_data = timeline_df.apply(
            lambda row: self._generate_nl_sl_verdict_and_comment(row, perspective), axis=1
        )
        
        timeline_df['Verdict'] = [vcs[0] for vcs in verdict_comment_score_data]
        timeline_df['Comment'] = [vcs[1] for vcs in verdict_comment_score_data]
        timeline_df['Score'] = [vcs[2] for vcs in verdict_comment_score_data]
        
        # Calculate average score
        avg_score = timeline_df['Score'].mean()
        
        # Identify favorable and unfavorable planets from NL and SL columns only
        favorable_planets = []
        unfavorable_planets = []
        
        # Get unique planets from NL and SL columns only (no SSL)
        planet_columns = ['NL_Planet', 'SL_Planet']
        unique_planets = pd.unique(timeline_df[planet_columns].values.ravel())
        unique_planets = [p for p in unique_planets if pd.notna(p)]
        
        for planet in unique_planets:
            score = self.calculate_planet_score(planet, perspective)
            if score > 0.05:
                favorable_planets.append(planet)
            elif score < -0.05:
                unfavorable_planets.append(planet)
        
        # Generate analysis summary with team-specific context
        team_name = "Asc" if perspective == 'ascendant' else "Desc"
        
        if abs(avg_score) < 0.08:
            summary = f"The aggregated NL+SL timeline shows an average score of {avg_score:.3f}. The timeline appears balanced at Star Lord and Sub Lord level, suggesting a tightly contested match."
        elif avg_score > 0:
            summary = f"The aggregated NL+SL timeline shows an average score of {avg_score:.3f}. This indicates a general advantage for {team_name} based on Star Lord and Sub Lord combinations."
        else:
            opponent_name = "Desc" if perspective == 'ascendant' else "Asc"
            summary = f"The aggregated NL+SL timeline shows an average score of {avg_score:.3f}. This indicates a general advantage for {opponent_name} based on Star Lord and Sub Lord combinations."
        
        analysis = {
            "summary": summary,
            "favorable_planets": sorted(favorable_planets),
            "unfavorable_planets": sorted(unfavorable_planets)
        }
        
        return timeline_df, analysis 

    def _get_debilitation_explanation(self, planet_name: str, base_score: float, final_score: float) -> str:
        """
        Generate explanatory text for debilitation using KP Agency Rule.
        
        Args:
            planet_name: Standardized planet name
            base_score: Original score before agency rule (not used for debilitated planets)
            final_score: Final score after agency rule application
            
        Returns:
            str: Explanation text for debilitation agency rule (empty if not debilitated)
        """
        if not self._is_planet_debilitated(planet_name):
            return ""
            
        if planet_name not in self.planets.index:
            return ""
            
        planet_info = self.planets.loc[planet_name]
        planet_sign = planet_info['sign']
        
        # === SIGN LORD MAPPING ===
        SIGN_LORD_MAPPING = {
            'Aries': 'Ma', 'Taurus': 'Ve', 'Gemini': 'Me', 'Cancer': 'Mo',
            'Leo': 'Su', 'Virgo': 'Me', 'Libra': 'Ve', 'Scorpio': 'Ma',
            'Sagittarius': 'Ju', 'Capricorn': 'Sa', 'Aquarius': 'Sa', 'Pisces': 'Ju'
        }
        
        sign_lord_short = SIGN_LORD_MAPPING.get(planet_sign)
        if not sign_lord_short:
            return ""
            
        sign_lord_full = PlanetNameUtils.to_full_name(sign_lord_short)
        if sign_lord_full not in self.planets.index:
            return ""
            
        # Get sign lord's score to determine direction
        sign_lord_score = self._calculate_base_score(sign_lord_full, 'ascendant')
        
        # Generate agency rule explanation
        explanation_parts = []
        
        # Basic agency rule explanation
        explanation_parts.append(f"🔗 {planet_name} debilitated in {planet_sign} (acts as {sign_lord_short} agent)")
        
        # Direction explanation
        if sign_lord_score > 0.1:
            explanation_parts.append(f"({sign_lord_short} positive → pro-Asc)")
        elif sign_lord_score < -0.1:
            explanation_parts.append(f"({sign_lord_short} negative → pro-Desc)")
        else:
            explanation_parts.append(f"({sign_lord_short} neutral)")
        
        # Neecha Bhanga if applicable
        if abs(sign_lord_score) > 0.03:
            explanation_parts.append("(Neecha Bhanga)")
        
        if explanation_parts:
            return " " + " ".join(explanation_parts)
        
        return ""

    def _get_exaltation_explanation(self, planet_name: str, base_score: float, final_score: float) -> str:
        """
        Generate explanatory text for exaltation intensity amplification to include in comments.
        
        Args:
            planet_name: Standardized planet name
            base_score: Original score before enhancements
            final_score: Final score after enhancements
            
        Returns:
            str: Explanation text for exaltation enhancements (empty if no enhancements)
        """
        if planet_name not in self.planets.index:
            return ""
            
        planet_info = self.planets.loc[planet_name]
        planet_sign = planet_info['sign']
        planet_longitude = planet_info['longitude']
        
        # Check for exaltation
        if planet_name not in self.EXALTATION_MAPPING:
            return ""
            
        exalt_sign, exalt_degree = self.EXALTATION_MAPPING[planet_name]
        is_exalted = planet_sign == exalt_sign
        
        if not is_exalted:
            return ""
            
        enhancement_amount = final_score - base_score
        
        if abs(enhancement_amount) < 0.05:
            return ""  # No significant enhancement
            
        # Calculate degree proximity for strength assessment
        degree_in_sign = planet_longitude % 30
        distance_from_exact = abs(degree_in_sign - exalt_degree)
        
        # Calculate amplification percentage
        amplification_percent = (abs(enhancement_amount) / abs(base_score)) * 100 if base_score != 0 else 0
        
        # Generate explanation based on classical KP principle
        explanation_parts = []
        
        # Determine intensity level
        if distance_from_exact <= 3.0:
            strength_desc = "Exact"
        elif distance_from_exact <= 8.0:
            strength_desc = "Strong"
        else:
            strength_desc = "Moderate"
        
        # Check if amplification preserves direction (correct KP behavior)
        if (base_score > 0 and enhancement_amount > 0) or (base_score < 0 and enhancement_amount < 0):
            # Correct amplification - same direction
            if amplification_percent >= 60:
                explanation_parts.append(f"🌟 {planet_name} exalted in {planet_sign} ({strength_desc} intensity)")
            elif amplification_percent >= 30:
                explanation_parts.append(f"✨ {planet_name} exalted in {planet_sign} ({strength_desc})")
            else:
                explanation_parts.append(f"🔸 {planet_name} exalted in {planet_sign}")
        else:
            # This should not happen with corrected logic, but just in case
            explanation_parts.append(f"⚠️ {planet_name} exalted in {planet_sign} (anomaly)")
        
        # Add degree proximity information
        if distance_from_exact <= 1.0:
            explanation_parts.append("(±1°)")
        elif distance_from_exact <= 3.0:
            explanation_parts.append("(±3°)")
        elif distance_from_exact <= 8.0:
            explanation_parts.append("(±8°)")
        
        # Add natural enhancement type
        if planet_name in ['Sun', 'Mars', 'Jupiter']:
            explanation_parts.append("(Authority)")
        elif planet_name in ['Moon', 'Venus']:
            explanation_parts.append("(Grace)")
        elif planet_name in ['Mercury', 'Saturn']:
            explanation_parts.append("(Wisdom)")
        
        # Add amplification info
        if amplification_percent >= 50:
            explanation_parts.append(f"(+{amplification_percent:.0f}%)")
        
        if explanation_parts:
            return " " + " ".join(explanation_parts)
        else:
            return ""

    def _calculate_ssl_hierarchical_score(self, ssl_score: float, sl_score: float, nl_score: float) -> float:
        """
        Calculate SSL-centric hierarchical score where SSL is the primary delivery agent
        but its expression is modified by the hierarchical pathway (NL → SL → SSL).
        
        This method addresses the concern that pure weighted averages can make SSL 
        influence negligible when NL has a strong score.
        
        Args:
            ssl_score: Score of the Sub-Sub Lord (primary delivery agent)
            sl_score: Score of the Sub Lord (immediate modifier)
            nl_score: Score of the Star Lord (general promise context)
            
        Returns:
            float: Final hierarchical score where SSL retains primary importance
        """
        
        # === METHOD 1: ENHANCED SSL WITH PATHWAY AMPLIFICATION ===
        # SSL retains 70-80% influence while pathway provides 20-30% modification
        
        # Step 1: SSL is the base delivery score (maintains primary importance)
        base_ssl_strength = abs(ssl_score)
        ssl_direction = 1 if ssl_score >= 0 else -1
        
        # Step 2: Calculate pathway harmony (how well NL and SL support SSL)
        pathway_harmony = self._calculate_pathway_harmony(nl_score, sl_score, ssl_score)
        
        # Step 3: Calculate pathway strength (average strength of the delivery path)
        pathway_strength = (abs(nl_score) + abs(sl_score)) / 2
        
        # Step 4: Apply modifications based on different scenarios
        
        if base_ssl_strength >= 0.5:
            # Strong SSL: Minimal pathway influence (SSL dominates)
            ssl_weight = 0.85
            pathway_weight = 0.15
            
        elif base_ssl_strength >= 0.3:
            # Moderate SSL: Balanced approach
            ssl_weight = 0.75
            pathway_weight = 0.25
            
        elif base_ssl_strength >= 0.1:
            # Weak SSL: Pathway can significantly modify
            ssl_weight = 0.65
            pathway_weight = 0.35
            
        else:
            # Very weak SSL: Maximum pathway influence
            ssl_weight = 0.60
            pathway_weight = 0.40
        
        # Step 5: Calculate pathway modification
        pathway_modification = pathway_harmony * pathway_strength * pathway_weight
        
        # Step 6: Calculate final score
        enhanced_ssl_score = (ssl_score * ssl_weight) + pathway_modification
        
        # Step 7: Apply pathway amplification/dampening for extreme cases
        if pathway_harmony > 0.5 and pathway_strength > 0.3:
            # Strong supportive pathway amplifies SSL
            amplification_factor = 1 + (pathway_harmony * 0.2)
            enhanced_ssl_score *= amplification_factor
            
        elif pathway_harmony < -0.5 and pathway_strength > 0.3:
            # Strong opposing pathway dampens SSL
            dampening_factor = 1 - (abs(pathway_harmony) * 0.15)
            enhanced_ssl_score *= dampening_factor
        
        # Step 8: Ensure SSL direction is preserved (crucial for authentic KP)
        # If SSL and final score have different directions, limit the modification
        final_direction = 1 if enhanced_ssl_score >= 0 else -1
        if ssl_direction != final_direction and base_ssl_strength > 0.2:
            # Strong SSL should not be completely overturned by pathway
            enhanced_ssl_score = ssl_score * 0.7  # Reduce but maintain direction
        
        return round(enhanced_ssl_score, 4)
    
    def _calculate_pathway_harmony(self, nl_score: float, sl_score: float, ssl_score: float) -> float:
        """
        Calculate how harmoniously the hierarchical pathway works together.
        Positive harmony means all levels support each other.
        Negative harmony means there are conflicts in the pathway.
        
        Returns:
            float: Harmony score between -1.0 (complete conflict) and +1.0 (perfect harmony)
        """
        
        # Determine directional alignment
        nl_direction = 1 if nl_score >= 0 else -1
        sl_direction = 1 if sl_score >= 0 else -1
        ssl_direction = 1 if ssl_score >= 0 else -1
        
        # Calculate directional harmony
        directional_scores = []
        
        # NL-SL alignment
        if nl_direction == sl_direction:
            directional_scores.append(min(abs(nl_score), abs(sl_score)))
        else:
            directional_scores.append(-min(abs(nl_score), abs(sl_score)))
        
        # SL-SSL alignment (more important as it's closer to delivery)
        if sl_direction == ssl_direction:
            directional_scores.append(min(abs(sl_score), abs(ssl_score)) * 1.5)  # 1.5x weight
        else:
            directional_scores.append(-min(abs(sl_score), abs(ssl_score)) * 1.5)
        
        # NL-SSL overall alignment
        if nl_direction == ssl_direction:
            directional_scores.append(min(abs(nl_score), abs(ssl_score)) * 0.8)  # 0.8x weight
        else:
            directional_scores.append(-min(abs(nl_score), abs(ssl_score)) * 0.8)
        
        # Calculate weighted harmony
        harmony_score = sum(directional_scores) / len(directional_scores)
        
        # Normalize to [-1, 1] range
        max_possible_harmony = max(abs(nl_score), abs(sl_score), abs(ssl_score)) * 1.5
        if max_possible_harmony > 0:
            normalized_harmony = harmony_score / max_possible_harmony
            return max(-1.0, min(1.0, normalized_harmony))
        
        return 0.0

 