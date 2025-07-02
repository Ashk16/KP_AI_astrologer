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

# === ENHANCED KP CUSP SUB LORD ANALYSIS ===
# Cusp importance weights for authentic KP methodology
CUSP_IMPORTANCE_WEIGHTS = {
    11: 1.0,  # Most critical - Fulfillment of desires/event outcome
    1: 0.8,   # Team/self strength and overall well-being  
    6: 0.8,   # Victory over opponents/competition
    7: 0.6,   # Opponent strength (reverse analysis)
    10: 0.5,  # Success, achievements, recognition
    4: 0.3,   # End of activity, change of field
    8: 0.3,   # Obstacles, sudden events
    12: 0.3,  # Losses, expenditure
}

# Victory and defeat house classifications for cusp analysis
VICTORY_HOUSES = [1, 6, 10, 11]
DEFEAT_HOUSES = [4, 5, 7, 8, 9, 12]
NEUTRAL_HOUSES = [2, 3]


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
        Enhanced KP Muhurta Chart Analysis with Authentic Cusp Sub Lord Integration:
        
        TRADITIONAL LAYERS:
        1. Star Lord of 1st Cusp (Primary Indicator)
        2. Sub Lord of 1st Cusp (Modification Factor)
        3. Star Lord of 6th Cusp (Victory Indicator)
        4. Sub Lord of 6th Cusp (Victory Modification)
        5. Sub-Sub Lords for confirmation
        6. Ruling Planets support
        
        ENHANCED AUTHENTIC KP LAYER:
        7. Cusp Sub Lord Analysis (Ultimate Deciding Factor)
        
        Args:
            scoring_method: 'proportional' or 'binary'
        """
        analysis_parts = []
        
        # --- Quick Team Setup ---
        analysis_parts.append(f"🏏 **Enhanced KP Analysis** - Asc vs Desc")
        analysis_parts.append("")
        
        # --- TRADITIONAL KP ANALYSIS (Existing System) ---
        analysis_parts.append("## 📊 **TRADITIONAL KP ANALYSIS**")
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
        analysis_parts.append(f"• 1st Cusp Sub-Sub Lord: **{cssl_1}** | Score: **{cssl_1_score:+.2f}**")
        analysis_parts.append(f"• 6th Cusp Sub-Sub Lord: **{cssl_6}** | Score: **{cssl_6_score:+.2f}**")
        
        avg_cssl_score = (cssl_1_score + cssl_6_score) / 2
        
        if avg_cssl_score > 0.2:
            cssl_verdict = "✅ **Confirmation Asc**"
        elif avg_cssl_score < -0.2:
            cssl_verdict = "❌ **Confirmation Desc**"
        else:
            cssl_verdict = "⚖️ **Mixed Confirmation**"
        
        analysis_parts.append(f"• Combined Result: {cssl_verdict} (Avg: {avg_cssl_score:+.2f})")
        analysis_parts.append("")
        
        # --- RULING PLANETS ---
        analysis_parts.append("**🔮 RULING PLANETS:**")
        
        asc_star_lord = self.cusps.loc[1]['nl']
        asc_sign_lord = self.cusps.loc[1]['sign_lord']
        moon_star_lord = self.planets.get('Moon', {}).get('nl', 'Unknown')
        moon_sign_lord = self.planets.get('Moon', {}).get('sign_lord', 'Unknown')
        day_lord = self._get_day_lord()
        
        ruling_planets = [asc_star_lord, asc_sign_lord, moon_star_lord, moon_sign_lord, day_lord]
        rp_counts = {planet: ruling_planets.count(planet) for planet in set(ruling_planets)}
        
        analysis_parts.append(f"• Ascendant Star Lord: **{asc_star_lord}**")
        analysis_parts.append(f"• Ascendant Sign Lord: **{asc_sign_lord}**")
        analysis_parts.append(f"• Moon Star Lord: **{moon_star_lord}**")
        analysis_parts.append(f"• Moon Sign Lord: **{moon_sign_lord}**")
        analysis_parts.append(f"• Day Lord: **{day_lord}**")
        
        # Calculate RP team scores
        rp_team_a_count = 0
        rp_team_b_count = 0
        
        for planet, count in rp_counts.items():
            planet_full = PlanetNameUtils.to_full_name(planet)
            if planet_full in self.planets.index:
                planet_score = self.calculate_planet_score(planet_full, 'ascendant')
                if planet_score > 0:
                    rp_team_a_count += count
                elif planet_score < 0:
                    rp_team_b_count += count
        
        if rp_team_a_count > rp_team_b_count:
            rp_verdict = "✅ **Support Asc**"
        elif rp_team_b_count > rp_team_a_count:
            rp_verdict = "❌ **Support Desc**"
        else:
            rp_verdict = "⚖️ **Neutral Support**"
        
        analysis_parts.append(f"• Result: {rp_verdict} (Asc:{rp_team_a_count} vs Desc:{rp_team_b_count})")
        analysis_parts.append("")
        
        # === NEW: AUTHENTIC KP CUSP SUB LORD ANALYSIS ===
        analysis_parts.append("## 🏆 **AUTHENTIC KP CUSP SUB LORD ANALYSIS**")
        analysis_parts.append("*The Ultimate Deciding Factor in Classical KP*")
        analysis_parts.append("")
        
        cusp_analysis = self.analyze_cusp_sub_lords('ascendant')
        
        # Key Decisor (11th Cusp)
        key_decisor = cusp_analysis['summary']['key_decisor']
        analysis_parts.append(f"**🎯 KEY DECISOR - {key_decisor['name']}:**")
        analysis_parts.append(f"• Sub Lord: **{key_decisor['sub_lord']}**")
        analysis_parts.append(f"• Impact: **{key_decisor['impact']}**")
        analysis_parts.append(f"• Reasoning: {key_decisor['reasoning']}")
        analysis_parts.append("")
        
        # Supporting and Opposing Cusps
        supporting_cusps = cusp_analysis['summary']['supportive_cusps']
        opposing_cusps = cusp_analysis['summary']['opposing_cusps']
        
        if supporting_cusps:
            analysis_parts.append("**✅ SUPPORTING CUSPS (Favor Ascendant):**")
            for cusp in supporting_cusps[:3]:  # Top 3
                cusp_name = self._get_cusp_name(cusp['cusp'])
                analysis_parts.append(f"• H{cusp['cusp']} ({cusp_name}): Sub Lord **{cusp['sub_lord']}** (Strength: {cusp['strength']:.2f})")
            analysis_parts.append("")
        
        if opposing_cusps:
            analysis_parts.append("**❌ OPPOSING CUSPS (Favor Descendant):**")
            for cusp in opposing_cusps[:3]:  # Top 3
                cusp_name = self._get_cusp_name(cusp['cusp'])
                analysis_parts.append(f"• H{cusp['cusp']} ({cusp_name}): Sub Lord **{cusp['sub_lord']}** (Strength: {cusp['strength']:.2f})")
            analysis_parts.append("")
        
        # Cusp Analysis Verdict
        final_verdict = cusp_analysis['final_verdict']
        analysis_parts.append("**🏅 CUSP SUB LORD VERDICT:**")
        analysis_parts.append(f"• Primary Decision: **{final_verdict['primary_verdict']}**")
        analysis_parts.append(f"• Overall Assessment: **{final_verdict['overall_verdict']}**")
        analysis_parts.append(f"• Confidence Level: **{cusp_analysis['confidence_level']}**")
        analysis_parts.append(f"• Final Score: **{final_verdict['final_score']:+.2f}**")
        analysis_parts.append(f"• Key Reason: {final_verdict['primary_reason']}")
        analysis_parts.append("")
        
        # === ENHANCED SYNTHESIS ===
        analysis_parts.append("## ⚖️ **ENHANCED WEIGHTED SYNTHESIS**")
        analysis_parts.append("*Integrating Traditional + Authentic KP Methods*")
        analysis_parts.append("")
        
        if scoring_method == 'proportional':
            final_analysis = self._calculate_enhanced_proportional_synthesis(
                c1sl_score, c1subl_score, c6sl_score, c6subl_score, avg_cssl_score, 
                rp_team_a_count, rp_team_b_count, cusp_analysis, analysis_parts)
        else:
            final_analysis = self._calculate_enhanced_binary_synthesis(
                c1sl_verdict, c1subl_verdict, c6sl_verdict, c6subl_verdict, cssl_verdict, 
                rp_verdict, cusp_analysis, analysis_parts)
        
        analysis_parts.append("")
        analysis_parts.append("---")
        analysis_parts.append("")
        
        return {
            'analysis': '\n'.join(analysis_parts),
            'verdict': final_analysis['verdict'],
            'confidence': final_analysis['confidence'],
            'asc_probability': final_analysis['asc_probability'],
            'desc_probability': final_analysis['desc_probability'],
            'cusp_analysis': cusp_analysis,
            'traditional_scores': {
                'c1sl_score': c1sl_score,
                'c1subl_score': c1subl_score,
                'c6sl_score': c6sl_score,
                'c6subl_score': c6subl_score,
                'avg_cssl_score': avg_cssl_score
            }
        }

    def _calculate_enhanced_proportional_synthesis(self, c1sl_score, c1subl_score, c6sl_score, c6subl_score, avg_cssl_score, rp_team_a_count, rp_team_b_count, cusp_analysis, analysis_parts):
        """Enhanced proportional synthesis integrating cusp sub lord analysis."""
        analysis_parts.append("**🏆 ENHANCED PROPORTIONAL SYNTHESIS:**")
        
        # === TRADITIONAL SCORING WEIGHTS ===
        traditional_weights = {
            'c1sl': 2.5,    # Star Lord of 1st Cusp
            'c1subl': 1.5,  # Sub Lord of 1st Cusp  
            'c6sl': 2.0,    # Star Lord of 6th Cusp
            'c6subl': 1.0,  # Sub Lord of 6th Cusp
            'cssl': 1.0,    # Combined Sub-Sub Lords
            'rp': 0.5       # Ruling Planets
        }
        
        # === CUSP SUB LORD WEIGHTS (ENHANCED) ===
        cusp_weights = {
            'cusp_analysis': 4.0  # Highest weight for authentic KP method
        }
        
        traditional_score = 0
        traditional_total_weight = 0
        
        # Calculate traditional score with proportional weights
        c1sl_magnitude = abs(c1sl_score)
        c1sl_weight = traditional_weights['c1sl'] * min(1.0, c1sl_magnitude)
        if c1sl_score > 0:
            traditional_score += c1sl_weight
        elif c1sl_score < 0:
            traditional_score -= c1sl_weight
        traditional_total_weight += c1sl_weight
        
        c1subl_magnitude = abs(c1subl_score)
        c1subl_weight = traditional_weights['c1subl'] * min(1.0, c1subl_magnitude)
        if c1subl_score > 0:
            traditional_score += c1subl_weight
        elif c1subl_score < 0:
            traditional_score -= c1subl_weight
        traditional_total_weight += c1subl_weight
        
        c6sl_magnitude = abs(c6sl_score)
        c6sl_weight = traditional_weights['c6sl'] * min(1.0, c6sl_magnitude)
        if c6sl_score > 0:
            traditional_score += c6sl_weight
        elif c6sl_score < 0:
            traditional_score -= c6sl_weight
        traditional_total_weight += c6sl_weight
        
        c6subl_magnitude = abs(c6subl_score)
        c6subl_weight = traditional_weights['c6subl'] * min(1.0, c6subl_magnitude)
        if c6subl_score > 0:
            traditional_score += c6subl_weight
        elif c6subl_score < 0:
            traditional_score -= c6subl_weight
        traditional_total_weight += c6subl_weight
        
        cssl_magnitude = abs(avg_cssl_score)
        cssl_weight = traditional_weights['cssl'] * min(1.0, cssl_magnitude)
        if avg_cssl_score > 0:
            traditional_score += cssl_weight
        elif avg_cssl_score < 0:
            traditional_score -= cssl_weight
        traditional_total_weight += cssl_weight
        
        # Ruling Planets
        rp_weight = traditional_weights['rp']
        if rp_team_a_count > rp_team_b_count:
            traditional_score += rp_weight
        elif rp_team_b_count > rp_team_a_count:
            traditional_score -= rp_weight
        traditional_total_weight += rp_weight
        
        # Normalize traditional score
        traditional_normalized = traditional_score / traditional_total_weight if traditional_total_weight > 0 else 0
        
        # === CUSP SUB LORD ANALYSIS SCORING ===
        cusp_final_score = cusp_analysis['final_verdict']['final_score']
        cusp_weight = cusp_weights['cusp_analysis'] * min(1.0, abs(cusp_final_score))
        
        # === COMBINED SCORING ===
        combined_score = (traditional_normalized * sum(traditional_weights.values()) + 
                         cusp_final_score * cusp_weight)
        total_possible_weight = sum(traditional_weights.values()) + cusp_weight
        
        final_score = combined_score / total_possible_weight if total_possible_weight > 0 else 0
        
        # Calculate final probabilities
        if final_score > 0:
            # Ascendant favored
            asc_advantage = min(abs(final_score) * 100, 40)  # Cap at 40% advantage
            asc_probability = 50 + asc_advantage
            desc_probability = 50 - asc_advantage
        else:
            # Descendant favored
            desc_advantage = min(abs(final_score) * 100, 40)  # Cap at 40% advantage
            desc_probability = 50 + desc_advantage
            asc_probability = 50 - desc_advantage
        
        # Determine confidence and verdict
        if abs(final_score) > 0.4:
            confidence = "Very High"
            verdict = "STRONG_ASCENDANT" if final_score > 0 else "STRONG_DESCENDANT"
        elif abs(final_score) > 0.2:
            confidence = "High"
            verdict = "MODERATE_ASCENDANT" if final_score > 0 else "MODERATE_DESCENDANT"
        elif abs(final_score) > 0.1:
            confidence = "Medium"
            verdict = "SLIGHT_ASCENDANT" if final_score > 0 else "SLIGHT_DESCENDANT"
        else:
            confidence = "Low"
            verdict = "VERY_CLOSE"
        
        # Enhanced analysis output
        analysis_parts.append(f"• **Traditional KP Score**: {traditional_normalized:+.3f} (Weight: {sum(traditional_weights.values()):.1f})")
        analysis_parts.append(f"• **Cusp Sub Lord Score**: {cusp_final_score:+.3f} (Weight: {cusp_weight:.1f})")
        analysis_parts.append(f"• **Combined Final Score**: {final_score:+.3f}")
        analysis_parts.append("")
        analysis_parts.append(f"• **Win Probability**: Asc({asc_probability:.1f}%) vs Desc({desc_probability:.1f}%)")
        analysis_parts.append(f"• **Verdict**: {verdict}")
        analysis_parts.append(f"• **Confidence**: {confidence}")
        
        # Key insight based on method agreement
        cusp_verdict = cusp_analysis['final_verdict']['primary_verdict']
        traditional_favors_asc = traditional_normalized > 0
        cusp_favors_asc = 'ASCENDANT' in cusp_verdict
        
        if traditional_favors_asc == cusp_favors_asc:
            agreement = "✅ **METHODS AGREE** - High reliability"
        else:
            agreement = "⚠️ **METHODS DISAGREE** - Cusp analysis decisive"
        
        analysis_parts.append(f"• **Method Agreement**: {agreement}")
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'asc_probability': asc_probability,
            'desc_probability': desc_probability,
            'final_score': final_score,
            'traditional_score': traditional_normalized,
            'cusp_score': cusp_final_score,
            'methods_agree': traditional_favors_asc == cusp_favors_asc
        }
    
    def _calculate_enhanced_binary_synthesis(self, c1sl_verdict, c1subl_verdict, c6sl_verdict, c6subl_verdict, cssl_verdict, rp_verdict, cusp_analysis, analysis_parts):
        """Enhanced binary synthesis integrating cusp sub lord analysis."""
        analysis_parts.append("**🏆 ENHANCED BINARY SYNTHESIS:**")
        
        # === TRADITIONAL BINARY WEIGHTS ===
        traditional_weights = {
            'c1sl': 2.5,    # Star Lord of 1st Cusp
            'c1subl': 1.5,  # Sub Lord of 1st Cusp  
            'c6sl': 2.0,    # Star Lord of 6th Cusp
            'c6subl': 1.0,  # Sub Lord of 6th Cusp
            'cssl': 1.0,    # Combined Sub-Sub Lords
            'rp': 0.5       # Ruling Planets
        }
        
        # === CUSP SUB LORD WEIGHTS ===
        cusp_weights = {
            'cusp_primary': 4.0,  # Primary cusp verdict (11th house)
            'cusp_overall': 2.0   # Overall cusp analysis
        }
        
        traditional_asc_points = 0
        traditional_desc_points = 0
        traditional_total_points = sum(traditional_weights.values())
        
        # Analyze traditional verdicts
        verdicts = {
            'c1sl': c1sl_verdict,
            'c1subl': c1subl_verdict,
            'c6sl': c6sl_verdict,
            'c6subl': c6subl_verdict,
            'cssl': cssl_verdict,
            'rp': rp_verdict
        }
        
        for factor, verdict in verdicts.items():
            if any(keyword in verdict for keyword in ['Asc', 'Support Asc', 'Confirmation Asc', 'Victory Asc']):
                traditional_asc_points += traditional_weights[factor]
            elif any(keyword in verdict for keyword in ['Desc', 'Support Desc', 'Confirmation Desc', 'Victory Desc']):
                traditional_desc_points += traditional_weights[factor]
        
        # === CUSP SUB LORD ANALYSIS ===
        cusp_final_verdict = cusp_analysis['final_verdict']
        cusp_primary_verdict = cusp_final_verdict['primary_verdict']
        cusp_overall_verdict = cusp_final_verdict['overall_verdict']
        
        cusp_asc_points = 0
        cusp_desc_points = 0
        cusp_total_points = sum(cusp_weights.values())
        
        # Primary cusp verdict (most important)
        if 'ASCENDANT' in cusp_primary_verdict:
            cusp_asc_points += cusp_weights['cusp_primary']
        elif 'DESCENDANT' in cusp_primary_verdict:
            cusp_desc_points += cusp_weights['cusp_primary']
        
        # Overall cusp verdict
        if 'ASCENDANT' in cusp_overall_verdict:
            cusp_asc_points += cusp_weights['cusp_overall']
        elif 'DESCENDANT' in cusp_overall_verdict:
            cusp_desc_points += cusp_weights['cusp_overall']
        
        # === COMBINED BINARY ANALYSIS ===
        total_asc_points = traditional_asc_points + cusp_asc_points
        total_desc_points = traditional_desc_points + cusp_desc_points
        total_possible_points = traditional_total_points + cusp_total_points
        
        # Calculate percentages
        total_assigned_points = total_asc_points + total_desc_points
        if total_assigned_points > 0:
            asc_percentage = (total_asc_points / total_assigned_points) * 100
            desc_percentage = (total_desc_points / total_assigned_points) * 100
        else:
            asc_percentage = 50.0
            desc_percentage = 50.0
        
        # Determine verdict and confidence
        point_difference = abs(total_asc_points - total_desc_points)
        
        if point_difference >= 4.0:
            confidence = "Very High"
            verdict = "STRONG_ASCENDANT" if total_asc_points > total_desc_points else "STRONG_DESCENDANT"
        elif point_difference >= 2.0:
            confidence = "High"
            verdict = "MODERATE_ASCENDANT" if total_asc_points > total_desc_points else "MODERATE_DESCENDANT"
        elif point_difference >= 1.0:
            confidence = "Medium"
            verdict = "SLIGHT_ASCENDANT" if total_asc_points > total_desc_points else "SLIGHT_DESCENDANT"
        else:
            confidence = "Low"
            verdict = "VERY_CLOSE"
        
        # Analysis output
        analysis_parts.append(f"• **Traditional Points**: Asc({traditional_asc_points:.1f}) vs Desc({traditional_desc_points:.1f})")
        analysis_parts.append(f"• **Cusp Sub Lord Points**: Asc({cusp_asc_points:.1f}) vs Desc({cusp_desc_points:.1f})")
        analysis_parts.append(f"• **Total Points**: Asc({total_asc_points:.1f}) vs Desc({total_desc_points:.1f})")
        analysis_parts.append("")
        analysis_parts.append(f"• **Win Probability**: Asc({asc_percentage:.1f}%) vs Desc({desc_percentage:.1f}%)")
        analysis_parts.append(f"• **Verdict**: {verdict}")
        analysis_parts.append(f"• **Confidence**: {confidence}")
        
        # Method agreement check
        traditional_favors_asc = traditional_asc_points > traditional_desc_points
        cusp_favors_asc = cusp_asc_points > cusp_desc_points
        
        if traditional_favors_asc == cusp_favors_asc:
            agreement = "✅ **METHODS AGREE** - High reliability"
        else:
            agreement = "⚠️ **METHODS DISAGREE** - Cusp analysis weighted higher"
        
        analysis_parts.append(f"• **Method Agreement**: {agreement}")
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'asc_probability': asc_percentage,
            'desc_probability': desc_percentage,
            'total_asc_points': total_asc_points,
            'total_desc_points': total_desc_points,
            'traditional_asc_points': traditional_asc_points,
            'traditional_desc_points': traditional_desc_points,
            'cusp_asc_points': cusp_asc_points,
            'cusp_desc_points': cusp_desc_points,
            'methods_agree': traditional_favors_asc == cusp_favors_asc
        }

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

    def analyze_cusp_sub_lords(self, perspective: str = 'ascendant') -> dict:
        """
        Authentic KP Cusp Sub Lord Analysis - The Ultimate Deciding Factor.
        
        This method implements classical KP methodology where cusp sub lords
        are the final arbiters of event outcomes, especially the 11th cusp sub lord
        which determines fulfillment of desires.
        
        Args:
            perspective: Either 'ascendant' or 'descendant'
            
        Returns:
            dict: Comprehensive cusp sub lord analysis with final verdict
        """
        analysis = {
            'cusp_analyses': {},
            'summary': {},
            'final_verdict': {},
            'confidence_level': 'Medium'
        }
        
        # Priority cusps for competition analysis (cricket matches)
        priority_cusps = [11, 1, 6, 7, 10, 4, 8, 12]
        
        total_weighted_score = 0
        total_possible_weight = 0
        detailed_breakdown = []
        
        for cusp_num in priority_cusps:
            cusp_info = self.cusps.loc[cusp_num]
            cusp_analysis = self._analyze_single_cusp_sub_lord(cusp_num, cusp_info, perspective)
            
            analysis['cusp_analyses'][cusp_num] = cusp_analysis
            
            # Calculate weighted contribution
            cusp_weight = CUSP_IMPORTANCE_WEIGHTS.get(cusp_num, 0.1)
            weighted_contribution = cusp_analysis['impact_score'] * cusp_weight
            
            total_weighted_score += weighted_contribution
            total_possible_weight += cusp_weight
            
            detailed_breakdown.append({
                'cusp': cusp_num,
                'sub_lord': cusp_analysis['sub_lord'],
                'impact': cusp_analysis['impact_direction'],
                'strength': cusp_analysis['impact_magnitude'],
                'weight': cusp_weight,
                'contribution': weighted_contribution
            })
        
        # Calculate final weighted average
        final_cusp_score = total_weighted_score / total_possible_weight if total_possible_weight > 0 else 0
        
        # Generate summary
        analysis['summary'] = {
            'total_weighted_score': final_cusp_score,
            'key_decisor': self._identify_key_decisor(analysis['cusp_analyses']),
            'supportive_cusps': self._identify_supportive_cusps(analysis['cusp_analyses'], perspective),
            'opposing_cusps': self._identify_opposing_cusps(analysis['cusp_analyses'], perspective),
            'detailed_breakdown': detailed_breakdown
        }
        
        # Generate final verdict based on cusp sub lord analysis
        analysis['final_verdict'] = self._generate_cusp_verdict(final_cusp_score, analysis['cusp_analyses'], perspective)
        
        # Determine confidence level
        analysis['confidence_level'] = self._calculate_cusp_confidence(analysis['cusp_analyses'])
        
        return analysis

    def _analyze_single_cusp_sub_lord(self, cusp_num: int, cusp_info: dict, perspective: str) -> dict:
        """
        Analyzes a single cusp's sub lord to determine its impact on the event.
        
        Args:
            cusp_num: Cusp number (1-12)
            cusp_info: Cusp details from cusps DataFrame
            perspective: 'ascendant' or 'descendant'
            
        Returns:
            dict: Analysis of the cusp sub lord's impact
        """
        sub_lord_short = cusp_info['sl']
        sub_lord_full = PlanetNameUtils.to_full_name(sub_lord_short)
        
        analysis = {
            'cusp_number': cusp_num,
            'cusp_name': self._get_cusp_name(cusp_num),
            'sub_lord': sub_lord_short,
            'sub_lord_full': sub_lord_full,
            'significators': [],
            'impact_direction': 'NEUTRAL',
            'impact_magnitude': 0.0,
            'impact_score': 0.0,
            'reasoning': ''
        }
        
        if sub_lord_full not in self.planets.index:
            analysis['reasoning'] = f"Sub lord {sub_lord_short} not found in planetary data"
            return analysis
        
        # Get sub lord's significators
        significators = self.get_significators(sub_lord_full)
        analysis['significators'] = significators
        
        if not significators:
            analysis['reasoning'] = f"Sub lord {sub_lord_short} has no significators"
            return analysis
        
        # Classify significators into victory/defeat/neutral houses
        victory_sigs = [h for h, r in significators if h in VICTORY_HOUSES]
        defeat_sigs = [h for h, r in significators if h in DEFEAT_HOUSES]
        neutral_sigs = [h for h, r in significators if h in NEUTRAL_HOUSES]
        
        # Calculate impact based on house significators
        victory_strength = len(victory_sigs) * 1.0
        defeat_strength = len(defeat_sigs) * 1.0
        neutral_strength = len(neutral_sigs) * 0.2
        
        # Special weighting for rule strength (Rule 1 is strongest)
        weighted_victory = sum([1.0 if r == 1 else 0.8 if r == 2 else 0.5 if r == 3 else 0.3 
                               for h, r in significators if h in VICTORY_HOUSES])
        weighted_defeat = sum([1.0 if r == 1 else 0.8 if r == 2 else 0.5 if r == 3 else 0.3 
                              for h, r in significators if h in DEFEAT_HOUSES])
        
        # Determine impact direction and magnitude
        net_impact = weighted_victory - weighted_defeat
        
        if net_impact > 0:
            analysis['impact_direction'] = 'FAVORS_ASCENDANT' if perspective == 'ascendant' else 'FAVORS_DESCENDANT'
            analysis['impact_magnitude'] = min(net_impact / 3.0, 1.0)  # Normalize to max 1.0
            analysis['impact_score'] = analysis['impact_magnitude']
        elif net_impact < 0:
            analysis['impact_direction'] = 'FAVORS_DESCENDANT' if perspective == 'ascendant' else 'FAVORS_ASCENDANT'
            analysis['impact_magnitude'] = min(abs(net_impact) / 3.0, 1.0)
            analysis['impact_score'] = -analysis['impact_magnitude']
        else:
            analysis['impact_direction'] = 'NEUTRAL'
            analysis['impact_magnitude'] = 0.0
            analysis['impact_score'] = 0.0
        
        # Generate reasoning
        analysis['reasoning'] = self._generate_cusp_reasoning(cusp_num, sub_lord_short, 
                                                            victory_sigs, defeat_sigs, 
                                                            analysis['impact_direction'])
        
        return analysis

    def _get_cusp_name(self, cusp_num: int) -> str:
        """Returns descriptive name for cusp number."""
        cusp_names = {
            1: "Ascendant/Self", 2: "Wealth/Resources", 3: "Courage/Effort", 
            4: "Endings/Comfort", 5: "Speculation/Intelligence", 6: "Victory/Competition",
            7: "Opponents/Partnership", 8: "Obstacles/Transformation", 9: "Fortune/Higher Knowledge",
            10: "Success/Achievement", 11: "Gains/Fulfillment", 12: "Losses/Expenditure"
        }
        return cusp_names.get(cusp_num, f"House {cusp_num}")

    def _generate_cusp_reasoning(self, cusp_num: int, sub_lord: str, victory_houses: list, 
                               defeat_houses: list, impact_direction: str) -> str:
        """Generates human-readable reasoning for cusp analysis."""
        cusp_name = self._get_cusp_name(cusp_num)
        
        if impact_direction == 'FAVORS_ASCENDANT':
            return f"{cusp_name} sub lord {sub_lord} signifies victory houses {victory_houses}, supporting ascendant team"
        elif impact_direction == 'FAVORS_DESCENDANT':
            return f"{cusp_name} sub lord {sub_lord} signifies defeat houses {defeat_houses}, supporting descendant team"
        else:
            return f"{cusp_name} sub lord {sub_lord} shows mixed or neutral signals"

    def _identify_key_decisor(self, cusp_analyses: dict) -> dict:
        """Identifies the most decisive cusp (usually 11th house)."""
        # 11th cusp is always the key decisor in KP
        eleventh_analysis = cusp_analyses.get(11, {})
        return {
            'cusp': 11,
            'name': 'Eleventh House (Fulfillment of Desires)',
            'sub_lord': eleventh_analysis.get('sub_lord', 'Unknown'),
            'impact': eleventh_analysis.get('impact_direction', 'NEUTRAL'),
            'reasoning': 'The 11th cusp sub lord is the ultimate deciding factor in KP for event outcomes'
        }

    def _identify_supportive_cusps(self, cusp_analyses: dict, perspective: str) -> list:
        """Identifies cusps supporting the given perspective."""
        target_direction = 'FAVORS_ASCENDANT' if perspective == 'ascendant' else 'FAVORS_DESCENDANT'
        
        supportive = []
        for cusp_num, analysis in cusp_analyses.items():
            if analysis.get('impact_direction') == target_direction:
                supportive.append({
                    'cusp': cusp_num,
                    'sub_lord': analysis.get('sub_lord'),
                    'strength': analysis.get('impact_magnitude', 0)
                })
        
        return sorted(supportive, key=lambda x: x['strength'], reverse=True)

    def _identify_opposing_cusps(self, cusp_analyses: dict, perspective: str) -> list:
        """Identifies cusps opposing the given perspective."""
        opposing_direction = 'FAVORS_DESCENDANT' if perspective == 'ascendant' else 'FAVORS_ASCENDANT'
        
        opposing = []
        for cusp_num, analysis in cusp_analyses.items():
            if analysis.get('impact_direction') == opposing_direction:
                opposing.append({
                    'cusp': cusp_num,
                    'sub_lord': analysis.get('sub_lord'),
                    'strength': analysis.get('impact_magnitude', 0)
                })
        
        return sorted(opposing, key=lambda x: x['strength'], reverse=True)

    def _generate_cusp_verdict(self, final_score: float, cusp_analyses: dict, perspective: str) -> dict:
        """Generates final verdict based on cusp sub lord analysis."""
        # Get 11th cusp analysis (most important)
        eleventh_cusp = cusp_analyses.get(11, {})
        eleventh_impact = eleventh_cusp.get('impact_direction', 'NEUTRAL')
        
        # Primary verdict based on 11th cusp
        if eleventh_impact == 'FAVORS_ASCENDANT':
            primary_verdict = "ASCENDANT_FAVORED"
            primary_reason = "11th cusp sub lord (fulfillment) supports ascendant team"
        elif eleventh_impact == 'FAVORS_DESCENDANT':
            primary_verdict = "DESCENDANT_FAVORED"
            primary_reason = "11th cusp sub lord (fulfillment) supports descendant team"
        else:
            primary_verdict = "COMPETITIVE"
            primary_reason = "11th cusp sub lord shows neutral or mixed signals"
        
        # Modify based on overall cusp score
        if final_score > 0.3:
            overall_verdict = "STRONG_ASCENDANT"
            confidence = "High"
        elif final_score > 0.1:
            overall_verdict = "MODERATE_ASCENDANT"
            confidence = "Medium"
        elif final_score < -0.3:
            overall_verdict = "STRONG_DESCENDANT"
            confidence = "High"
        elif final_score < -0.1:
            overall_verdict = "MODERATE_DESCENDANT"
            confidence = "Medium"
        else:
            overall_verdict = "CLOSE_CONTEST"
            confidence = "Low"
        
        return {
            'primary_verdict': primary_verdict,
            'overall_verdict': overall_verdict,
            'final_score': final_score,
            'confidence': confidence,
            'primary_reason': primary_reason,
            'eleventh_cusp_impact': eleventh_impact
        }

    def _calculate_cusp_confidence(self, cusp_analyses: dict) -> str:
        """Calculates confidence level based on cusp analysis consistency."""
        # Get 11th cusp strength (most important)
        eleventh_strength = cusp_analyses.get(11, {}).get('impact_magnitude', 0)
        
        # Count supporting vs opposing cusps
        ascendant_favoring = sum(1 for analysis in cusp_analyses.values() 
                               if analysis.get('impact_direction') == 'FAVORS_ASCENDANT')
        descendant_favoring = sum(1 for analysis in cusp_analyses.values() 
                                if analysis.get('impact_direction') == 'FAVORS_DESCENDANT')
        
        # High confidence if 11th cusp is strong and other cusps align
        if eleventh_strength > 0.7:
            return "Very High"
        elif eleventh_strength > 0.5:
            return "High" 
        elif eleventh_strength > 0.3:
            return "Medium"
        elif abs(ascendant_favoring - descendant_favoring) > 2:
            return "Medium"
        else:
            return "Low"

 