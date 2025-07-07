import pandas as pd
import os
import sys
from collections import defaultdict
import numpy as np

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

# === CORRECTED HOUSE WEIGHTS FOR CRICKET/SPORTS (KP SPORTS ASTROLOGY) ===
HOUSE_WEIGHTS = {
    # Victory Houses for Ascendant Team
    6: 1.0,   # Victory over opponents/enemies (strongest for winning)
    11: 0.9,  # Gains, profits, fulfillment of desires, success
    1: 0.5,   # Self, strength, health, overall well-being of the team
    10: 0.7,  # Achievements, public recognition, performance excellence
    3: 0.3,   # Courage, effort, initiative (supportive for victory)
    2: 0.2,   # Resources & Support. Team's form, available resources
    
    # Defeat Houses for Ascendant Team
    8: -1.0,  # Crisis & Sudden Events. Primary house for wickets, injuries, collapses
    12: -0.9, # Total Loss & Self-Undoing. Ultimate failure, errors
    5: -0.8,  # Opponent's Gains. 11th from 7th, opponent achieving goals
    7: -0.6,  # Opponent's Strength. Opponent is formidable and playing well
    9: -0.3,  # Opponent's Courage & Efforts. 3rd from 7th, opponent's initiative
    4: -0.2   # End of Play / Home Advantage. Complex house signifying end of activity
}

# === AUTHENTIC VEDIC PLANETARY FRIENDSHIP/ENMITY TABLE ===
# Based on classical Vedic astrology texts (Brihat Parasara Hora Shastra)
PLANETARY_RELATIONSHIPS = {
    'Sun': {
        'friends': ['Moon', 'Mars', 'Jupiter'],
        'neutrals': ['Mercury'],
        'enemies': ['Venus', 'Saturn', 'Rahu', 'Ketu']
    },
    'Moon': {
        'friends': ['Sun', 'Mercury'],
        'neutrals': ['Mars', 'Jupiter', 'Venus', 'Saturn'],
        'enemies': ['Rahu', 'Ketu']
    },
    'Mars': {
        'friends': ['Sun', 'Moon', 'Jupiter'],
        'neutrals': ['Venus', 'Saturn'],
        'enemies': ['Mercury', 'Rahu', 'Ketu']
    },
    'Mercury': {
        'friends': ['Sun', 'Venus'],
        'neutrals': ['Mars', 'Jupiter', 'Saturn'],
        'enemies': ['Moon', 'Rahu', 'Ketu']
    },
    'Jupiter': {
        'friends': ['Sun', 'Moon', 'Mars'],
        'neutrals': ['Saturn'],
        'enemies': ['Mercury', 'Venus', 'Rahu', 'Ketu']
    },
    'Venus': {
        'friends': ['Mercury', 'Saturn'],
        'neutrals': ['Mars', 'Jupiter'],
        'enemies': ['Sun', 'Moon', 'Rahu', 'Ketu']
    },
    'Saturn': {
        'friends': ['Mercury', 'Venus'],
        'neutrals': ['Jupiter'],
        'enemies': ['Sun', 'Moon', 'Mars', 'Rahu', 'Ketu']
    },
    'Rahu': {
        'friends': ['Mercury', 'Venus', 'Saturn'],  # Generally considered friendly to Mercury, Venus, Saturn
        'neutrals': [],
        'enemies': ['Sun', 'Moon', 'Mars', 'Jupiter', 'Ketu']
    },
    'Ketu': {
        'friends': ['Mars', 'Jupiter'],  # Generally considered friendly to Mars, Jupiter
        'neutrals': [],
        'enemies': ['Sun', 'Moon', 'Mercury', 'Venus', 'Saturn', 'Rahu']
    }
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
    
    # === ENHANCED EXALTATION/DEBILITATION MAPPING (Classical KP with Degrees) ===
    EXALTATION_MAPPING = {
        'Sun': ('Aries', 10.0),      # Sun exalted in Aries at 10°
        'Moon': ('Taurus', 3.0),     # Moon exalted in Taurus at 3°
        'Mars': ('Capricorn', 28.0), # Mars exalted in Capricorn at 28°
        'Mercury': ('Virgo', 15.0),  # Mercury exalted in Virgo at 15°
        'Jupiter': ('Cancer', 5.0),  # Jupiter exalted in Cancer at 5°
        'Venus': ('Pisces', 27.0),   # Venus exalted in Pisces at 27°
        'Saturn': ('Libra', 20.0),   # Saturn exalted in Libra at 20°
        'Rahu': ('Taurus', 20.0),    # Rahu exalted in Taurus at 20° (as per KP tradition)
        'Ketu': ('Scorpio', 15.0),   # Ketu exalted in Scorpio at 15° (as per KP tradition)
    }
    
    DEBILITATION_MAPPING = {
        'Sun': ('Libra', 10.0),      # Sun debilitated in Libra at 10°
        'Moon': ('Scorpio', 3.0),    # Moon debilitated in Scorpio at 3°
        'Mars': ('Cancer', 28.0),    # Mars debilitated in Cancer at 28°
        'Mercury': ('Pisces', 15.0), # Mercury debilitated in Pisces at 15°
        'Jupiter': ('Capricorn', 5.0), # Jupiter debilitated in Capricorn at 5°
        'Venus': ('Virgo', 27.0),    # Venus debilitated in Virgo at 27°
        'Saturn': ('Aries', 20.0),   # Saturn debilitated in Aries at 20°
        'Rahu': ('Scorpio', 20.0),   # Rahu debilitated in Scorpio at 20° (opposite to exaltation)
        'Ketu': ('Taurus', 15.0),    # Ketu debilitated in Taurus at 15° (opposite to exaltation)
    }
    
    # === DYNAMIC TIMELINE PLANETARY STRENGTH SYSTEM ===
    NATURAL_STRENGTH_MULTIPLIERS = {
        'exaltation': 3.0,
        'own_sign': 2.0,
        'friend_sign': 1.5,
        'neutral_sign': 1.0,
        'enemy_sign': 0.7,
        'debilitation': 0.3
    }
    
    # === REFINED POSITIONAL STRENGTH SYSTEM (Sports/Cricket Context) ===
    POSITIONAL_STRENGTH_MULTIPLIERS = {
        'kendra': 2.0,      # Angular houses: 1, 4, 7, 10 (foundations)
        'trinal_pure': 1.9, # Pure trinal houses: 5, 9 (1 is kendra, gets kendra strength)
        'upachaya_pure': 1.6, # Pure upachaya houses: 3, 11 (6,10 overlap with other categories)
        'upachaya_mixed': 1.4, # Mixed upachaya: 6 (also dusthana)
        'dusthana_mild': 0.9,  # Mild dusthana: 6 (opponent victory, but still action-oriented)
        'dusthana_strong': 0.7, # Strong dusthana: 8, 12 (obstacles, losses)
        'maraka_mild': 0.8,    # Mild maraka: 2 (resources, secondary)
        'maraka_strong': 0.6,  # Strong maraka: 7 (direct opposition)
        'sukha': 1.2,          # Sukha house: 4 (comfort, foundation)
        'neutral': 1.0         # Other houses
    }
    
    # === ENHANCED TEMPORAL STRENGTH WEIGHTS (Cricket/Sports Context) ===
    TEMPORAL_STRENGTH_WEIGHTS = {
        'nl': 1.0,        # Nakshatra Lord (Star Lord) - primary timing influence
        'sl': 0.75,       # Sub Lord - decisive secondary influence (more critical in KP)
        'ssl': 0.5        # Sub-Sub Lord - fine-tuning influence (reduced from 0.6)
    }

    # === CUSP IMPORTANCE WEIGHTS (Classical KP) ===

    # Default house weights (KP sports astrology)
    DEFAULT_HOUSE_WEIGHTS = {
        1: 0.5, 2: 0.2, 3: 0.3, 4: -0.2, 5: -0.8, 6: 1.0, 7: -0.6, 8: -1.0, 9: -0.3, 10: 0.7, 11: 0.9, 12: -0.9
    }
    # Default timeline weights
    DEFAULT_TIMELINE_WEIGHTS = {
        'ssl_timeline': {'NL': 0.1, 'SL': 0.2, 'SSL': 0.7},
        'asc_timeline': {'NL': 0.2, 'SL': 0.3, 'Context': 0.5}
    }

    def __init__(self, engine: KPEngine, team_a_name: str, team_b_name: str, house_weights=None, timeline_weights=None):
        self.engine = engine
        self.team_a = team_a_name
        self.team_b = team_b_name
        self.planets = engine.get_all_planets_df()
        self.cusps = engine.get_all_cusps_df()
        # Use provided weights or defaults
        self.house_weights = house_weights if house_weights is not None else self.DEFAULT_HOUSE_WEIGHTS.copy()
        self.timeline_weights = timeline_weights if timeline_weights is not None else self.DEFAULT_TIMELINE_WEIGHTS.copy()
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
        # Use dynamic house weights if set, else default
        return self.house_weights.get(house_num, self.DEFAULT_HOUSE_WEIGHTS.get(house_num, 0.0))

    def calculate_planet_score(self, planet_name: str, perspective: str = 'ascendant') -> float:
        """
        Calculates a score for a planet based on its weighted significations.
        Implements authentic KP principles where all planets (except Rahu/Ketu) give their own results
        with intensity modifications based on dignity.
        
        Args:
            planet_name: Name of the planet (can be either short or full name)
            perspective: Either 'ascendant' or 'descendant'
            
        Returns:
            float: The calculated score from the given perspective (with intensity modifications)
        """
        from functools import reduce
        
        # Standardize planet name for processing
        planet_name = PlanetNameUtils.standardize_for_index(planet_name)

        # === RAHU/KETU SPECIAL HANDLING ===
        if planet_name in ['Rahu', 'Ketu']:
            return self._calculate_rahu_ketu_score(planet_name, perspective)

        # === CALCULATE BASE SCORE FROM SIGNIFICATORS ===
        base_score = self._calculate_base_score(planet_name, perspective)
        
        # === CALCULATE INTENSITY MODIFIER ===
        # Get individual strength factors
        natural_strength = self._get_planetary_natural_strength(planet_name)
        positional_strength = self._get_planetary_positional_strength(planet_name)
        significator_relevance = self._get_significator_relevance(planet_name, perspective)
        
        # Calculate final intensity modifier
        # We use geometric mean to prevent extreme diminishing when multiple reducing factors are present
        intensity_factors = [natural_strength, positional_strength, significator_relevance]
        intensity_modifier = pow(reduce(lambda x, y: x * y, intensity_factors), 1/len(intensity_factors))
        
        # Ensure the modifier stays within reasonable bounds (0.3 to 3.0)
        intensity_modifier = max(0.3, min(3.0, intensity_modifier))
        
        # Apply intensity modifier to base score
        final_score = base_score * intensity_modifier
        
        return final_score

    def _is_planet_debilitated(self, planet_name: str) -> bool:
        """
        Check if a planet is debilitated in its current sign using enhanced mapping.
        
        Args:
            planet_name: Standardized planet name
            
        Returns:
            bool: True if planet is debilitated, False otherwise
        """
        if planet_name not in self.planets.index:
            return False
            
        if planet_name not in self.DEBILITATION_MAPPING:
            return False
            
        planet_info = self.planets.loc[planet_name]
        planet_sign = planet_info['sign']
        debil_sign, _ = self.DEBILITATION_MAPPING[planet_name]
        
        return planet_sign == debil_sign
    
    # Function removed as its functionality is now integrated into _get_planetary_natural_strength

    # Function removed as debilitation is now handled through intensity modifiers in calculate_planet_score

    # Function removed as exaltation is now handled through intensity modifiers in calculate_planet_score

    def _calculate_base_score(self, planet_name: str, perspective: str = 'ascendant') -> float:
        """
        Calculates base score using only the top 2 most impactful significator rules.
        This eliminates the normalization bias where planets with more significators 
        get artificially lower scores.
        
        Args:
            planet_name: Name of the planet
            perspective: Either 'ascendant' or 'descendant'
            
        Returns:
            float: Base score from top 2 significator rules
        """
        planet_name = PlanetNameUtils.standardize_for_index(planet_name)
        
        significations = self.get_significators(planet_name)
        if not significations:
            return 0.0

        # Calculate impact for each significator rule
        rule_impacts = []
        for house, rule in significations:
            rule_weight = SIGNIFICATOR_RULE_WEIGHTS.get(rule, 0)
            house_weight = self._get_house_weight(house, perspective)
            weighted_score = rule_weight * house_weight
            impact = abs(weighted_score)  # Absolute impact for sorting
            rule_impacts.append((impact, weighted_score, house, rule))
        
        # Sort by impact (descending) and take top 2
        rule_impacts.sort(key=lambda x: x[0], reverse=True)
        top_2_rules = rule_impacts[:2]
        
        # Calculate score from top 2 rules only
        total_score = sum(weighted_score for _, weighted_score, _, _ in top_2_rules)
        
        return total_score

    # === RAHU/KETU AGENCY IMPLEMENTATION ===
    
    def _find_rahu_ketu_agent(self, node_name: str) -> str:
        """
        Find Rahu/Ketu's primary agent using KP hierarchy
        Priority: Conjunction > Nakshatra Lord > Sign Lord
        """
        if node_name not in self.planets.index:
            return 'Sun'  # Fallback
        
        planet_info = self.planets.loc[node_name]
        
        # Priority 1: Check for conjunction (within 6 degrees)
        conjunct_planet = self._find_conjunction_partner(node_name, orb=6.0)
        if conjunct_planet:
            return conjunct_planet
        
        # Priority 2: Nakshatra Lord (Star Lord)
        nakshatra_lord = planet_info['nl']
        nakshatra_lord_full = PlanetNameUtils.to_full_name(nakshatra_lord)
        if nakshatra_lord_full and nakshatra_lord_full not in ['Rahu', 'Ketu']:
            return nakshatra_lord_full
        
        # Priority 3: Sign Lord (Dispositor)
        sign_lord = self._get_sign_lord_for_planet(node_name)
        return sign_lord
    
    def _find_conjunction_partner(self, planet_name: str, orb: float = 6.0) -> str:
        """
        Find if planet is conjunct with another planet within orb
        """
        if planet_name not in self.planets.index:
            return None
        
        planet_longitude = self.planets.loc[planet_name]['longitude']
        
        for other_planet in self.planets.index:
            if other_planet == planet_name or other_planet in ['Rahu', 'Ketu']:
                continue
            
            other_longitude = self.planets.loc[other_planet]['longitude']
            
            # Calculate angular distance
            diff = abs(planet_longitude - other_longitude)
            if diff > 180:
                diff = 360 - diff
            
            if diff <= orb:
                return other_planet
        
        return None
    
    def _get_sign_lord_for_planet(self, planet_name: str) -> str:
        """
        Get sign lord for a planet
        """
        if planet_name not in self.planets.index:
            return 'Sun'  # Fallback
        
        planet_sign = self.planets.loc[planet_name]['sign']
        
        # Sign lord mapping
        SIGN_LORD_MAPPING = {
            'Aries': 'Mars', 'Taurus': 'Venus', 'Gemini': 'Mercury', 'Cancer': 'Moon',
            'Leo': 'Sun', 'Virgo': 'Mercury', 'Libra': 'Venus', 'Scorpio': 'Mars',
            'Sagittarius': 'Jupiter', 'Capricorn': 'Saturn', 'Aquarius': 'Saturn', 'Pisces': 'Jupiter'
        }
        
        return SIGN_LORD_MAPPING.get(planet_sign, 'Sun')
    
    def _get_top_2_score_from_significators(self, significators: list, perspective: str) -> float:
        """
        Helper method to get top 2 score from a list of significators
        """
        if not significators:
            return 0.0
        
        # Calculate impact for each significator rule
        rule_impacts = []
        for house, rule in significators:
            rule_weight = SIGNIFICATOR_RULE_WEIGHTS.get(rule, 0)
            house_weight = self._get_house_weight(house, perspective)
            weighted_score = rule_weight * house_weight
            impact = abs(weighted_score)
            rule_impacts.append((impact, weighted_score))
        
        # Sort by impact and take top 2
        rule_impacts.sort(key=lambda x: x[0], reverse=True)
        top_2_rules = rule_impacts[:2]
        
        return sum(weighted_score for _, weighted_score in top_2_rules)
    
    def _calculate_rahu_ketu_score(self, node_name: str, perspective: str = 'ascendant') -> float:
        """
        Calculate Rahu/Ketu score using KP principles:
        1. Primary influence (80%) comes from agent's significations
        2. Secondary influence (20%) comes from own significations
        3. Amplification/Detachment applies to agent's score first
        4. Own significations cannot reverse the agent's indication (positive/negative)
        """
        
        # Get own significators and calculate score
        own_significators = self.get_significators(node_name)
        own_score = self._get_top_2_score_from_significators(own_significators, perspective)
        
        # Find primary agent and get agent significators
        primary_agent = self._find_rahu_ketu_agent(node_name)
        agent_significators = self.get_significators(primary_agent)
        agent_score = self._get_top_2_score_from_significators(agent_significators, perspective)
        
        # Apply Rahu/Ketu modifiers to agent score first
        if node_name == 'Rahu':
            modified_agent_score = agent_score * 1.15  # Rahu amplifies agent's results
        else:  # Ketu
            modified_agent_score = agent_score * 0.85  # Ketu detaches from agent's results
            
        # Calculate weighted combination
        weighted_agent = modified_agent_score * 0.8
        weighted_own = own_score * 0.2
        
        # Prevent sign reversal - own significations cannot flip agent's indication
        if modified_agent_score > 0:
            # If agent is positive, final score must stay positive
            final_score = max(0.1, weighted_agent + weighted_own)
        elif modified_agent_score < 0:
            # If agent is negative, final score must stay negative
            final_score = min(-0.1, weighted_agent + weighted_own)
        else:
            # If agent is exactly 0, allow own significations to determine direction
            final_score = weighted_agent + weighted_own
        
        return final_score
        
    def _get_previous_sign(self, planet_name: str) -> str:
        """
        Get the previous sign of a planet from stored history.
        Returns None if no history exists.
        """
        # TODO: Implement sign history tracking
        # For now, return None to indicate no history
        return None
    
    def _get_rahu_ketu_significators_bucketed(self, node_name: str) -> dict:
        """
        Get Rahu/Ketu significators bucketed into Own and Agent categories
        """
        own_significators = self.get_significators(node_name)
        primary_agent = self._find_rahu_ketu_agent(node_name)
        agent_significators = self.get_significators(primary_agent)
        
        return {
            'own': own_significators,
            'agent': agent_significators,
            'agent_name': primary_agent
        }

    # === END RAHU/KETU AGENCY IMPLEMENTATION ===

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
        
        # Generate significators string with special handling for Rahu/Ketu
        significators_str = {}
        for planet in df.index:
            if planet in ['Rahu', 'Ketu']:
                # Show bucketed significators for Rahu/Ketu
                bucket_info = self._get_rahu_ketu_significators_bucketed(planet)
                own_houses = [str(s[0]) for s in bucket_info['own']]
                agent_houses = [str(s[0]) for s in bucket_info['agent']]
                significators_str[planet] = f"Own: {', '.join(own_houses)} | Agent({bucket_info['agent_name']}): {', '.join(agent_houses)}"
            else:
                # Regular significators for other planets
                significators_str[planet] = ", ".join(map(str, [s[0] for s in self.get_significators(planet)]))
        
        
        # Generate comments explaining debilitation/exaltation effects and Rahu/Ketu agency
        comments = {}
        for planet in df.index:
            comment_parts = []
            
            if planet in ['Rahu', 'Ketu']:
                # Special comment for Rahu/Ketu agency
                bucket_info = self._get_rahu_ketu_significators_bucketed(planet)
                agent_name = bucket_info['agent_name']
                agent_score = self._get_top_2_score_from_significators(bucket_info['agent'], 'ascendant')
                own_score = self._get_top_2_score_from_significators(bucket_info['own'], 'ascendant')
                
                agent_influence = "positive" if agent_score > 0 else "negative"
                own_influence = "pro-Asc" if own_score > 0 else "pro-Desc"
                
                comment = f"{planet} acts as {agent_name} agent ({agent_name} {agent_influence} → {own_influence})"
                
                if planet == 'Rahu':
                    comment += " (Amplified +15%)"
                else:
                    comment += " (Detached -15%)"
                
                comments[planet] = comment
            else:
                # Regular comments for other planets
                base_score = self._calculate_base_score(planet)
                final_score = scores[planet]
                
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
        KP Muhurta Chart Analysis using CSSL Methodology.
        Focuses on 1st, 6th, and 7th cusps with proper KP interpretation.
        """
        analysis_parts = []
        
        # --- Header ---
        analysis_parts.append(f"🏏 **KP CSSL Analysis** - {self.team_a} vs {self.team_b}")
        analysis_parts.append("")
        
        # Get CSSL for key cusps and their scores
        cssl_1 = self.cusps.loc[1]['ssl']
        cssl_6 = self.cusps.loc[6]['ssl']
        cssl_7 = self.cusps.loc[7]['ssl']
        
        cssl_1_full = PlanetNameUtils.to_full_name(cssl_1)
        cssl_6_full = PlanetNameUtils.to_full_name(cssl_6)
        cssl_7_full = PlanetNameUtils.to_full_name(cssl_7)
        
        cssl_1_score = self.calculate_planet_score(cssl_1_full, 'ascendant') if cssl_1_full in self.planets.index else 0.0
        cssl_6_score = self.calculate_planet_score(cssl_6_full, 'ascendant') if cssl_6_full in self.planets.index else 0.0
        cssl_7_score = self.calculate_planet_score(cssl_7_full, 'ascendant') if cssl_7_full in self.planets.index else 0.0
        
        # === REFINED HOUSE CATEGORIZATION ===
        VICTORY_HOUSES = [1, 6, 10, 11]  # Primary victory houses
        DEFEAT_HOUSES = [8, 12]          # Primary defeat houses
        SUPPORT_HOUSES = [2, 3, 9]       # Secondary support
        CHALLENGE_HOUSES = [4, 5, 7]     # Secondary challenges
        
        def categorize_houses(significators):
            houses = [h for h, r in significators]
            return {
                'victory': [h for h in houses if h in VICTORY_HOUSES],
                'defeat': [h for h in houses if h in DEFEAT_HOUSES],
                'support': [h for h in houses if h in SUPPORT_HOUSES],
                'challenge': [h for h in houses if h in CHALLENGE_HOUSES]
            }
        
        # === ANALYSIS OF EACH CUSP ===
        def analyze_cusp_strength(score, houses_dict, cusp_type):
            """Generate verdict based on KP principles"""
            victory_count = len(houses_dict['victory'])
            defeat_count = len(houses_dict['defeat'])
            
            if score > 0.5:
                return "✅ Strong Positive Indication"
            elif score > 0.2:
                return "✅ Moderate Positive"
            elif score < -0.5:
                return "❌ Strong Negative Indication"
            elif score < -0.2:
                return "❌ Moderate Negative"
            else:
                if victory_count > defeat_count:
                    return "⚖️ Slightly Favorable"
                elif defeat_count > victory_count:
                    return "⚖️ Slightly Challenging"
                return "⚖️ Neutral"
        
        # Analyze each cusp
        cusps_analysis = {}
        for cusp_num, cssl, score in [(1, cssl_1_full, cssl_1_score), 
                                    (6, cssl_6_full, cssl_6_score),
                                    (7, cssl_7_full, cssl_7_score)]:
            
            sigs = self.get_significators(cssl) if cssl in self.planets.index else []
            houses_dict = categorize_houses(sigs)
            
            cusps_analysis[cusp_num] = {
                'score': score,
                'houses': houses_dict,
                'verdict': analyze_cusp_strength(score, houses_dict, 
                    'opponent' if cusp_num == 7 else 'self')
            }
        
        # === PROBABILITY CALCULATION ===
        # More decisive probability spread based on scores
        weighted_score = (
            cusps_analysis[1]['score'] * 0.3 +  # 1st cusp
            cusps_analysis[6]['score'] * 0.5 +  # 6th cusp
            -cusps_analysis[7]['score'] * 0.2   # 7th cusp (inverted)
        )
        
        # Convert score to probability with wider spread
        base_prob = 50 + (weighted_score * 25)  # Multiplier increased from 10 to 25
        win_prob = max(min(base_prob, 85), 15)  # Allow 15-85% range instead of previous narrow range
        
        # === GENERATE DETAILED ANALYSIS ===
        for cusp_num in [1, 6, 7]:
            analysis = cusps_analysis[cusp_num]
            cusp_name = {1: "Self/Team Strength", 6: "Victory/Defeat", 7: "Opponent Strength"}[cusp_num]
            
            analysis_parts.append(f"\n**{'🏠' if cusp_num == 1 else '🏆' if cusp_num == 6 else '🎯'} {cusp_num}st CUSP CSSL ({cusp_name}):**")
            analysis_parts.append(f"• Sub-Sub Lord: {PlanetNameUtils.to_short_name(locals()[f'cssl_{cusp_num}'])} | Score: {analysis['score']:+.2f}")
            
            houses = analysis['houses']
            if houses:
                analysis_parts.append(f"• Victory Houses: {houses['victory']} | Defeat Houses: {houses['defeat']}")
                if houses['support'] or houses['challenge']:
                    analysis_parts.append(f"• Support Houses: {houses['support']} | Challenge Houses: {houses['challenge']}")
            analysis_parts.append(f"• Assessment: {analysis['verdict']}")
        
        # === FINAL VERDICT ===
        analysis_parts.append("\n📊 **FINAL VERDICT**")
        team_a_prob = win_prob
        team_b_prob = 100 - win_prob
        
        predicted_winner = self.team_a if team_a_prob > team_b_prob else self.team_b
        win_margin = abs(team_a_prob - team_b_prob)
        
        confidence = "High" if win_margin > 25 else "Medium" if win_margin > 15 else "Low"
        contest_type = "Decisive Victory" if win_margin > 25 else "Clear Advantage" if win_margin > 15 else "Close Contest"
        
        analysis_parts.append(f"• Predicted Winner: **{predicted_winner}**")
        analysis_parts.append(f"• Win Probability: {max(team_a_prob, team_b_prob):.1f}%")
        analysis_parts.append(f"• Contest Type: 🎯 {contest_type}")
        analysis_parts.append(f"• Confidence Level: {confidence}")
        
        analysis_parts.append(f"\n📈 **DETAILED PROBABILITIES:**")
        analysis_parts.append(f"• {self.team_a}: {team_a_prob:.1f}%")
        analysis_parts.append(f"• {self.team_b}: {team_b_prob:.1f}%")
        
        return "\n".join(analysis_parts)

    def _calculate_enhanced_proportional_synthesis_DEPRECATED(self, c1sl_score, c1subl_score, c6sl_score, c6subl_score, avg_cssl_score, rp_team_a_count, rp_team_b_count, cusp_analysis, analysis_parts):
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
    
    def _calculate_enhanced_binary_synthesis_DEPRECATED(self, c1sl_verdict, c1subl_verdict, c6sl_verdict, c6subl_verdict, cssl_verdict, rp_verdict, cusp_analysis, analysis_parts):
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
        elif combined_score >= 0.3:  # Updated from 0.12 for new top-2 scale
            verdict = f"Advantage {team_name}"
            cricket_context = "Good period for consolidation and steady progress"
            confidence_level = "MEDIUM"
        elif combined_score > 0.125:  # Updated from 0.05 for new top-2 scale
            verdict = f"Balanced (Slight {team_name})"
            cricket_context = "Marginal advantage - gradual progress expected"
            confidence_level = "LOW"
        elif combined_score <= -0.625:  # Updated from -0.25 for new top-2 scale
            verdict = f"Strong Advantage {opponent_name}"
            cricket_context = "Challenging period - wickets or pressure likely"
            confidence_level = "HIGH"
        elif combined_score <= -0.3:  # Updated from -0.12 for new top-2 scale
            verdict = f"Advantage {opponent_name}"
            cricket_context = "Opposition builds pressure and momentum"
            confidence_level = "MEDIUM"
        elif combined_score < -0.125:  # Updated from -0.05 for new top-2 scale
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
        
        if ssl_score >= 0.75:  # Updated from 0.3 for new top-2 scale
            verdict = f"Strong Advantage {team_name}"
            cricket_context = "Excellent period for building partnerships and dominating opponents"
            confidence_level = "HIGH"
        elif ssl_score >= 0.375:  # Updated from 0.15 for new top-2 scale
            verdict = f"Advantage {team_name}"
            cricket_context = "Good period for consolidation and steady progress"
            confidence_level = "MEDIUM"
        elif ssl_score > 0.125:  # Updated from 0.05 for new top-2 scale
            verdict = f"Balanced (Slight {team_name})"
            cricket_context = "Marginal advantage - gradual progress expected"
            confidence_level = "LOW"
        elif ssl_score <= -0.75:  # Updated from -0.3 for new top-2 scale
            verdict = f"Strong Advantage {opponent_name}"
            cricket_context = "Challenging period - wickets or pressure likely"
            confidence_level = "HIGH"
        elif ssl_score <= -0.375:  # Updated from -0.15 for new top-2 scale
            verdict = f"Advantage {opponent_name}"
            cricket_context = "Opposition builds pressure and momentum"
            confidence_level = "MEDIUM"
        elif ssl_score < -0.125:  # Updated from -0.05 for new top-2 scale
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
        Enhanced timeline analysis using dynamic layer influence methodology.
        For Moon timeline (with NL, SL, SSL), use the user-configurable weighted method.
        """
        if timeline_df.empty:
            return timeline_df, {"summary": "No timeline data available", "favorable_planets": [], "unfavorable_planets": []}
        
        enhanced_rows = []
        
        for _, row in timeline_df.iterrows():
            nl_planet = row.get('NL_Planet')
            sl_planet = row.get('SL_Planet') 
            ssl_planet = row.get('SSL_Planet')
            if pd.isna(ssl_planet):
                ssl_planet = None
            # Calculate individual layer scores
            nl_score = self.calculate_planet_score(nl_planet, perspective) if pd.notna(nl_planet) else 0.0
            sl_score = self.calculate_planet_score(sl_planet, perspective) if pd.notna(sl_planet) else 0.0
            ssl_score = self.calculate_planet_score(ssl_planet, perspective) if ssl_planet else 0.0
            # --- NEW: Use weighted method for Moon timeline ---
            weighted_score = self._calculate_ssl_hierarchical_score(ssl_score, sl_score, nl_score)
            final_score = weighted_score  # No convergence factor for new method
            # Optionally, keep dynamic influences for display
            dynamics = self._calculate_dynamic_layer_influences(nl_planet, sl_planet, ssl_planet, perspective)
            # Generate enhanced verdict and comment using the new weighted score
            verdict, comment = self._generate_dynamic_verdict_and_comment(
                row, perspective, dynamics, nl_score, sl_score, ssl_score, final_score
            )
            enhanced_row = row.to_dict()
            enhanced_row.update({
                'Score': final_score,
                'Verdict': verdict,
                'Comment': comment,
                'NL_Influence': dynamics['nl_influence'],
                'SL_Influence': dynamics['sl_influence'],
                'SSL_Influence': dynamics['ssl_influence'],
                'Event_Magnitude': dynamics['event_magnitude'],
                'Convergence_Factor': dynamics['convergence_factor']
            })
            enhanced_rows.append(enhanced_row)
        enhanced_df = pd.DataFrame(enhanced_rows)
        # Calculate analysis summary
        avg_score = enhanced_df['Score'].mean()
        avg_magnitude = enhanced_df['Event_Magnitude'].mean()
        # Identify favorable and unfavorable planets
        favorable_planets, unfavorable_planets = self._identify_timeline_planets(enhanced_df, perspective)
        # Generate team-specific summary
        team_name = "Asc" if perspective == 'ascendant' else "Desc"
        if abs(avg_score) < 0.1:
            summary = f"The enhanced dynamic timeline shows an average score of {avg_score:.3f} with event magnitude {avg_magnitude:.2f}. The timeline appears balanced with moderate intensity periods, suggesting a tightly contested match."
        elif avg_score > 0:
            summary = f"The enhanced dynamic timeline shows an average score of {avg_score:.3f} with event magnitude {avg_magnitude:.2f}. This indicates a general advantage for {team_name} with varying intensity periods based on planetary layer dynamics."
        else:
            opponent_name = "Desc" if perspective == 'ascendant' else "Asc"
            summary = f"The enhanced dynamic timeline shows an average score of {avg_score:.3f} with event magnitude {avg_magnitude:.2f}. This indicates a general advantage for {opponent_name} with varying intensity periods based on planetary layer dynamics."
        analysis = {
            "summary": summary,
            "favorable_planets": sorted(favorable_planets),
            "unfavorable_planets": sorted(unfavorable_planets),
            "average_magnitude": avg_magnitude,
            "high_intensity_periods": len(enhanced_df[enhanced_df['Event_Magnitude'] > 3.0]),
            "method": "dynamic_layer_analysis"
        }
        return enhanced_df, analysis

    def analyze_aggregated_timeline(self, timeline_df, perspective='ascendant'):
        """
        Enhanced aggregated timeline analysis using dynamic NL+SL methodology.
        For context score, use weighted sum of planet scores by their SSL proportion in the timeline.
        """
        if timeline_df.empty:
            return timeline_df, {"summary": "No timeline data available", "favorable_planets": [], "unfavorable_planets": []}
        enhanced_rows = []
        # --- Calculate SSL proportions for context score ---
        ssl_counts = {}
        total_periods = len(timeline_df)
        for _, row in timeline_df.iterrows():
            ssl = row.get('SSL_Planet')
            if pd.notna(ssl):
                ssl = str(ssl)
                ssl_counts[ssl] = ssl_counts.get(ssl, 0) + 1
        ssl_proportion = {p: (ssl_counts.get(p, 0) / total_periods) for p in self.planets.index}
        # Precompute planet scores
        planet_score = {p: self.calculate_planet_score(p, perspective) for p in self.planets.index}
        for _, row in timeline_df.iterrows():
            nl_planet = row.get('NL_Planet')
            sl_planet = row.get('SL_Planet')
            # Calculate individual layer scores
            nl_score = self.calculate_planet_score(nl_planet, perspective) if pd.notna(nl_planet) else 0.0
            sl_score = self.calculate_planet_score(sl_planet, perspective) if pd.notna(sl_planet) else 0.0
            # --- NEW: Weighted context score ---
            context_score = sum(planet_score[p] * ssl_proportion[p] for p in self.planets.index)
            # Use user-configurable weights for final score
            final_score = self._calculate_asc_timeline_score(nl_score, sl_score, context_score)
            # Optionally, keep dynamic influences for display
            dynamics = self._calculate_dynamic_layer_influences(nl_planet, sl_planet, None, perspective)
            # Generate verdict and comment
            verdict, comment = self._generate_dynamic_nl_sl_verdict_and_comment(
                row, perspective, dynamics, nl_score, sl_score, final_score
            )
            enhanced_row = row.to_dict()
            enhanced_row.update({
                'Score': final_score,
                'Verdict': verdict,
                'Comment': comment,
                'NL_Influence': dynamics['nl_influence'],
                'SL_Influence': dynamics['sl_influence'],
                'Event_Magnitude': dynamics['event_magnitude'],
                'Convergence_Factor': dynamics['convergence_factor']
            })
            enhanced_rows.append(enhanced_row)
        enhanced_df = pd.DataFrame(enhanced_rows)
        # Calculate analysis summary
        avg_score = enhanced_df['Score'].mean()
        avg_magnitude = enhanced_df['Event_Magnitude'].mean()
        # Identify favorable and unfavorable planets from NL and SL columns only
        favorable_planets, unfavorable_planets = self._identify_timeline_planets(enhanced_df, perspective, aggregated=True)
        # Generate team-specific summary
        team_name = "Asc" if perspective == 'ascendant' else "Desc"
        if abs(avg_score) < 0.08:
            summary = f"The enhanced dynamic NL+SL timeline shows an average score of {avg_score:.3f} with event magnitude {avg_magnitude:.2f}. The timeline appears balanced at Star Lord and Sub Lord level with dynamic layer interactions, suggesting a tightly contested match."
        elif avg_score > 0:
            summary = f"The enhanced dynamic NL+SL timeline shows an average score of {avg_score:.3f} with event magnitude {avg_magnitude:.2f}. This indicates a general advantage for {team_name} based on dynamic Star Lord and Sub Lord layer analysis."
        else:
            opponent_name = "Desc" if perspective == 'ascendant' else "Asc"
            summary = f"The enhanced dynamic NL+SL timeline shows an average score of {avg_score:.3f} with event magnitude {avg_magnitude:.2f}. This indicates a general advantage for {opponent_name} based on dynamic Star Lord and Sub Lord layer analysis."
        analysis = {
            "summary": summary,
            "favorable_planets": sorted(favorable_planets),
            "unfavorable_planets": sorted(unfavorable_planets),
            "average_magnitude": avg_magnitude,
            "high_intensity_periods": len(enhanced_df[enhanced_df['Event_Magnitude'] > 2.0]),
            "method": "dynamic_nl_sl_analysis"
        }
        return enhanced_df, analysis

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
        if final_score > 0.75:  # Updated from 0.3 for new top-2 scale
            overall_verdict = "STRONG_ASCENDANT"
            confidence = "High"
        elif final_score > 0.25:  # Updated from 0.1 for new top-2 scale
            overall_verdict = "MODERATE_ASCENDANT"
            confidence = "Medium"
        elif final_score < -0.75:  # Updated from -0.3 for new top-2 scale
            overall_verdict = "STRONG_DESCENDANT"
            confidence = "High"
        elif final_score < -0.25:  # Updated from -0.1 for new top-2 scale
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
        """
        Calculate confidence level based on cusp analysis quality.
        """
        if not cusp_analyses:
            return "Low"
        
        # Count strong cusps (those with high impact magnitude)
        strong_cusps = sum(1 for analysis in cusp_analyses.values() 
                          if analysis.get('impact_magnitude', 0) > 0.5)
        
        total_cusps = len(cusp_analyses)
        strong_ratio = strong_cusps / total_cusps if total_cusps > 0 else 0
        
        if strong_ratio >= 0.7:
            return "Very High"
        elif strong_ratio >= 0.5:
            return "High" 
        elif strong_ratio >= 0.3:
            return "Medium"
        else:
            return "Low"

    # === DYNAMIC TIMELINE ENHANCEMENT METHODS ===
    
    def _get_planetary_natural_strength(self, planet_name: str) -> float:
        """
        Calculate natural strength based on sign placement using authentic KP/Vedic principles.
        Uses moderate intensity modifiers more aligned with KP astrology principles.
        
        Args:
            planet_name: Standardized planet name
            
        Returns:
            float: Natural strength multiplier (0.6-1.5)
        """
        if planet_name not in self.planets.index:
            return 1.0  # Neutral if planet not found
            
        planet_info = self.planets.loc[planet_name]
        planet_sign = planet_info['sign']
        
        # Initialize list to collect all applicable strength factors
        strength_factors = []
        
        # Check debilitation - reduces strength by 40%
        if planet_name in self.DEBILITATION_MAPPING:
            debil_sign, _ = self.DEBILITATION_MAPPING[planet_name]
            if planet_sign == debil_sign:
                strength_factors.append(0.6)  # Weakened but not completely powerless
        
        # Check exaltation - increases strength by 50%
        if planet_name in self.EXALTATION_MAPPING:
            exalt_sign, _ = self.EXALTATION_MAPPING[planet_name]
            if planet_sign == exalt_sign:
                strength_factors.append(1.5)  # Enhanced but not overwhelmingly strong
        
        # Complete sign ownership mapping (authentic Vedic astrology)
        SIGN_OWNERSHIP = {
            'Sun': ['Leo'],
            'Moon': ['Cancer'],
            'Mars': ['Aries', 'Scorpio'],
            'Mercury': ['Gemini', 'Virgo'],
            'Jupiter': ['Sagittarius', 'Pisces'],
            'Venus': ['Taurus', 'Libra'],
            'Saturn': ['Capricorn', 'Aquarius'],
            'Rahu': [],  # Rahu doesn't own any sign
            'Ketu': []   # Ketu doesn't own any sign
        }
        
        # Check own sign - increases strength by 30%
        if planet_name in SIGN_OWNERSHIP and planet_sign in SIGN_OWNERSHIP[planet_name]:
            strength_factors.append(1.3)  # Strong but not as strong as exaltation
        
        # Check friend/enemy relationships through sign lord
        sign_lord = self._get_sign_lord(planet_sign)
        
        if planet_name in PLANETARY_RELATIONSHIPS:
            relationships = PLANETARY_RELATIONSHIPS[planet_name]
            
            if sign_lord in relationships['friends']:
                strength_factors.append(1.2)  # Mild positive influence
            elif sign_lord in relationships['enemies']:
                strength_factors.append(0.8)  # Mild negative influence
            elif sign_lord in relationships['neutrals']:
                strength_factors.append(1.0)  # No modification
        
        # If no factors found, return neutral strength
        if not strength_factors:
            return 1.0
            
        # Calculate geometric mean of all applicable factors
        # This ensures balanced consideration of multiple conditions
        return float(np.exp(np.mean(np.log(strength_factors))))
    
    def _get_sign_lord(self, sign_name: str) -> str:
        """
        Get the ruling planet of a zodiac sign.
        
        Args:
            sign_name: Name of the zodiac sign
            
        Returns:
            str: Name of the ruling planet
        """
        SIGN_LORDS = {
            'Aries': 'Mars',
            'Taurus': 'Venus',
            'Gemini': 'Mercury',
            'Cancer': 'Moon',
            'Leo': 'Sun',
            'Virgo': 'Mercury',
            'Libra': 'Venus',
            'Scorpio': 'Mars',
            'Sagittarius': 'Jupiter',
            'Capricorn': 'Saturn',
            'Aquarius': 'Saturn',
            'Pisces': 'Jupiter'
        }
        
        return SIGN_LORDS.get(sign_name, 'Unknown')
    
    def _get_planetary_positional_strength(self, planet_name: str) -> float:
        """
        Calculate positional strength based on house placement using refined hierarchy.
        Uses priority-based categorization to handle overlapping house classifications.
        
        Args:
            planet_name: Standardized planet name
            
        Returns:
            float: Positional strength multiplier
        """
        significators = self.get_significators(planet_name)
        if not significators:
            return self.POSITIONAL_STRENGTH_MULTIPLIERS['neutral']
        
        # Get primary house (highest rule weight)
        primary_house = max(significators, key=lambda x: SIGNIFICATOR_RULE_WEIGHTS.get(x[1], 0))[0]
        
        # Use priority-based hierarchy to handle overlapping classifications
        # Priority: Kendra > Trinal > Upachaya > Maraka > Dusthana > Others
        
        if primary_house in [1, 4, 7, 10]:
            # Kendra (Angular) houses - highest priority
            if primary_house == 1:
                return self.POSITIONAL_STRENGTH_MULTIPLIERS['kendra']  # 1 is both kendra and trinal
            elif primary_house == 4:
                return self.POSITIONAL_STRENGTH_MULTIPLIERS['kendra']  # Also sukha, but kendra takes priority
            elif primary_house == 7:
                return self.POSITIONAL_STRENGTH_MULTIPLIERS['maraka_strong']  # 7 is kendra but also strong maraka
            elif primary_house == 10:
                return self.POSITIONAL_STRENGTH_MULTIPLIERS['kendra']  # 10 is both kendra and upachaya
        
        elif primary_house in [5, 9]:
            # Pure trinal houses (1 already handled as kendra)
            return self.POSITIONAL_STRENGTH_MULTIPLIERS['trinal_pure']
        
        elif primary_house in [3, 11]:
            # Pure upachaya houses (6, 10 handled elsewhere)
            return self.POSITIONAL_STRENGTH_MULTIPLIERS['upachaya_pure']
        
        elif primary_house == 6:
            # Mixed house: upachaya + dusthana + victory house
            return self.POSITIONAL_STRENGTH_MULTIPLIERS['upachaya_mixed']
        
        elif primary_house == 2:
            # Mild maraka (resources)
            return self.POSITIONAL_STRENGTH_MULTIPLIERS['maraka_mild']
        
        elif primary_house in [8, 12]:
            # Strong dusthana houses
            return self.POSITIONAL_STRENGTH_MULTIPLIERS['dusthana_strong']
        
        else:
            return self.POSITIONAL_STRENGTH_MULTIPLIERS['neutral']
    
    def _get_significator_relevance(self, planet_name: str, perspective: str) -> float:
        """
        Calculate how relevant planet's significators are for match outcome.
        
        Args:
            planet_name: Standardized planet name
            perspective: 'ascendant' or 'descendant'
            
        Returns:
            float: Relevance multiplier (0.4 to 1.0)
        """
        significators = self.get_significators(planet_name)
        if not significators:
            return 0.4
        
        # Count primary significators (victory/defeat related houses)
        victory_houses = [1, 6, 10, 11]
        defeat_houses = [5, 7, 8, 12]
        
        primary_count = sum(1 for house, rule in significators 
                          if house in victory_houses + defeat_houses 
                          and SIGNIFICATOR_RULE_WEIGHTS.get(rule, 0) >= 1.0)
        
        total_significators = len(significators)
        
        if total_significators == 0:
            return 0.4
        
        relevance_ratio = primary_count / total_significators
        
        # Map to relevance multiplier
        if relevance_ratio >= 0.8:
            return 1.0
        elif relevance_ratio >= 0.6:
            return 0.8
        elif relevance_ratio >= 0.4:
            return 0.6
        else:
            return 0.4
    
    def _calculate_planetary_effective_power(self, planet_name: str, layer: str, perspective: str) -> float:
        """
        Calculate effective power of a planet in a specific timeline layer.
        
        Args:
            planet_name: Standardized planet name
            layer: 'nl', 'sl', or 'ssl'
            perspective: 'ascendant' or 'descendant'
            
        Returns:
            float: Effective power value
        """
        # Get planetary strengths
        natural_strength = self._get_planetary_natural_strength(planet_name)
        positional_strength = self._get_planetary_positional_strength(planet_name)
        temporal_weight = self.TEMPORAL_STRENGTH_WEIGHTS[layer]
        significator_relevance = self._get_significator_relevance(planet_name, perspective)
        
        # Calculate effective power
        effective_power = natural_strength * positional_strength * temporal_weight * significator_relevance
        
        return effective_power
    
    def _calculate_dynamic_layer_influences(self, nl_planet: str, sl_planet: str, ssl_planet: str, perspective: str) -> dict:
        """
        Calculate dynamic layer influences for a timeline period.
        
        Args:
            nl_planet: Nakshatra Lord planet name
            sl_planet: Sub Lord planet name  
            ssl_planet: Sub-Sub Lord planet name
            perspective: 'ascendant' or 'descendant'
            
        Returns:
            dict: Layer influences and convergence data
        """
        # Calculate effective powers
        nl_power = self._calculate_planetary_effective_power(nl_planet, 'nl', perspective)
        sl_power = self._calculate_planetary_effective_power(sl_planet, 'sl', perspective)
        ssl_power = self._calculate_planetary_effective_power(ssl_planet, 'ssl', perspective) if ssl_planet else 0.0
        
        total_power = nl_power + sl_power + ssl_power
        
        if total_power == 0:
            return {
                'nl_influence': 0.33,
                'sl_influence': 0.33,
                'ssl_influence': 0.34,
                'convergence_factor': 1.0,
                'event_magnitude': 1.0
            }
        
        # Calculate influence percentages
        nl_influence = nl_power / total_power
        sl_influence = sl_power / total_power
        ssl_influence = ssl_power / total_power if ssl_planet else 0.0
        
        # Calculate convergence factor with enhanced planetary relationship analysis
        nl_score = self.calculate_planet_score(nl_planet, perspective)
        sl_score = self.calculate_planet_score(sl_planet, perspective)
        ssl_score = self.calculate_planet_score(ssl_planet, perspective) if ssl_planet else 0.0
        
        # Check alignment of planetary directions
        scores = [nl_score, sl_score, ssl_score] if ssl_planet else [nl_score, sl_score]
        positive_scores = [s for s in scores if s > 0.05]
        negative_scores = [s for s in scores if s < -0.05]
        
        # Enhanced convergence calculation considering planetary relationships
        planets = [nl_planet, sl_planet] + ([ssl_planet] if ssl_planet else [])
        
        # Check for enemy planets working together (creates disharmony)
        enemy_combinations = 0
        total_combinations = 0
        
        for i, planet1 in enumerate(planets):
            for j, planet2 in enumerate(planets[i+1:], i+1):
                total_combinations += 1
                if planet1 in PLANETARY_RELATIONSHIPS and planet2 in PLANETARY_RELATIONSHIPS[planet1]['enemies']:
                    enemy_combinations += 1
        
        # Base convergence on score alignment
        if len(positive_scores) == len(scores) or len(negative_scores) == len(scores):
            base_convergence = 2.0  # All align
        elif len(positive_scores) >= 2 or len(negative_scores) >= 2:
            base_convergence = 1.3  # Majority align  
        else:
            base_convergence = 0.8  # Mixed signals
        
        # Apply enemy planet penalty (enemies working together reduces harmony)
        if total_combinations > 0:
            enemy_ratio = enemy_combinations / total_combinations
            enemy_penalty = 1.0 - (enemy_ratio * 0.3)  # Up to 30% reduction
            convergence_factor = base_convergence * enemy_penalty
        else:
            convergence_factor = base_convergence
        
        # Ensure convergence factor stays within reasonable bounds
        convergence_factor = max(0.4, min(2.5, convergence_factor))
        
        # Calculate event magnitude
        event_magnitude = total_power * convergence_factor
        
        return {
            'nl_influence': nl_influence,
            'sl_influence': sl_influence,
            'ssl_influence': ssl_influence,
            'convergence_factor': convergence_factor,
            'event_magnitude': event_magnitude,
            'nl_power': nl_power,
            'sl_power': sl_power,
            'ssl_power': ssl_power,
            'total_power': total_power
        }

    def _generate_dynamic_verdict_and_comment(self, row, perspective, dynamics, nl_score, sl_score, ssl_score, final_score):
        """
        Generates enhanced verdict and comment based on dynamic layer influences.
        
        Args:
            row: Row from timeline DataFrame
            perspective: Either 'ascendant' or 'descendant'
            dynamics: Dynamic layer influences
            nl_score: Star Lord score
            sl_score: Sub Lord score
            ssl_score: Sub-Sub Lord score
            final_score: Final score based on dynamic layer influences
            
        Returns:
            tuple: (verdict, comment)
        """
        nl_planet = row.get('NL_Planet')
        sl_planet = row.get('SL_Planet')
        ssl_planet = row.get('SSL_Planet')
        
        # Standardize planet names for processing
        nl_standardized = PlanetNameUtils.standardize_for_index(nl_planet) if pd.notna(nl_planet) else 'Unknown'
        sl_standardized = PlanetNameUtils.standardize_for_index(sl_planet) if pd.notna(sl_planet) else 'Unknown'
        ssl_standardized = PlanetNameUtils.standardize_for_index(ssl_planet) if pd.notna(ssl_planet) else 'Unknown'
        
        team_name = "Asc" if perspective == 'ascendant' else "Desc"
        opponent_name = "Desc" if perspective == 'ascendant' else "Asc"
        
        # Determine verdict based on final score
        if final_score >= 0.75:  # Updated from 0.3 for new top-2 scale
            verdict = f"Strong Advantage {team_name}"
            cricket_context = "Excellent period for building partnerships and dominating opponents"
            confidence_level = "HIGH"
        elif final_score >= 0.375:  # Updated from 0.15 for new top-2 scale
            verdict = f"Advantage {team_name}"
            cricket_context = "Good period for consolidation and steady progress"
            confidence_level = "MEDIUM"
        elif final_score > 0.125:  # Updated from 0.05 for new top-2 scale
            verdict = f"Balanced (Slight {team_name})"
            cricket_context = "Marginal advantage - gradual progress expected"
            confidence_level = "LOW"
        elif final_score <= -0.75:  # Updated from -0.3 for new top-2 scale
            verdict = f"Strong Advantage {opponent_name}"
            cricket_context = "Challenging period - wickets or pressure likely"
            confidence_level = "HIGH"
        elif final_score <= -0.375:  # Updated from -0.15 for new top-2 scale
            verdict = f"Advantage {opponent_name}"
            cricket_context = "Opposition builds pressure and momentum"
            confidence_level = "MEDIUM"
        elif final_score < -0.125:  # Updated from -0.05 for new top-2 scale
            verdict = f"Balanced (Slight {opponent_name})"
            cricket_context = "Slight opposition edge - careful play needed"
            confidence_level = "LOW"
        else:
            verdict = "Balanced Period"
            cricket_context = "Evenly matched phase with gradual developments"
            confidence_level = "LOW"
        
        # Generate planetary role descriptions
        nl_promise_desc = f"dominance ({dynamics['nl_influence']:.1%})"
        sl_mod_desc = f"modification ({dynamics['sl_influence']:.1%})"
        ssl_del_desc = f"delivery ({dynamics['ssl_influence']:.1%})"
        
        comment_parts = []
        
        # Add basic planetary descriptions
        comment_parts.append(f"🌟 {nl_planet} {nl_promise_desc}")
        comment_parts.append(f"⚖️ {sl_planet} {sl_mod_desc}")
        if ssl_planet:
            comment_parts.append(f"🎯 {ssl_planet} {ssl_del_desc}")
        
        # Add convergence information
        if dynamics['convergence_factor'] >= 1.8:
            comment_parts.append("🔥 High convergence - aligned energies")
        elif dynamics['convergence_factor'] <= 0.9:
            comment_parts.append("⚡ Mixed signals - competing influences")
        
        comment_parts.append(f"🏏 {cricket_context}")
        comment_parts.append(f"📊 Magnitude: {dynamics['event_magnitude']:.2f} | Score: {final_score:+.3f} | {confidence_level}")
        
        detailed_comment = " | ".join(comment_parts)
        
        return verdict, detailed_comment

    def _generate_dynamic_nl_sl_verdict_and_comment(self, row, perspective, dynamics, nl_score, sl_score, final_score):
        """
        Generates enhanced verdict and comment for NL+SL analysis based on dynamic layer influences.
        
        Args:
            row: Row from timeline DataFrame
            perspective: Either 'ascendant' or 'descendant'
            dynamics: Dynamic layer influences
            nl_score: Star Lord score
            sl_score: Sub Lord score
            final_score: Final score based on dynamic layer influences
            
        Returns:
            tuple: (verdict, comment)
        """
        nl_planet = row.get('NL_Planet')
        sl_planet = row.get('SL_Planet')
        
        # Standardize planet names for processing
        nl_standardized = PlanetNameUtils.standardize_for_index(nl_planet) if pd.notna(nl_planet) else 'Unknown'
        sl_standardized = PlanetNameUtils.standardize_for_index(sl_planet) if pd.notna(sl_planet) else 'Unknown'
        
        team_name = "Asc" if perspective == 'ascendant' else "Desc"
        opponent_name = "Desc" if perspective == 'ascendant' else "Asc"
        
        # Determine verdict based on final score
        if final_score >= 0.625:  # Updated from 0.25 for new top-2 scale
            verdict = f"Strong Advantage {team_name}"
            cricket_context = "Excellent period for building partnerships and dominating opponents"
            confidence_level = "HIGH"
        elif final_score >= 0.3:  # Updated from 0.12 for new top-2 scale
            verdict = f"Advantage {team_name}"
            cricket_context = "Good period for consolidation and steady progress"
            confidence_level = "MEDIUM"
        elif final_score > 0.125:  # Updated from 0.05 for new top-2 scale
            verdict = f"Balanced (Slight {team_name})"
            cricket_context = "Marginal advantage - gradual progress expected"
            confidence_level = "LOW"
        elif final_score <= -0.625:  # Updated from -0.25 for new top-2 scale
            verdict = f"Strong Advantage {opponent_name}"
            cricket_context = "Challenging period - wickets or pressure likely"
            confidence_level = "HIGH"
        elif final_score <= -0.3:  # Updated from -0.12 for new top-2 scale
            verdict = f"Advantage {opponent_name}"
            cricket_context = "Opposition builds pressure and momentum"
            confidence_level = "MEDIUM"
        elif final_score < -0.125:  # Updated from -0.05 for new top-2 scale
            verdict = f"Balanced (Slight {opponent_name})"
            cricket_context = "Slight opposition edge - careful play needed"
            confidence_level = "LOW"
        else:
            verdict = "Balanced Period"
            cricket_context = "Evenly matched phase with gradual developments"
            confidence_level = "LOW"
        
        # Generate planetary role descriptions
        nl_promise_desc = f"dominance ({dynamics['nl_influence']:.1%})"
        sl_mod_desc = f"modification ({dynamics['sl_influence']:.1%})"
        
        comment_parts = []
        
        # Add basic planetary descriptions
        comment_parts.append(f"🌟 {nl_planet} {nl_promise_desc}")
        comment_parts.append(f"⚖️ {sl_planet} {sl_mod_desc}")
        
        # Add convergence information
        if dynamics['convergence_factor'] >= 1.8:
            comment_parts.append("�� High convergence - aligned energies")
        elif dynamics['convergence_factor'] <= 0.9:
            comment_parts.append("⚡ Mixed signals - competing influences")
        
        comment_parts.append(f"🏏 {cricket_context}")
        comment_parts.append(f"📊 Magnitude: {dynamics['event_magnitude']:.2f} | NL:{nl_score:+.2f} SL:{sl_score:+.2f} Combined:{final_score:+.3f} | {confidence_level}")
        
        detailed_comment = " | ".join(comment_parts)
        
        return verdict, detailed_comment

    def _identify_timeline_planets(self, df, perspective, aggregated=False):
        """
        Identifies favorable and unfavorable planets based on the timeline DataFrame.
        
        Args:
            df: Timeline DataFrame
            perspective: Either 'ascendant' or 'descendant'
            aggregated: Boolean indicating if this is an aggregated timeline
            
        Returns:
            tuple: (favorable_planets, unfavorable_planets)
        """
        if aggregated:
            planet_columns = ['NL_Planet', 'SL_Planet']
        else:
            planet_columns = ['NL_Planet', 'SL_Planet', 'SSL_Planet']
        
        unique_planets = pd.unique(df[planet_columns].values.ravel())
        unique_planets = [p for p in unique_planets if pd.notna(p)]
        
        favorable_planets = []
        unfavorable_planets = []
        
        for planet in unique_planets:
            score = self.calculate_planet_score(planet, perspective)
            if score > 0:
                favorable_planets.append(planet)
            elif score < 0:
                unfavorable_planets.append(planet)
        
        return favorable_planets, unfavorable_planets

    def _calculate_asc_timeline_score(self, nl_score: float, sl_score: float, context_score: float) -> float:
        # Use weights for Asc timeline
        w = self.timeline_weights.get('asc_timeline', self.DEFAULT_TIMELINE_WEIGHTS['asc_timeline'])
        return (nl_score * w['NL']) + (sl_score * w['SL']) + (context_score * w['Context'])



 