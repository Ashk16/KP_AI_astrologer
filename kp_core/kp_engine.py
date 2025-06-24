import swisseph as swe
import pandas as pd
from datetime import datetime, timedelta
import os

# --- Constants and Mappings ---

PLANET_NAMES = {
    swe.SUN: 'Sun', swe.MOON: 'Moon', swe.MARS: 'Mars', swe.MERCURY: 'Mercury',
    swe.JUPITER: 'Jupiter', swe.VENUS: 'Venus', swe.SATURN: 'Saturn',
    swe.MEAN_NODE: 'Rahu', -swe.MEAN_NODE: 'Ketu', swe.ASC: 'Asc'
}

# Short planet names for display
PLANET_SHORT_NAMES = {
    'Sun': 'Su', 'Moon': 'Mo', 'Mars': 'Ma', 'Mercury': 'Me', 'Jupiter': 'Ju',
    'Venus': 'Ve', 'Saturn': 'Sa', 'Rahu': 'Ra', 'Ketu': 'Ke', 'Ascendant': 'Asc'
}

ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra",
    "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

SIGN_LORDS = {
    "Aries": "Mars", "Taurus": "Venus", "Gemini": "Mercury", "Cancer": "Moon",
    "Leo": "Sun", "Virgo": "Mercury", "Libra": "Venus", "Scorpio": "Mars",
    "Sagittarius": "Jupiter", "Capricorn": "Saturn", "Aquarius": "Saturn", "Pisces": "Jupiter"
}

NAKSHATRA_LORDS = [
    "Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter",
    "Saturn", "Mercury"
] * 3

# This table is the heart of the Sub-Lord calculation in KP.
# It's derived from the Vimsottari Dasha sequence but applied to the zodiac.
SUB_LORD_SEQUENCE = [
    "Ke", "Ve", "Su", "Mo", "Ma", "Ra", "Ju", "Sa", "Me"
]

# Vimsottari Dasha years for each planet lord
DASA_YEARS = {
    "Ke": 7, "Ve": 20, "Su": 6, "Mo": 10, "Ma": 7,
    "Ra": 18, "Ju": 16, "Sa": 19, "Me": 17
}
TOTAL_DASA_YEARS = 120

class KPEngine:
    """
    Handles core KP astrological calculations.
    """
    def __init__(self, dt, lat, lon):
        """
        Initializes the engine with date, time, and location.

        Args:
            dt (datetime): UTC datetime object.
            lat (float): Latitude.
            lon (float): Longitude.
        """
        self.utc_dt = dt
        self.lat = lat
        self.lon = lon
        self.jd = swe.julday(self.utc_dt.year, self.utc_dt.month, self.utc_dt.day, 
                             self.utc_dt.hour + self.utc_dt.minute/60 + self.utc_dt.second/3600)
        
        # Set Swiss Ephemeris path
        # This needs to be configured to the location of your SE files
        # It's better to set this as an environment variable.
        swe.set_ephe_path(os.environ.get('SWEP_PATH'))

        self.planets = self._calculate_all_body_details()
        self.cusps = self._calculate_all_cusp_details()
        self.planets['Asc'] = self.cusps[1] # Ensure Asc is in planets list

    def _get_lordships(self, longitude):
        """
        Calculates Nakshatra, Sub, and Sub-Sub Lords for a given longitude
        using precise Vimsottari Dasha proportions.
        """
        # --- Nakshatra (Star Lord) Calculation ---
        nakshatra_span = 13 + 1/3
        nakshatra_num = int(longitude / nakshatra_span)
        nakshatra_lord_name = NAKSHATRA_LORDS[nakshatra_num]
        nakshatra_lord = PLANET_SHORT_NAMES[nakshatra_lord_name]

        # --- Sub Lord Calculation ---
        start_of_nakshatra = nakshatra_num * nakshatra_span
        arc_in_nakshatra = longitude - start_of_nakshatra
        
        position_in_nakshatra = 0
        sub_lord = None
        # The sequence of sub-lords within a nakshatra follows the dasa sequence
        # starting from the nakshatra lord itself.
        sub_lord_dasa_sequence = SUB_LORD_SEQUENCE[SUB_LORD_SEQUENCE.index(nakshatra_lord):] + \
                                 SUB_LORD_SEQUENCE[:SUB_LORD_SEQUENCE.index(nakshatra_lord)]

        for lord in sub_lord_dasa_sequence:
            sub_lord_span = (DASA_YEARS[lord] / TOTAL_DASA_YEARS) * nakshatra_span
            if arc_in_nakshatra >= position_in_nakshatra and arc_in_nakshatra < position_in_nakshatra + sub_lord_span:
                sub_lord = lord
                break
            position_in_nakshatra += sub_lord_span

        # --- Sub-Sub Lord Calculation ---
        arc_in_sub_lord = arc_in_nakshatra - position_in_nakshatra
        
        position_in_sub_lord = 0
        sub_sub_lord = None
        # The sequence of sub-sub-lords within a sub-lord period also follows the dasa sequence,
        # starting from the sub-lord itself.
        sub_sub_lord_dasa_sequence = SUB_LORD_SEQUENCE[SUB_LORD_SEQUENCE.index(sub_lord):] + \
                                     SUB_LORD_SEQUENCE[:SUB_LORD_SEQUENCE.index(sub_lord)]

        for lord in sub_sub_lord_dasa_sequence:
            # The span of a sub-sub-lord within a sub-lord's arc
            sub_sub_lord_span = (DASA_YEARS[lord] / TOTAL_DASA_YEARS) * ((DASA_YEARS[sub_lord] / TOTAL_DASA_YEARS) * nakshatra_span)
            if arc_in_sub_lord >= position_in_sub_lord and arc_in_sub_lord < position_in_sub_lord + sub_sub_lord_span:
                sub_sub_lord = lord
                break
            position_in_sub_lord += sub_sub_lord_span

        return nakshatra_lord, sub_lord, sub_sub_lord

    def _calculate_all_body_details(self):
        """Calculates positions and lordships for all planets."""
        planet_data = {}
        for p_id, name in PLANET_NAMES.items():
            if name == 'Asc': continue # Handled in cusps

            if name in ['Rahu', 'Ketu']:
                # pos is an immutable tuple, so we can't modify it directly.
                pos, _ = swe.calc_ut(self.jd, swe.MEAN_NODE, swe.FLG_SWIEPH)
                longitude = pos[0]
                if name == 'Ketu':
                    # Ketu is 180 degrees opposite Rahu.
                    longitude = (longitude + 180) % 360
            else:
                pos, _ = swe.calc_ut(self.jd, p_id, swe.FLG_SWIEPH)
                longitude = pos[0]

            sign_num = int(longitude / 30)
            sign = ZODIAC_SIGNS[sign_num]
            nl, sl, ssl = self._get_lordships(longitude)

            planet_data[name] = {
                'longitude': longitude,
                'sign': sign,
                'sign_lord': PLANET_SHORT_NAMES[SIGN_LORDS[sign]],
                'nl': nl,
                'sl': sl,
                'ssl': ssl
            }
        return planet_data

    def _calculate_all_cusp_details(self):
        """Calculates positions and lordships for all cusps."""
        cusps, ascmc = swe.houses(self.jd, self.lat, self.lon, b'P')
        cusp_data = {}
        for i in range(12):
            longitude = cusps[i]
            sign_num = int(longitude / 30)
            sign = ZODIAC_SIGNS[sign_num]
            nl, sl, ssl = self._get_lordships(longitude)

            cusp_data[i + 1] = {
                'longitude': longitude,
                'sign': sign,
                'sign_lord': PLANET_SHORT_NAMES[SIGN_LORDS[sign]],
                'nl': nl,
                'sl': sl,
                'ssl': ssl
            }
        return cusp_data

    def get_planet_details(self, planet_name):
        """Returns the full details for a given planet."""
        if planet_name not in self.planets:
            return None
        return self.planets[planet_name]

    def get_all_planets_df(self):
        """Returns all planetary data as a pandas DataFrame."""
        df = pd.DataFrame.from_dict(self.planets, orient='index')
        df.index.name = 'Planet'
        # We will add Nakshatra, Sub Lord, Sub-Sub Lord columns later
        return df

    def get_cusp_details(self, cusp_number):
        """Returns the details for a given cusp."""
        if cusp_number not in self.cusps:
            return None
        return self.cusps[cusp_number]

    def get_all_cusps_df(self):
        """Returns all cusp data as a pandas DataFrame."""
        df = pd.DataFrame.from_dict(self.cusps, orient='index')
        df.index.name = 'Cusp'
        return df

if __name__ == '__main__':
    # Example Usage for testing
    # This requires the Swiss Ephemeris files to be installed and path set
    try:
        # --- You might need to set the SWEP_PATH environment variable ---
        # For example: os.environ['SWEP_PATH'] = 'C:/sweph/ephe'
        
        utc_now = datetime.utcnow()
        engine = KPEngine(dt=utc_now, lat=19.0760, lon=72.8777)
        
        print("--- Planetary Positions ---")
        print(engine.get_all_planets_df())
        
        print("\n--- Cusp Positions ---")
        print(engine.get_all_cusps_df())
        
        print("\n--- Moon Details ---")
        print(engine.get_planet_details('Moon'))

    except Exception as e:
        print(f"An error occurred. Please ensure the Swiss Ephemeris path is set correctly.")
        print(f"Error: {e}")
        print("Hint: Set the SWEP_PATH environment variable to your Swiss Ephemeris files directory.") 