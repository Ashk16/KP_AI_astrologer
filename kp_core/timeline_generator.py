from datetime import datetime, timedelta
import swisseph as swe
import pandas as pd
import os
import sys

# --- Path Correction ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from kp_core.kp_engine import KPEngine, PLANET_NAMES

class TimelineGenerator:
    """
    Generates a timeline of lord changes for a given celestial body.
    """
    def __init__(self, start_dt_utc, lat, lon, body, duration_hours):
        """
        Args:
            start_dt_utc (datetime): The UTC start time of the match.
            lat (float): Latitude of the event.
            lon (float): Longitude of the event.
            body (int): The swisseph ID for the celestial body (e.g., swe.ASC, swe.MOON).
            duration_hours (float): The duration of the match in hours.
        """
        self.start_dt = start_dt_utc
        self.lat = lat
        self.lon = lon
        self.body = body
        self.end_dt = self.start_dt + timedelta(hours=duration_hours)
        self.body_name = PLANET_NAMES.get(self.body, 'Unknown')

    def _get_body_details_at_time(self, dt):
        """Calculates lordship for the body at a specific time."""
        jd = swe.julday(dt.year, dt.month, dt.day, dt.hour + dt.minute/60 + dt.second/3600)
        
        if self.body == swe.ASC:
            cusps, _ = swe.houses(jd, self.lat, self.lon, b'P')
            longitude = cusps[0]
        else:
            pos, _ = swe.calc_ut(jd, self.body, swe.FLG_SWIEPH)
            longitude = pos[0]

        # We need a KPEngine instance to use its lordship calculation logic
        # This is slightly inefficient but ensures consistency.
        # A dummy KPEngine is created here. For optimization, this could be refactored.
        engine = KPEngine(dt, self.lat, self.lon)
        nl, sl, ssl = engine._get_lordships(longitude)
        return {'nl': nl, 'sl': sl, 'ssl': ssl, 'longitude': longitude}


    def find_next_ssl_transition(self, current_dt, initial_ssl):
        """
        Finds the exact time of the next SSL transition using a two-phase search.
        """
        time_cursor = current_dt
        
        # Phase 1: Coarse search (1-minute intervals) to find a window with a change
        search_window_start = None
        while time_cursor < self.end_dt:
            time_cursor += timedelta(minutes=1)
            if time_cursor >= self.end_dt: # Ensure we don't search past the end time
                 break
            details = self._get_body_details_at_time(time_cursor)
            if details['ssl'] != initial_ssl:
                search_window_start = time_cursor - timedelta(minutes=1)
                break
        
        if not search_window_start:
            return None # No transition found within the match duration

        # Phase 2: Fine-grained binary search (to the second)
        low = search_window_start
        high = time_cursor
        transition_point = high

        while (high - low).total_seconds() > 1:
            mid = low + (high - low) / 2
            details = self._get_body_details_at_time(mid)
            if details['ssl'] == initial_ssl:
                low = mid
            else:
                high = mid
                transition_point = mid
        
        # The transition happens at the beginning of the 'high' interval
        return transition_point.replace(microsecond=0)

    def generate_timeline(self):
        """
        Generates the full timeline of CSL, SL, and SSL changes.
        """
        timeline = []
        current_time = self.start_dt
        
        while current_time < self.end_dt:
            # Get details at the start of the current period
            start_details = self._get_body_details_at_time(current_time)
            
            # Find the next transition time
            transition_time = self.find_next_ssl_transition(current_time, start_details['ssl'])
            
            end_time = transition_time if transition_time and transition_time < self.end_dt else self.end_dt

            # De-duplication: if a period is less than a second, skip it
            if (end_time - current_time).total_seconds() < 1:
                current_time = end_time
                continue

            # Append the period to the timeline
            timeline.append({
                'Start Time': current_time,
                'End Time': end_time,
                'NL': start_details['nl'],
                'SL': start_details['sl'],
                'SSL': start_details['ssl']
            })
            
            # Move cursor to the start of the next period
            current_time = end_time

        return pd.DataFrame(timeline)


if __name__ == '__main__':
    # Example usage for testing
    try:
        # Ensure SWEP_PATH is set
        os.environ['SWEP_PATH'] = 'C:/sweph/ephe'

        start_time = datetime.utcnow()
        lat, lon = 19.0760, 72.8777
        duration = 3.5 # hours

        print(f"--- Generating Ascendant SSL Timeline for {duration} hours ---")
        asc_gen = TimelineGenerator(start_time, lat, lon, swe.ASC, duration)
        asc_timeline = asc_gen.generate_timeline()
        print(asc_timeline)

        print(f"\n--- Generating Moon SSL Timeline for {duration} hours ---")
        moon_gen = TimelineGenerator(start_time, lat, lon, swe.MOON, duration)
        moon_timeline = moon_gen.generate_timeline()
        print(moon_timeline)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc() 