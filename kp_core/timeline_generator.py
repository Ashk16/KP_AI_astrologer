import pandas as pd
from datetime import datetime, timedelta
import swisseph as swe
import os
import sys

# --- Path Correction ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from kp_core.kp_engine import KPEngine, PLANET_NAMES, PlanetNameUtils

class TimelineGenerator:
    """
    Generates a timeline of lord changes for a given celestial point (Ascendant, Descendant, or Moon).
    """
    def __init__(self, engine: KPEngine, body_name: str):
        """
        Args:
            engine (KPEngine): An initialized KPEngine instance.
            body_name (str): Either 'Ascendant', 'Descendant', or 'Moon'.
        """
        if body_name not in ['Ascendant', 'Descendant', 'Moon']:
            raise ValueError("body_name must be 'Ascendant', 'Descendant', or 'Moon'")
        
        self.engine = engine
        self.body_name = body_name
        
        # Set body_id based on the type
        if body_name == 'Ascendant':
            self.body_id = 1
        elif body_name == 'Descendant':
            self.body_id = 7
        else:  # Moon
            self.body_id = None  # Moon is not a cusp, handled differently

    def _get_body_details_at_time(self, dt_utc: datetime):
        """Calculates lordship for the body at a specific time."""
        if self.body_name == 'Moon':
            # For Moon, we need to calculate its position at the given time
            jd = swe.julday(dt_utc.year, dt_utc.month, dt_utc.day, 
                           dt_utc.hour + dt_utc.minute/60 + dt_utc.second/3600)
            pos, _ = swe.calc_ut(jd, swe.MOON, swe.FLG_SWIEPH)
            longitude = pos[0]
        else:
            # For Ascendant/Descendant, use cusp calculation
            longitude = self.engine.get_cusp_longitude_at_time(dt_utc, self.body_id)
        
        nl, sl, ssl = self.engine._get_lordships(longitude)
        return {'nl': nl, 'sl': sl, 'ssl': ssl, 'longitude': longitude}

    def find_next_ssl_transition(self, current_dt: datetime, initial_ssl: str):
        """
        Finds the exact time of the next SSL transition using a two-phase search.
        """
        time_cursor = current_dt
        # The end_dt is determined by when the timeline generation is called
        end_dt = getattr(self, '_end_dt_for_search', current_dt + timedelta(hours=8)) # Failsafe

        # Phase 1: Coarse search (1-minute intervals)
        search_window_start = None
        while time_cursor < end_dt:
            time_cursor += timedelta(minutes=1)
            if time_cursor >= end_dt:
                break
            details = self._get_body_details_at_time(time_cursor)
            if details['ssl'] != initial_ssl:
                search_window_start = time_cursor - timedelta(minutes=1)
                break
        
        if not search_window_start:
            return None

        # Phase 2: Fine-grained binary search (to the second)
        low, high = search_window_start, time_cursor
        transition_point = high

        while (high - low).total_seconds() > 1:
            mid = low + (high - low) / 2
            details = self._get_body_details_at_time(mid)
            if details['ssl'] == initial_ssl:
                low = mid
            else:
                high, transition_point = mid, mid
        
        return transition_point.replace(microsecond=0)

    def generate_timeline_df(self, start_dt: datetime, duration_hours: float):
        """
        Generates the full timeline DataFrame.
        """
        self._end_dt_for_search = start_dt + timedelta(hours=duration_hours)
        timeline_data = []
        current_time = start_dt
        
        while current_time < self._end_dt_for_search:
            start_details = self._get_body_details_at_time(current_time)
            transition_time = self.find_next_ssl_transition(current_time, start_details['ssl'])
            end_time = transition_time if transition_time and transition_time < self._end_dt_for_search else self._end_dt_for_search

            if (end_time - current_time).total_seconds() < 1:
                current_time = end_time
                continue

            # Calculate score using a dummy AnalysisEngine for now.
            # This part can be integrated better later.
            score = 0 # Placeholder

            timeline_data.append({
                'Start Time': current_time,
                'End Time': end_time,
                'NL_Planet': start_details['nl'],
                'SL_Planet': start_details['sl'],
                'SSL_Planet': start_details['ssl'],
                'Score': score
            })
            
            current_time = end_time

        return pd.DataFrame(timeline_data)


if __name__ == '__main__':
    # Example usage for testing
    try:
        # Ensure SWEP_PATH is set
        os.environ['SWEP_PATH'] = 'C:/sweph/ephe'

        start_time = datetime.utcnow()
        lat, lon = 19.0760, 72.8777
        duration = 3.5 # hours

        print(f"--- Generating Ascendant SSL Timeline for {duration} hours ---")
        asc_gen = TimelineGenerator(KPEngine(start_time, lat, lon), 'Ascendant')
        asc_timeline = asc_gen.generate_timeline_df(start_time, duration)
        print(asc_timeline)

        print(f"\n--- Generating Descendant SSL Timeline for {duration} hours ---")
        desc_gen = TimelineGenerator(KPEngine(start_time, lat, lon), 'Descendant')
        desc_timeline = desc_gen.generate_timeline_df(start_time, duration)
        print(desc_timeline)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc() 