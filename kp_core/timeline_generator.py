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

    def find_next_sl_transition(self, current_dt: datetime, initial_sl: str):
        """
        Finds the exact time of the next SL (Sub Lord) transition using a two-phase search.
        """
        time_cursor = current_dt
        # The end_dt is determined by when the timeline generation is called
        end_dt = getattr(self, '_end_dt_for_search', current_dt + timedelta(hours=8)) # Failsafe

        # Phase 1: Coarse search (5-minute intervals for better coverage)
        search_window_start = None
        max_search_iterations = int((end_dt - current_dt).total_seconds() / 300) + 10  # 5-minute intervals + buffer
        search_iterations = 0
        
        while time_cursor < end_dt and search_iterations < max_search_iterations:
            search_iterations += 1
            time_cursor += timedelta(minutes=5)  # Reduced from 10 to 5 minutes
            if time_cursor >= end_dt:
                break
            try:
                details = self._get_body_details_at_time(time_cursor)
                if details['sl'] != initial_sl:
                    search_window_start = time_cursor - timedelta(minutes=5)
                    break
            except Exception:
                # Skip problematic times and continue
                continue
        
        if not search_window_start:
            return None

        # Phase 2: Fine-grained binary search (to the second, not minute!)
        low, high = search_window_start, time_cursor
        transition_point = high
        binary_iterations = 0
        max_binary_iterations = 20  # Prevent infinite binary search

        while (high - low).total_seconds() > 1 and binary_iterations < max_binary_iterations:  # Search to SECOND precision for SL
            binary_iterations += 1
            mid = low + (high - low) / 2
            try:
                details = self._get_body_details_at_time(mid)
                if details['sl'] == initial_sl:
                    low = mid
                else:
                    high, transition_point = mid, mid
            except Exception:
                # If calculation fails, use high as transition point
                break
        
        # Return with seconds precision, not rounded to minutes!
        return transition_point.replace(microsecond=0)  # Keep seconds, remove only microseconds

    def find_next_nl_or_sl_transition(self, current_dt: datetime, initial_nl: str, initial_sl: str):
        """
        Finds the exact time when either NL (Star Lord) or SL (Sub Lord) changes.
        This is used for aggregated timelines to capture the complete NL+SL period.
        """
        time_cursor = current_dt
        end_dt = getattr(self, '_end_dt_for_search', current_dt + timedelta(hours=8))

        # Phase 1: Coarse search (2-minute intervals for better precision)
        search_window_start = None
        max_search_iterations = int((end_dt - current_dt).total_seconds() / 120) + 10
        search_iterations = 0
        
        while time_cursor < end_dt and search_iterations < max_search_iterations:
            search_iterations += 1
            time_cursor += timedelta(minutes=2)
            if time_cursor >= end_dt:
                break
            try:
                details = self._get_body_details_at_time(time_cursor)
                # Check if either NL or SL has changed
                if details['nl'] != initial_nl or details['sl'] != initial_sl:
                    search_window_start = time_cursor - timedelta(minutes=2)
                    break
            except Exception:
                continue
        
        if not search_window_start:
            return None

        # Phase 2: Fine-grained binary search (to the second)
        low, high = search_window_start, time_cursor
        transition_point = high
        binary_iterations = 0
        max_binary_iterations = 20

        while (high - low).total_seconds() > 1 and binary_iterations < max_binary_iterations:
            binary_iterations += 1
            mid = low + (high - low) / 2
            try:
                details = self._get_body_details_at_time(mid)
                # Check if either NL or SL matches the initial values
                if details['nl'] == initial_nl and details['sl'] == initial_sl:
                    low = mid
                else:
                    high, transition_point = mid, mid
            except Exception:
                break
        
        return transition_point.replace(microsecond=0)

    def generate_timeline_df(self, start_dt: datetime, duration_hours: float):
        """
        Generates the full timeline DataFrame (granular SSL level).
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

    def generate_aggregated_timeline_df(self, start_dt: datetime, duration_hours: float):
        """
        Generates an aggregated timeline DataFrame at Star Lord + Sub Lord level only.
        Each row represents a unique NL+SL combination for its complete duration.
        """
        self._end_dt_for_search = start_dt + timedelta(hours=duration_hours)
        timeline_data = []
        current_time = start_dt
        max_iterations = 1000
        iteration_count = 0
        
        while current_time < self._end_dt_for_search and iteration_count < max_iterations:
            iteration_count += 1
            start_details = self._get_body_details_at_time(current_time)
            
            # Find when EITHER NL or SL changes (not just SL)
            transition_time = self.find_next_nl_or_sl_transition(
                current_time, 
                start_details['nl'], 
                start_details['sl']
            )
            
            end_time = transition_time if transition_time and transition_time < self._end_dt_for_search else self._end_dt_for_search

            # Ensure minimum advancement to prevent infinite loops
            min_advance_time = current_time + timedelta(seconds=30)
            if end_time <= current_time:
                end_time = min(min_advance_time, self._end_dt_for_search)

            # Skip very short periods (less than 5 seconds)
            if (end_time - current_time).total_seconds() < 5:
                current_time = end_time
                continue

            # Add the complete NL+SL period as a single row
            timeline_data.append({
                'Start Time': current_time,
                'End Time': end_time,
                'NL_Planet': start_details['nl'],
                'SL_Planet': start_details['sl']
            })
            
            current_time = end_time
            
            # Final safety check
            if current_time == start_dt and iteration_count > 1:
                current_time = start_dt + timedelta(seconds=60)

        if iteration_count >= max_iterations:
            print(f"Warning: Timeline generation stopped at maximum iterations ({max_iterations})")
        
        # Convert to DataFrame first
        df = pd.DataFrame(timeline_data)
        
        if len(df) == 0:
            return df
            
        # POST-PROCESSING: Consolidate consecutive identical NL+SL combinations
        consolidated_data = []
        i = 0
        
        while i < len(df):
            current_row = df.iloc[i]
            current_start = current_row['Start Time']
            current_end = current_row['End Time']
            current_nl = current_row['NL_Planet']
            current_sl = current_row['SL_Planet']
            
            # Look ahead to find all consecutive rows with same NL+SL combination
            j = i + 1
            while j < len(df):
                next_row = df.iloc[j]
                
                # Check if next row has same NL+SL combination AND is consecutive in time
                if (next_row['NL_Planet'] == current_nl and 
                    next_row['SL_Planet'] == current_sl and
                    abs((next_row['Start Time'] - current_end).total_seconds()) <= 1):  # Allow 1 second gap for precision
                    
                    # Extend the end time to include this row
                    current_end = next_row['End Time']
                    j += 1
                else:
                    break
            
            # Add the consolidated period
            consolidated_data.append({
                'Start Time': current_start,
                'End Time': current_end,
                'NL_Planet': current_nl,
                'SL_Planet': current_sl
            })
            
            # Move to the next unprocessed row
            i = j
        
        return pd.DataFrame(consolidated_data)


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