# KP Core Package
"""
Core astrological calculation engine for KP AI Astrologer
"""

__version__ = "1.0.0"

# Import main classes for easy access
from .kp_engine import KPEngine
from .analysis_engine import AnalysisEngine
from .timeline_generator import TimelineGenerator

__all__ = ['KPEngine', 'AnalysisEngine', 'TimelineGenerator'] 