#!/usr/bin/env python3
"""
KP AI Astrologer - Hugging Face Spaces Entry Point
A sophisticated astrological prediction system based on Krishnamurti Paddhati (KP) methodology.
"""

import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the main dashboard
from app.main_dashboard import main

if __name__ == "__main__":
    import streamlit as st
    
    # Add Hugging Face specific styling and header
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .hf-badge {
        position: fixed;
        top: 10px;
        right: 10px;
        background: #ff6b6b;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 12px;
        z-index: 999;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add header for Hugging Face Spaces
    st.markdown("""
    <div class="main-header">
        <h1>🌟 KP AI Astrologer</h1>
        <p>Sophisticated Cricket Match Predictions using Krishnamurti Paddhati Astrology</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add Hugging Face badge
    st.markdown("""
    <div class="hf-badge">
        🤗 Hugging Face Spaces
    </div>
    """, unsafe_allow_html=True)
    
    # Run the main application
    main() 