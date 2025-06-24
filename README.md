# KP AI Astrologer: Cricket Match Predictor

## Overview

This project provides comprehensive cricket match predictions using the principles of KP (Krishnamurti Paddhati) astrology. It generates detailed Muhurta charts, Ascendant and Moon timelines based on Sub-Sub Lord (SSL) transitions, and delivers AI-driven analysis to forecast match outcomes and momentum shifts.

## Key Features

- **Muhurta Chart Generation:** Creates a detailed astrological chart for the start of the match, including planetary positions, cusps, and significators.
- **Dynamic Timelines:** Generates timelines for the Ascendant and Moon based on the precise transition times of the CSL, SL, and SSL, offering a granular view of the match's progression.
- **AI-Powered Analysis:**
    - **Muhurta Analysis:** Provides a synopsis of the match's potential outcome based on the initial chart.
    - **Ascendant Timeline Analysis:** Predicts periods of favor for each team, with specific astrological commentary on events like high scores or wickets.
    - **Moon Timeline Analysis:** Offers a secondary layer of analysis based on the Moon's transit, detailing potential momentum shifts.
- **Interactive Dashboard:** A user-friendly interface built with Streamlit to input match details and view the multi-layered predictions.

## Project Structure

```
KP_AI_Astrologer/
├── app/                  # Main application dashboard
├── kp_core/              # Core astrological calculation engine
├── config/               # Configuration files (e.g., planetary data)
├── scripts/              # Utility and testing scripts
├── docs/                 # Detailed project documentation
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## Tech Stack

- Python
- Streamlit
- Pandas
- Pyswisseph
- Pytz 