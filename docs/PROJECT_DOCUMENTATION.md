# Project Documentation: KP AI Astrologer

## 1. Introduction

This document outlines the architecture, features, and implementation plan for the KP AI Astrologer, a tool for predicting cricket match outcomes using KP astrology.

## 2. System Architecture

The application will be built with a modular architecture to separate concerns and facilitate focused development.

-   **Frontend (`app`):** A Streamlit-based web interface for user input (match details) and displaying the astrological analysis.
-   **Backend Core (`kp_core`):** A Python-based engine responsible for all astrological calculations.
    -   `kp_engine.py`: Handles the generation of the Muhurta chart, planetary positions, cusps, and other core astrological details.
    -   `timeline_generator.py`: Constructs the dynamic Ascendant and Moon timelines based on SSL transitions, ensuring no duplicate entries.
    -   `analysis_engine.py`: Contains the logic for interpreting the astrological data and generating predictive text (Muhurta analysis, timeline verdicts, and comments).
-   **Configuration (`config`):** Stores static data required for calculations, such as Nakshatra lords, Sub lords, etc.
-   **Scripts (`scripts`):** Contains standalone scripts for testing, data generation, or debugging specific functionalities.

## 3. Core Features & Implementation Plan

### Phase 1: Foundation and Core Logic

1.  **Project Setup (Complete):**
    -   [x] Create directory structure.
    -   [x] Initialize `README.md` and `PROJECT_DOCUMENTATION.md`.
    -   [ ] Create `requirements.txt`.

2.  **Path Correction:**
    -   Implement the `sys.path` modification at the entry point of the application (`app/main_dashboard.py`) to ensure all modules are imported correctly across the project structure.

3.  **Core KP Engine (`kp_core/kp_engine.py`):**
    -   Port and adapt the core KP functions from the reference repository.
    -   Develop functions to calculate:
        -   Ascendant and Moon longitude.
        -   Planetary positions for a given time and location.
        -   Cusp positions.
        -   Planetary lordships (Nakshatra, Sub, Sub-Sub).
        -   House significators.

### Phase 2: Timeline Generation

1.  **Timeline Generator (`kp_core/timeline_generator.py`):**
    -   Implement the logic to calculate SSL transition times with second-level precision for a given celestial body (Ascendant or Moon).
    -   The timeline should be generated for the duration of the match.
    -   Incorporate the de-duplication logic from the reference project to handle rapid transitions.
    -   The output for each interval should include:
        -   Start and End Time.
        -   Ruling Lords: Nakshatra Lord (NL), Sub Lord (SL), Sub-Sub Lord (SSL).

### Phase 3: Dashboard and UI

1.  **Main Dashboard (`app/main_dashboard.py`):**
    -   Create the Streamlit interface.
    -   **Input Sidebar:**
        -   Date of Match
        -   Time of Match (local)
        -   Timezone
        -   Location (Latitude, Longitude)
        -   Team A (Ascendant)
        -   Team B (Descendant)
    -   **Display Area:**
        -   A tabbed or sectioned layout.
        -   A placeholder for the three main prediction outputs.

### Phase 4: AI Analysis and Final Output

1.  **Analysis Engine (`kp_core/analysis_engine.py`):**
    -   **Muhurta Analysis:** Develop a function that takes the generated Muhurta chart and produces a textual analysis of which team is favored, based on KP principles (e.g., strength of Ascendant lord, cusp significations).
    -   **Timeline Analysis:** Develop a function that iterates through the generated timelines (Ascendant and Moon) and, for each interval, provides:
        -   `Verdict`: "Favors Team A", "Favors Team B", "Neutral".
        -   `Comment`: A descriptive astrological reason (e.g., "SSL is a strong significator for houses 6 & 11, indicating victory over opponents," or "SSL in the star of a planet in the 12th house may cause an unexpected loss/wicket.").

2.  **Integration and Display:**
    -   Integrate the analysis engine with the main dashboard.
    -   On form submission, trigger the full pipeline: Chart -> Timelines -> Analysis.
    -   Display the three distinct outputs clearly in the UI:
        1.  Muhurta Chart Analysis.
        2.  Ascendant CSSL Timeline Table.
        3.  Moon SSL Timeline Table. 