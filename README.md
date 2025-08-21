# KP AI Astrologer 🌟

A sophisticated astrological prediction system based on **Krishnamurti Paddhati (KP)** methodology for cricket match predictions and Muhurta analysis.

## 🔮 Features

### **Core Capabilities**
- **Authentic KP Muhurta Analysis**: Complete KP-based match prediction system following the KP Muhurta Promise Checklist
- **Timeline Analysis**: Minute-by-minute predictions during matches using Moon SSL transitions
- **Classical KP Rules**: Authentic implementation with significator hierarchy, debilitation and exaltation rules
- **Swiss Ephemeris Integration**: Precise astronomical calculations with ayanamsa support

### **Advanced Analytics**
- **Multi-Layer Scoring**: Star Lord (NL), Sub Lord (SL), and Sub-Sub Lord (SSL) analysis
- **Ruling Planets Analysis**: Comprehensive ruling planet calculations
- **Significator Hierarchy**: Weighted scoring based on KP Rules 1-4 (1.0, 0.5, 0.3, 0.1)
- **House Classification**: Strong houses (1,2,3,6,10,11) vs Weak houses (4,5,7,8,9,12)
- **Retrograde Detection**: Automatic retrograde planet identification and display
- **Rahu/Ketu Agency**: Advanced node analysis with agent planet logic

## 🚀 How to Use

1. **Launch Application**: Run `streamlit run app/main_dashboard.py`
2. **Enter Match Details**: Input teams, date, time, and venue
3. **Select Ayanamsa**: Choose from Krishnamurti, Lahiri, Raman, or True Citra
4. **View Analysis**: 
   - **Authentic KP Muhurta Analysis**: Complete promise-based prediction
   - **Cusp Details**: All 12 houses with significators
   - **Planetary Positions**: With scoring and retrograde indicators
   - **Moon SSL Timeline**: Detailed timeline for match duration

## 🎯 Analysis Methods

### **Authentic KP Muhurta Analysis**
Based on the KP Muhurta Promise Checklist:
- **House Groups Classification**: Strong vs Weak houses
- **Primary Promise Test**: Cuspal Sub Lords (1st, 6th, 7th, 11th)
- **Ruling Planets Analysis**: Current moment's ruling planets
- **Tie Breakers**: Retrograde impact and Moon strength
- **Special Cases**: Planet conjunctions and modifications
- **Final Verdict**: Confidence levels and practical recommendations

### **Moon SSL Timeline**
- Second-level precision SSL transitions
- Dynamic planet scoring throughout match duration
- Verdicts and comments for each time period
- Color-coded visualization based on planetary strength

## 🏏 Use Cases

- **Cricket Match Predictions**: Pre-match and live analysis
- **Muhurta Selection**: Finding auspicious timing for events
- **Astrological Research**: KP methodology exploration
- **Educational Tool**: Learning classical KP astrology principles

## 🛠️ Technology Stack

- **Frontend**: Streamlit with interactive UI
- **Backend**: Python with authentic KP calculations
- **Astronomy**: Swiss Ephemeris (PySwisseph) for precise planetary positions
- **Analytics**: Pandas, NumPy for data processing
- **Visualization**: Matplotlib for charts and color coding

## 📚 About KP Astrology

Krishnamurti Paddhati is a stellar astrology system developed by Prof. K.S. Krishnamurti. This implementation follows authentic KP principles:

- **Sub Divisional System**: 249 sub divisions of the zodiac
- **Significator Theory**: Houses signified by planets based on occupation, ownership, and star lordship
- **Ruling Planet System**: Moment of query analysis
- **Promise and Timing**: Distinguishing between what is promised vs when it happens
- **Scientific Approach**: Precise mathematical calculations and statistical validation

## 🔧 Installation

```bash
# Clone the repository
git clone <repository-url>
cd KP_AI_astrologer

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app/main_dashboard.py
```

## 📁 Project Structure

```
KP_AI_astrologer/
├── app/                    # Streamlit frontend
│   └── main_dashboard.py   # Main application interface
├── kp_core/               # Core KP calculation engine
│   ├── kp_engine.py       # Planetary calculations and chart generation
│   ├── analysis_engine.py # KP analysis and prediction logic
│   └── timeline_generator.py # SSL timeline generation
├── docs/                  # Documentation
├── match_archive/         # Saved analysis results
└── swisseph/             # Swiss Ephemeris data files
```

## 🌟 Key Features

- **Authentic KP Implementation**: Based on classical KP texts and methodology
- **Promise Checklist**: Systematic analysis following KP Muhurta principles
- **Real-time Timeline**: Live analysis during match progression
- **Multiple Ayanamsas**: Support for different sidereal calculations
- **Retrograde Handling**: Proper consideration of retrograde planetary effects
- **Hierarchical Scoring**: Weighted significator analysis
- **Visual Interface**: Intuitive Streamlit-based dashboard
- **Data Persistence**: Save and load analysis results
- **Team Management**: Flexible team name mapping and assignment 