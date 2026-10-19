# AeroTwin — Environmental Intelligence & Cardiorespiratory Digital Twin

## 1. Team Details

- **Team Name:** Binary Mind
- **Team Leader:** Parth Singh
- **Team Members:**
  - Parth Singh — Team Leader
  - Nandani Kumari — Team Member
- **College:** ABES Engineering College, Ghaziabad
- **Incubator:** N/A

---

## 2. Project Title

**AeroTwin: Multimodal Environmental Intelligence & In-Silico Cardiorespiratory Digital Twin**

---

## 3. Problem Statement

Ambient air pollution (specifically fine particulate matter $\text{PM}_{2.5}$ and $\text{PM}_{10}$) drives acute cardiorespiratory flare-ups, asthma exacerbations, and autonomic strain across urban populations. However, traditional municipal air quality monitors offer only population-level averages, failing to account for:
1. Individual micro-environmental schedules (indoor vs. transit exposure).
2. Underlying baseline clinical susceptibility (age, asthma, BMI, hypertension, smoking history).
3. Real-time physiological strain responses (resting pulse elevation, oxygen saturation dips, HRV suppression).

AeroTwin bridges this divide by fusing CPCB ground telemetry, atmospheric forecasting, and virtual cardiorespiratory twin modeling into an actionable, clinical-grade digital twin.

---

## 4. Healthcare Use Case

AeroTwin integrates three real-time streams to drive personalized proactive interventions:
- **🌍 Environmental Stream**: Continuous $\text{PM}_{2.5}$, $\text{PM}_{10}$, $\text{NO}_2$, $\text{AQI}$ from ground stations synchronized with Open-Meteo forecasts.
- **🫁 Static EHR Profile**: Patient susceptibility archetypes capturing chronological age, BMI, chronic asthma, and hypertension.
- **⌚ Dynamic Wearable Telemetry**: Real-time physiological signals ($\Delta\text{RHR}$, $\text{SpO}_2$, $\text{HRV}$, sleep architecture, step cadence).

### Clinical Interventions & What-If Simulation
Users can simulate micro-environmental mitigations (HEPA indoor purifiers, N95 masks, commute adjustments) and observe the simulated drop in internal inhaled dose and flare-up risk probability 24 hours before peak smog episodes occur.

> **Clinical Disclaimer:** AeroTwin is a research and educational prototype for analytics and risk stratification, not a validated medical diagnostic device.

---

## 5. Technical Stack

- **Backend & APIs:** Python 3, Flask (RESTful routing, streaming endpoints)
- **Data Engineering:** Pandas, NumPy (spatial-temporal group imputation, chronological rolling windows)
- **Machine Learning & Modeling:** Scikit-learn, Joblib, Gradient Boosted Regression Trees
- **Interactive Visualization:** Plotly.js, JetBrains Mono & Inter typography
- **Frontend Architecture:** Obsidian Dark Cyber-Clinical UI (`#070A0D` canvas, `#0D1217` surfaces, `#10B981` emerald accents)

---

## 6. AI/ML Model & Empirical Evaluation

- **Target:** 24-hour Next-Day Atmospheric $\text{PM}_{2.5}$ concentration ($\mu\text{g/m}^3$).
- **Features:** Multi-lag pollutant concentrations ($\text{PM}_{2.5}$, $\text{PM}_{10}$, $\text{NO}_2$, $\text{CO}$, $\text{SO}_2$, $\text{AQI}$), rolling 24h moving averages, rate of change, and seasonal date vectors.
- **Validation Strategy:** Chronological held-out test split with zero temporal data leakage.
- **Empirical Holdout Metrics:**
  - **Coefficient of Determination ($R^2$):** `0.8140`
  - **Mean Absolute Error (MAE):** `12.15 µg/m³`
  - **Root Mean Squared Error (RMSE):** `20.34 µg/m³`
  - **CPCB Category Agreement:** `84.2%`
- **Exposure Simulation Engine:** Physical mass-balance equation:
  $$C_{\text{inhaled}} = \frac{T_{\text{indoor}} \cdot C_{\text{ambient}} \cdot (1 - \eta_{\text{purifier}}) + T_{\text{transit}} \cdot C_{\text{ambient}} \cdot (1 - \eta_{\text{mask}}) + T_{\text{ambient}} \cdot C_{\text{ambient}}}{24}$$

---

## 7. Project Architecture & Directory Guide

```text
AeroTwin/
├── app/
│   ├── app.py                     # Flask application & REST API endpoints
│   └── templates/
│       ├── index.html             # Environmental Intelligence & Data Architecture
│       └── twin.html              # Cardiorespiratory Patient Twin interface
├── cleaning/
│   └── clean_data.py              # Ingestion, imputation & cleaning pipeline
├── data/
│   ├── healthcare/                # Synthetic patient cohorts & vitals streams
│   ├── processed/                 # Cleaned datasets ready for deployment
│   └── raw/                       # Raw source archives (.gitkeep)
├── model/
│   ├── train.py                   # Model training script
│   ├── predict.py                 # Fast inference runtime
│   ├── evaluate.py                # Holdout evaluation & metric generation
│   ├── simulation.py              # What-If personal exposure simulator
│   ├── realtime_service.py        # Live telemetry synchronization
│   ├── artifacts/                 # Saved models (.joblib) & metrics (.json)
│   └── healthcare/                # Cardiorespiratory twin engine & vitals
├── tests/
│   ├── test_step8_dashboard.py    # Integration test suite for dashboard & APIs
│   ├── test_healthcare_twin.py    # Unit tests for cardiorespiratory twin
│   └── test_csv_upload.py         # Tests for CSV upload & dynamic location
├── requirements.txt               # Dependencies
├── RUN_GUIDE.md                   # PowerShell local execution guide
└── README.md                      # Project documentation
```

---

## 8. Installation & Execution

### 1. Clone & Setup
```bash
git clone https://github.com/srivastavanandani190-lang/AeroTwin.git
cd AeroTwin
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Test Suites (34 Tests)
```bash
python -m unittest tests/test_csv_upload.py tests/test_step8_dashboard.py tests/test_healthcare_twin.py
```

### 5. Launch Application
```bash
python app/app.py
```
Access the application locally:
- **Environmental Intelligence & Data Architecture:** `http://127.0.0.1:5000/`
- **Patient Digital Twin:** `http://127.0.0.1:5000/twin`

---

## 9. Data Sources & Privacy

- **CPCB Ground Sensors:** Central Pollution Control Board ambient air quality telemetry across major Indian metropolitan centers (Delhi, Ghaziabad, Ahmedabad, Mumbai, Bengaluru, etc.).
- **Open-Meteo API:** Atmospheric weather and thermal inversion telemetry.
- **Privacy Assurance:** Zero real patient identifiable health information (PHI) is included. All patient cohorts are calibrated in-silico based on published medical literature (ICMR / RSSDI consensus).

---

## 10. Open-Source License

This project is licensed under the MIT License — see the LICENSE file for details.

---

## 11. Disclaimer

AeroTwin is an engineering prototype and research proof-of-concept developed for hackathon demonstration. It is not an FDA/CDSCO certified medical device and should not be used as a substitute for clinical diagnostics or physician care.
