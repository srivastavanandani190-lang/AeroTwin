# AeroTwin — Windows PowerShell Run Guide

This guide provides step-by-step instructions to run the **AeroTwin** Digital Twin and Air Quality Analytics dashboard locally on Windows PowerShell.

---

## 1. Open PowerShell in Project Directory

Open Windows PowerShell and navigate to the project directory:

```powershell
Set-Location -Path "c:\Users\sriva\OneDrive\Desktop\TWINLIGHT"
```

Verify you are in the correct root directory:
```powershell
Get-Item .
```

---

## 2. Create a Virtual Environment (Recommended)

If you haven't created a virtual environment yet, create one using Python 3:

```powershell
python -m venv venv
```

---

## 3. Activate the Environment

Activate the virtual environment in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

> **PowerShell Execution Policy Tip:** If you encounter `Script Execution Disabled` error, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
> .\venv\Scripts\Activate.ps1
> ```

---

## 4. Install Dependencies

Install the required Python packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To quickly verify that the core libraries are installed:
```powershell
python -c "import flask, pandas, numpy, plotly, sklearn, joblib; print('All core libraries verified successfully!')"
```

---

## 5. Configure Environment Variables (Optional)

AeroTwin comes with `.env.example`. You can copy it to `.env`:

```powershell
Copy-Item .env.example .env
```

The application runs out of the box with default settings:
- `PORT=5000`
- `SECRET_KEY=aerotwin-vibrant-crystal-secret-key-2026`
- `FLASK_DEBUG=0`
- `OPENAQ_API_KEY`: Optional; only required if fetching live OpenAQ station feeds.

---

## 6. Run Automated Verification Tests

Before launching the server, you can run the test suites to ensure everything is functional:

```powershell
# 1. Run physical simulation tests (9 tests)
python -m unittest discover -s model -p "test*.py"

# 2. Run Environmental dashboard & integration tests (15 tests)
python -m unittest tests/test_step8_dashboard.py

# 3. Run Healthcare Digital Twin test suite (13 tests)
python -m unittest tests/test_healthcare_twin.py

# Or run all dashboard and healthcare integration tests together (28 tests):
python -m unittest tests/test_step8_dashboard.py tests/test_healthcare_twin.py
```

Expected output:
```text
Ran 28 tests in ~3.2s ... OK
```

---

## 7. Start the AeroTwin Application

Launch the Flask development server:

```powershell
python app/app.py
```

You will see:
```text
============================================================
    AEROTWIN - VIBRANT BLACK CRYSTAL ANALYTICS DASHBOARD    
============================================================
Server running locally at: http://127.0.0.1:5000/
Press Ctrl+C to terminate.
============================================================
```

---

## 8. Open the Digital Twin in Your Browser

- **Healthcare Digital Twin (Type 2 Diabetes)**:
  👉 [http://127.0.0.1:5000/twin](http://127.0.0.1:5000/twin)

- **Environmental Intelligence Studio (Atmospheric & Exposure)**:
  👉 [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 9. Hackathon Demo Workflow (Happiest Health Summit 2026)

1. Navigate to `http://127.0.0.1:5000/twin`.
2. Select a synthetic patient from the top dropdown (e.g., `SYN-T2D-001 - Aarav Sharma`).
3. Review Section 2 (**EHR & Medical History**) to inspect baseline HbA1c (8.4%), fasting glucose, and active medications (Metformin, Glimepiride).
4. In Section 3 (**Dynamic Wearable Stream**), click **"Play Replay"** or **"Step (+5m)"** to observe the continuous CGM time-series stream advance.
5. In Section 4 (**Prediction Engine**), observe the real-time **120-minute excursion probability** and the projected continuous trajectory curve updating with each step.
6. As the stream replay reaches the postprandial period, watch the **Ground Truth Validation banner** confirm the actual observed peak against the prediction.
7. In Section 5 (**Model Evaluation**), review the empirical confusion matrix ($N = 2,760$ held-out test windows) and performance gains ($56.8\%$ RMSE improvement over persistence).
8. In Section 6 (**What-If Simulation**), adjust the post-meal walking duration slider (e.g., 30 minutes) and watch the simulated glucose curve attenuate the peak below the 180 mg/dL threshold.
9. In Section 7 (**Environmental Context**), inspect ambient PM2.5 levels contextualized as an extrinsic mediator of insulin resistance.
10. In Section 8 (**Data Quality & Governance**), verify that all synthetic data boundaries and clinical disclaimers are transparently reported.

Open your web browser and navigate to:

```text
http://127.0.0.1:5000/
```

### Useful Quick Links:
- **Default Dashboard (Pan-India City Telemetry)**: `http://127.0.0.1:5000/`
- **Delhi Station Telemetry & 24h Forecaster**: `http://127.0.0.1:5000/?dataset=cleaned_city_day.csv&city=Delhi`
- **Station-Level Micro Data**: `http://127.0.0.1:5000/?dataset=cleaned_station_day.csv`
- **Live Real-Time Region Stream**: `http://127.0.0.1:5000/api/realtime/connect?city=Delhi`
- **What-If Simulation Studio**: Scroll down to the `#twin-studio` section on the homepage.
- **REST API - Live Telemetry Status**: `http://127.0.0.1:5000/api/realtime/status?city=Delhi`
- **REST API - Live Telemetry Refresh**: `http://127.0.0.1:5000/api/realtime/refresh?city=Delhi`
- **REST API - Simulation**: `http://127.0.0.1:5000/api/simulate?ambient_pm25=120&indoor_hours=18&purifier_reduction=80`
- **REST API - Forecast**: `http://127.0.0.1:5000/api/forecast?pm25=110&aqi=230&city=Delhi`
- **REST API - Model Info**: `http://127.0.0.1:5000/api/model-info`

---

## 8.1. Custom Dataset Upload & Real-Time Region Features

Beside the **Export Cleaned CSV** button in the top filter toolbar, three powerful new capabilities are now available:

1. **Upload Dataset (`[Upload Dataset]`)**:
   - Upload any tabular dataset (CSV, TSV, or Excel `.xlsx`/`.xls`).
   - Automatically cleans column headers and computes instant summary metrics, data quality profiles, dynamic distributions, and cross-correlation matrices.
   - All columns are instantly available in the **Interactive Custom Chart Studio**.
2. **Connect Real-Time (`[Connect Real-Time]`)**:
   - Queries live satellite and ground-sensor feeds (Open-Meteo Air Quality & Weather API) for whichever city or region is selected in the **REGION / CITY** dropdown.
   - Synthesizes a real-time hourly stream with current PM2.5, PM10, AQI, temperature, and relative humidity.
3. **Refresh Live (`[Refresh Live]`)**:
   - Forces an immediate telemetry update from live sensors and re-renders the dashboard with zero delay.


---

## 9. Stop the Server Safely

To stop the Flask development server, press:

```text
Ctrl + C
```

in the active PowerShell terminal window.

---

## 10. Troubleshooting Common Issues

### Issue A: Port 5000 is already in use
**Symptom:** `OSError: [Errno 10048] address already in use` or another app is listening on port 5000.  
**Solution:** AeroTwin includes automatic open-port fallback (tries 5000, 5001, 8080). You can also explicitly specify a custom port in PowerShell:
```powershell
$env:PORT = "5050"
python app/app.py
```
Then visit `http://127.0.0.1:5050/`.

### Issue B: "Script Execution Disabled" when activating venv
**Symptom:** `File ...\Activate.ps1 cannot be loaded because running scripts is disabled on this system.`  
**Solution:** Enable execution for the current terminal process:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1
```

### Issue C: Processed datasets not found
**Symptom:** Webpage shows `No cleaned datasets were found in data/processed/`.  
**Solution:** Run the data cleaning pipeline to generate all processed CSVs:
```powershell
python cleaning/clean_data.py
```

### Issue D: Model artifact not found
**Symptom:** Model section displays `Model artifact not found`.  
**Solution:** Retrain the model on the processed dataset:
```powershell
python model/train.py
```
This will generate `model/artifacts/pm25_forecaster.joblib` and `model/artifacts/model_metrics.json`.
