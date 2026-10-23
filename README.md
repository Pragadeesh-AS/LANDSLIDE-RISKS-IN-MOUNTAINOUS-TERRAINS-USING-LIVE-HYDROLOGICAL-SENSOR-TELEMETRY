# Landslide Risk Intelligence — Data Science Project

**Title:** Predictive Modeling of Landslide Risks in Mountainous Terrains Using Live Hydrological Sensor Telemetry

A runnable educational project featuring a Streamlit dashboard, synthetic hydrological/terrain data, Random Forest classification, model evaluation, exploratory data analysis, and a simulated telemetry mode.

## Important limitation

The supplied CSV is **synthetic demo data**, not field measurements. The resulting model is for learning and demonstration only. Do not use its scores for public alerts, evacuation decisions, or other real-world safety decisions. A real deployment requires calibrated sensors, verified historical landslide labels, local geological/geospatial data, validation by domain experts, and an approved warning protocol.

## Features

- Dashboard inputs for rainfall, soil moisture, groundwater level, slope, soil type, vegetation, and human disturbance
- Optional simulated “live telemetry” mode
- Random Forest classifier with preprocessing and one-hot encoding
- Estimated probability and Low / Moderate / High / Very High demo categories
- Accuracy and ROC-AUC evaluation
- EDA charts and feature importance
- Sample training dataset included

## Requirements

- Python 3.10 or newer recommended
- Internet access only for installing packages (the app itself uses local data)

## Run on Windows

1. Extract the ZIP file.
2. Open the extracted folder in VS Code.
3. Open Terminal in VS Code.
4. Create a virtual environment:

   ```bash
   py -m venv .venv
   .venv\\Scripts\\activate
   ```

5. Install packages:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

7. Open the local URL printed in the terminal, usually `http://localhost:8501`.

## Project structure

```text
landslide_risk_prediction/
├── app.py
├── model_utils.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── data/
│   └── landslide_training_data.csv
└── .gitignore
```

## Methodology

1. Load and inspect the dataset.
2. Separate input features from the target label.
3. Impute missing values and scale numeric columns.
4. One-hot encode soil type.
5. Split data into training and testing sets (80/20, stratified).
6. Train a Random Forest classifier.
7. Evaluate using accuracy and ROC-AUC.
8. Use the trained model to score dashboard inputs.

## How to make it a real sensor project

Use a microcontroller such as ESP32 with a rain gauge, capacitive soil-moisture sensor, and an appropriate groundwater/pore-pressure sensor. Send timestamped readings to a backend through MQTT or HTTP. Add sensor calibration, missing-data detection, secure transport, device identity, and a database. Do not connect this prototype to sirens or public alerts without expert validation.

## Dataset columns

- `rainfall_1h_mm`: rainfall over one hour (mm)
- `rainfall_24h_mm`: rainfall over 24 hours (mm)
- `soil_moisture_pct`: soil moisture percentage
- `groundwater_level_m`: demo groundwater level (m)
- `slope_angle_deg`: slope angle (degrees)
- `soil_type`: categorical soil type
- `vegetation_cover_pct`: vegetation cover percentage
- `human_disturbance`: 0 = no, 1 = yes
- `landslide_event`: synthetic binary target (0 = no event, 1 = event)
