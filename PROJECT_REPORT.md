# Project Report

## 1. Abstract
This project demonstrates a data-science workflow for estimating landslide event likelihood from hydrological and terrain indicators. A Streamlit dashboard accepts sensor-like readings and uses a Random Forest classifier to produce an illustrative probability score. Charts summarize the sample data and model feature importance. The included dataset is synthetic and the model is not suitable for real-world hazard decisions.

## 2. Problem Statement
Mountainous regions can experience slope instability after intense or prolonged rainfall. Hydrological telemetry such as rainfall, soil moisture, and groundwater conditions can help describe changing environmental conditions. The project explores how these signals can be combined with terrain and land-cover attributes in a predictive modeling pipeline.

## 3. Objectives
- Prepare a structured dataset of hydrological and terrain features.
- Train a supervised classification model.
- Display a risk score and interpretable supporting charts.
- Simulate sensor telemetry through an interactive dashboard.
- Document the limitations and requirements for real-world deployment.

## 4. Proposed System
Sensor readings (simulated in this version) → preprocessing → Random Forest model → probability score → dashboard and exploratory visualizations.

## 5. Dataset
The project includes 2,500 synthetic rows with rainfall, soil moisture, groundwater level, slope angle, soil type, vegetation cover, human disturbance, and a synthetic event label. The generated labels are based on a simple artificial relationship and are not actual observed landslides.

## 6. Algorithms and Tools
- Python for implementation
- Pandas and NumPy for data handling
- Scikit-learn for preprocessing, train/test split, Random Forest, and evaluation
- Streamlit for the dashboard
- Plotly for interactive visualizations

## 7. Methodology
1. Load the CSV dataset.
2. Separate predictors and target.
3. Impute missing numeric/categorical values.
4. Scale numeric features and one-hot encode soil type.
5. Split into training and test sets using an 80:20 stratified split.
6. Train a Random Forest classifier.
7. Evaluate with accuracy and ROC-AUC.
8. Present the prediction and EDA results in the dashboard.

## 8. Expected Output
The dashboard shows an estimated probability, a demonstration category, sensor readings, evaluation metrics, rainfall distribution, class balance, and feature importance. Output varies with the entered telemetry.

## 9. Limitations
- Synthetic data and labels do not represent field conditions.
- No physical sensor is connected.
- No geospatial maps, time-series forecasting, or production alert service is included.
- A probability score is not a calibrated probability of a real landslide.
- Performance metrics are meaningful only for the synthetic data used here.

## 10. Future Enhancements
- Integrate ESP32/IoT sensor stations using MQTT or HTTP.
- Store timestamped data in a time-series database.
- Add location, elevation, geology, land use, and historical landslide inventories.
- Validate with temporal and location-based holdout tests.
- Compare Random Forest with XGBoost and time-series baselines.
- Add data-quality checks, model monitoring, uncertainty reporting, and expert-reviewed alert thresholds.

## 11. Conclusion
This project provides an end-to-end educational prototype showing how sensor-style environmental inputs can be processed by a supervised machine-learning model and visualized in a dashboard. Real hazard prediction requires verified field data, local validation, and expert oversight.
