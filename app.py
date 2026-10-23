import time
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from model_utils import load_dataset, train_model, predict_risk, risk_label

st.set_page_config(page_title="Landslide Risk Intelligence", page_icon="⛰️", layout="wide")
st.title("⛰️ Landslide Risk Intelligence")
st.caption("Predictive Modeling of Landslide Risks in Mountainous Terrains Using Live Hydrological Sensor Telemetry")
st.warning("Educational prototype only. The included dataset is synthetic and predictions are not validated for real-world warnings or emergency decisions.")

data = load_dataset()
model, metrics, feature_names = train_model(data)

with st.sidebar:
    st.header("Sensor telemetry")
    st.write("Adjust the values to simulate a live sensor station.")
    live_mode = st.toggle("Simulate live telemetry", value=False)
    if live_mode:
        np.random.seed(int(time.time()) % 100000)
        rain1 = float(np.clip(np.random.normal(18, 8), 0, 100))
        rain24 = float(np.clip(rain1 * np.random.uniform(1.5, 5), 0, 300))
        moisture = float(np.clip(np.random.normal(65, 12), 5, 100))
        groundwater = float(np.clip(np.random.normal(2.4, .7), .1, 5))
        slope = float(np.clip(np.random.normal(35, 12), 5, 65))
        vegetation = float(np.clip(np.random.normal(55, 20), 0, 100))
    else:
        rain1 = st.slider("Rainfall in last hour (mm)", 0.0, 100.0, 18.0, 1.0)
        rain24 = st.slider("Rainfall in last 24 hours (mm)", 0.0, 300.0, 65.0, 5.0)
        moisture = st.slider("Soil moisture (%)", 5.0, 100.0, 62.0, 1.0)
        groundwater = st.slider("Groundwater level (m)", 0.1, 5.0, 2.1, .1)
        slope = st.slider("Slope angle (degrees)", 5.0, 65.0, 34.0, 1.0)
        vegetation = st.slider("Vegetation cover (%)", 0.0, 100.0, 55.0, 1.0)
    soil = st.selectbox("Soil type", ["Clay", "Sandy", "Loamy", "Rocky"])
    disturbance = st.selectbox("Human disturbance present?", ["No", "Yes"])
    st.caption("Simulated telemetry is generated locally; no physical sensors are connected.")

inputs = {
    "rainfall_1h_mm": rain1, "rainfall_24h_mm": rain24,
    "soil_moisture_pct": moisture, "groundwater_level_m": groundwater,
    "slope_angle_deg": slope, "soil_type": soil,
    "vegetation_cover_pct": vegetation,
    "human_disturbance": 1 if disturbance == "Yes" else 0
}
probability = predict_risk(model, inputs)
label, color, advice = risk_label(probability)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Estimated risk score", f"{probability*100:.1f}%")
m2.metric("Risk category", label)
m3.metric("Rainfall (24h)", f"{rain24:.1f} mm")
m4.metric("Soil moisture", f"{moisture:.1f}%")

st.subheader("Current assessment")
st.progress(int(probability*100))
st.markdown(f"### {label}")
st.write(advice)
st.caption(f"Last simulated assessment: {datetime.now().strftime('%d %b %Y, %H:%M:%S')}")

left, right = st.columns(2)
with left:
    st.subheader("Telemetry readings")
    telemetry = pd.DataFrame({"Sensor / feature":["Rainfall (1h)","Rainfall (24h)","Soil moisture","Groundwater level","Slope angle","Vegetation cover"],
        "Reading":[rain1,rain24,moisture,groundwater,slope,vegetation],
        "Unit":["mm","mm","%","m","degrees","%"]})
    st.dataframe(telemetry, use_container_width=True, hide_index=True)
with right:
    st.subheader("Model evaluation")
    st.write(f"**Model:** Random Forest Classifier")
    st.write(f"**Test accuracy:** {metrics['accuracy']:.3f}")
    st.write(f"**ROC AUC:** {metrics['roc_auc']:.3f}")
    st.caption("Metrics reflect only the synthetic teaching dataset and do not represent field performance.")

st.subheader("Exploratory data analysis")
c1, c2 = st.columns(2)
with c1:
    fig = px.histogram(data, x="rainfall_24h_mm", color="landslide_event",
                       barmode="overlay", nbins=30, title="24-hour rainfall distribution by event label",
                       labels={"landslide_event":"Synthetic event label","rainfall_24h_mm":"Rainfall (mm)"})
    st.plotly_chart(fig, use_container_width=True)
with c2:
    counts = data["landslide_event"].map({0:"No event",1:"Event"}).value_counts().rename_axis("Outcome").reset_index(name="Samples")
    fig = px.pie(counts, names="Outcome", values="Samples", title="Synthetic class balance")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Feature importance")
imp = pd.DataFrame({"Feature":feature_names, "Importance":model.feature_importances_}).sort_values("Importance", ascending=True)
fig = px.bar(imp, x="Importance", y="Feature", orientation="h", title="Random Forest feature importance")
st.plotly_chart(fig, use_container_width=True)

with st.expander("Project workflow"):
    st.markdown("""
    1. Hydrological and terrain features are collected or simulated.
    2. Data is cleaned and encoded.
    3. A Random Forest model is trained using labeled examples.
    4. The model estimates an event probability for the entered telemetry.
    5. A dashboard presents the score, readings, exploratory charts, and feature importance.

    **Future integration:** replace the demo generator with timestamped readings from calibrated rain gauges,
    soil-moisture probes, and groundwater sensors; validate against local landslide inventories; add drift
    monitoring, geospatial context, and an expert-reviewed alerting protocol.
    """)
