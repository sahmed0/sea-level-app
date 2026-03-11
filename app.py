# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd

from data import loadData
from models import getProphetForecast, getLinearPredictions, getPolyPredictions
from visuals import generateForecastPlot
from config import MAX_PREDICTION_YEAR

# ------------------------------------------------------------------------------
# 1. SETUP & HELPER FUNCTIONS
# ------------------------------------------------------------------------------

st.set_page_config(page_title="Sea Level Predictor", layout="wide")

# Unit conversion constants from Inches (Base)
UNIT_CONVERSIONS = {
    "Inches": {"multiplier": 1.0, "suffix": "\""},
    "Centimetres": {"multiplier": 2.54, "suffix": " cm"},
    "Metres": {"multiplier": 0.0254, "suffix": " m"}
}

# ------------------------------------------------------------------------------
# 2. THE APP INTERFACE
# ------------------------------------------------------------------------------

st.title("🌊 Sea Level Predictor")
st.markdown("""
Compare **Prophet**, **Linear Regression**, and **Polynomial Regression** models to forecast sea level rise based on CSIRO historical data.
""")

df = loadData()

# --- 1. PRE-CALCULATE PROPHET (Cached) ---
with st.spinner('Initialising models...'):
    fullProphetForecast = getProphetForecast(df, MAX_PREDICTION_YEAR)

st.divider()

# --- 2. LAYOUT: SIDE-BY-SIDE ---
settingsCol, displayCol = st.columns([1, 3], gap="large")

with settingsCol:
    st.subheader("Control Panel")
    
    st.markdown("### Prediction Settings")
    targetYear = st.slider(
        "Select Target Year", 
        min_value=2024, 
        max_value=MAX_PREDICTION_YEAR, 
        value=2050,
        help="Adjust the slider to see predictions for a specific future year."
    )

    st.divider()
    
    st.markdown("### Measurement Units")
    unitName = st.radio(
        "Select Unit",
        options=list(UNIT_CONVERSIONS.keys()),
        index=0,
        help="Choose the unit for sea level values."
    )
    unitMult = UNIT_CONVERSIONS[unitName]["multiplier"]
    unitSuffix = UNIT_CONVERSIONS[unitName]["suffix"]

    st.divider()
    
    st.markdown("### Model Selection")
    st.write("Toggle models on the plot:")
    showProphet = st.toggle('Facebook Prophet (Blue)', value=True)
    showLinear = st.toggle('Linear Regression (Red)', value=True)
    showPoly = st.toggle('Polynomial (Green)', value=True)

with displayCol:
    # --- 3. CALCULATIONS ---
    # Prophet filtering
    targetDateLimit = pd.Timestamp(year=targetYear, month=12, day=31)
    displayForecast = fullProphetForecast[fullProphetForecast['ds'] <= targetDateLimit]
    
    targetDateStart = pd.Timestamp(year=targetYear, month=1, day=1)
    prophetRow = fullProphetForecast[fullProphetForecast['ds'] == targetDateStart]
    
    if not prophetRow.empty:
        prophetPred = prophetRow['yhat'].values[0]
        prophetLower = prophetRow['yhat_lower'].values[0]
        prophetUpper = prophetRow['yhat_upper'].values[0]
    else:
        prophetPred, prophetLower, prophetUpper = 0, 0, 0

    # Simple Models
    linYears, linPreds, linTargetVal, linUnc = getLinearPredictions(df, targetYear)
    polyYears, polyPreds, polyTargetVal, polyUnc = getPolyPredictions(df, targetYear)

    # --- 4. VISUALISATION ---
    st.subheader(f"Results for {targetYear}")

    # Metrics
    mCol1, mCol2, mCol3 = st.columns(3)
    
    # Scale values for display
    prophetDisp = prophetPred * unitMult
    prophetUncDisp = (prophetUpper - prophetLower) / 2 * unitMult
    linDisp = linTargetVal * unitMult
    linUncDisp = linUnc * unitMult
    polyDisp = polyTargetVal * unitMult
    polyUncDisp = polyUnc * unitMult

    mCol1.metric("Prophet", f"{prophetDisp:.2f}{unitSuffix}", f"± {prophetUncDisp:.2f}{unitSuffix}")
    mCol2.metric("Linear", f"{linDisp:.2f}{unitSuffix}", f"± {linUncDisp:.2f}{unitSuffix}")
    mCol3.metric("Polynomial", f"{polyDisp:.2f}{unitSuffix}", f"± {polyUncDisp:.2f}{unitSuffix}")

    st.markdown("#### Model Comparison Plot")
    fig = generateForecastPlot(
        df, targetYear, showLinear, showPoly, showProphet, 
        linYears, linPreds, polyYears, polyPreds, displayForecast,
        unitMult, unitSuffix
    )
    st.pyplot(fig, use_container_width=True)

st.divider()
st.markdown(
    """
    <div style='text-align: center; color: grey;'>
        Created by <b>Sajid Ahmed</b> | <a href='https://github.com/sahmed0/sea-level-app.git'>GitHub</a>
    </div>
    """, 
    unsafe_allow_html=True
)
