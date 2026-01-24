# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress
from prophet import Prophet
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1. SETUP & HELPER FUNCTIONS
# ------------------------------------------------------------------------------

st.set_page_config(page_title="Sea Level Predictor", layout="centered")
MAX_PREDICTION_YEAR = 2100

@st.cache_data
def load_data():
    """ Loads the CSV data."""
    df = pd.read_csv('epa-sea-level.csv')
    return df

@st.cache_data
def get_prophet_forecast(df):
    """
    Trains the Prophet model ONCE and forecasts up to the global MAX_PREDICTION_YEAR (2100).
    This function is cached, so it only runs the first time the app loads.
    """
    # Prepare data
    data = df[['Year', 'CSIRO Adjusted Sea Level']].rename(columns={
        'Year': 'ds', 
        'CSIRO Adjusted Sea Level': 'y'
    })
    data['ds'] = pd.to_datetime(data['ds'], format='%Y')

    # Fit Model
    m = Prophet(daily_seasonality=False, weekly_seasonality=False)
    m.fit(data)

    # Calculate years needed to reach 2100
    last_year = df['Year'].max()
    years_to_predict = MAX_PREDICTION_YEAR - last_year
    
    # Create forecast
    future = m.make_future_dataframe(periods=years_to_predict, freq='YS')
    forecast = m.predict(future)
    
    # We only return the forecast dataframe because that's all we need for plotting
    return forecast

def get_linear_predictions(df, target_year):
    """ Calculates linear regression points up to target_year."""
    x = df['Year']
    y = df['CSIRO Adjusted Sea Level']
    res = linregress(x, y)
    
    years = np.arange(x.min(), target_year + 1)
    preds = res.slope * years + res.intercept
    target_val = res.slope * target_year + res.intercept
    return years, preds, target_val

def get_poly_predictions(df, target_year, degree=2):
    """ Calculates polynomial regression points up to target_year."""
    x = df['Year']
    y = df['CSIRO Adjusted Sea Level']
    coefs = np.polyfit(x, y, degree)
    poly_func = np.poly1d(coefs)
    
    years = np.arange(x.min(), target_year + 1)
    preds = poly_func(years)
    target_val = poly_func(target_year)
    return years, preds, target_val

# ------------------------------------------------------------------------------
# 2. THE APP INTERFACE
# ------------------------------------------------------------------------------

st.title("🌊 Sea Level Predictor")
st.markdown("""
Compare **Prophet**, **Linear Regression**, and **Polynomial Regression** models to forecast sea level rise based on CSIRO historical data.
""")

try:
    df = load_data()
    
except FileNotFoundError:
    st.error("Error: 'epa-sea-level.csv' not found.")
    st.stop()

# --- 1. PRE-CALCULATE PROPHET (Cached) ---
# This runs once. Subsequent slider moves will just retrieve this variable instantly.
with st.spinner('Initializing models...'):
    full_prophet_forecast = get_prophet_forecast(df)

# --- 2. USER INPUTS ---
st.divider()
st.subheader("Prediction Settings")
target_year = st.slider("Select Target Year", min_value=2024, max_value=MAX_PREDICTION_YEAR, value=2050)

# --- 3. FILTER DATA FOR DISPLAY ---
# Slice the Prophet dataframe to only show up to the user's target year
# Use the cached 'full_prophet_forecast' and just filter it.
target_date_limit = pd.Timestamp(year=target_year, month=12, day=31)
display_forecast = full_prophet_forecast[full_prophet_forecast['ds'] <= target_date_limit]

# Get the specific prediction row for the target year
target_date_start = pd.Timestamp(year=target_year, month=1, day=1)
prophet_row = full_prophet_forecast[full_prophet_forecast['ds'] == target_date_start]

# Safe extraction of values
if not prophet_row.empty:
    prophet_pred = prophet_row['yhat'].values[0]
    prophet_lower = prophet_row['yhat_lower'].values[0]
    prophet_upper = prophet_row['yhat_upper'].values[0]
else:
    prophet_pred = 0
    prophet_lower = 0
    prophet_upper = 0

# Calculate Simple Models (Fast enough to run on-the-fly)
lin_years, lin_preds, lin_target_val = get_linear_predictions(df, target_year)
poly_years, poly_preds, poly_target_val = get_poly_predictions(df, target_year)

# ------------------------------------------------------------------------------
# 4. VISUALISATION
# ------------------------------------------------------------------------------

st.markdown(f"### Predictions for {target_year}")

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Facebook Prophet", f"{prophet_pred:.2f}\"", f"± {(prophet_upper - prophet_lower)/2:.2f} inches uncertainty", delta_color="blue")
col2.metric("Linear Regression", f"{lin_target_val:.2f}\"", "Oversimplified", delta_color="red")
col3.metric("Polynomial (Deg 2)", f"{poly_target_val:.2f}\"", "Approximate", delta_color="green")

st.subheader("Model Comparison Plot")

# Toggles
st.write("Select models to display:")
c1, c2, c3 = st.columns(3)
show_prophet = c1.toggle('Show Prophet (Blue)', value=True)
show_linear = c2.toggle('Show Linear (Red)', value=True)
show_poly = c3.toggle('Show Polynomial (Green)', value=True)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Observed Data
ax.scatter(df['Year'], df['CSIRO Adjusted Sea Level'], color='black', label='Observed Data', s=15, alpha=0.6)

if show_linear:
    ax.plot(lin_years, lin_preds, color='red', linestyle='--', linewidth=2, label='Linear Regression')

if show_poly:
    ax.plot(poly_years, poly_preds, color='green', linestyle='--', linewidth=2, label='Polynomial Regression')

if show_prophet:
    forecast_years = display_forecast['ds'].dt.year
    ax.plot(forecast_years, display_forecast['yhat'], color='blue', linewidth=2, label='Prophet')
    ax.fill_between(forecast_years, 
                    display_forecast['yhat_lower'], 
                    display_forecast['yhat_upper'], 
                    color='blue', alpha=0.15)

ax.set_xlabel('Year')
ax.set_ylabel('Sea Level (inches)')
ax.set_title(f'Forecast Models up to {target_year}')
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)

st.divider()
st.markdown(
    """
    <div style='text-align: center; color: grey;'>
        Created by <b>Sajid Ahmed</b> | <a href='https://github.com/sahmed0'>GitHub</a>
    </div>
    """, 
    unsafe_allow_html=True
)


