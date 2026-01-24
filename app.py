# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1. SETUP & HELPER FUNCTIONS
# ------------------------------------------------------------------------------

st.set_page_config(page_title="Sea Level Predictor", layout="centered")

@st.cache_data
def load_data():
    """Loads the CSV data and caches it for speed."""
    # Ensure the CSV file is in the same folder as this script
    df = pd.read_csv('epa-sea-level.csv')
    return df

def train_prophet_model(df, years_to_predict):
    """
    Trains a Prophet model and forecasts 'years_to_predict' into the future.
    """
    # Prepare data for Prophet (rename columns to 'ds' and 'y')
    data = df[['Year', 'CSIRO Adjusted Sea Level']].rename(columns={
        'Year': 'ds', 
        'CSIRO Adjusted Sea Level': 'y'
    })
    data['ds'] = pd.to_datetime(data['ds'], format='%Y')

    # Initialise and fit model
    m = Prophet(daily_seasonality=False, weekly_seasonality=False)
    m.fit(data)

    # Create future dataframe
    future = m.make_future_dataframe(periods=years_to_predict, freq='YS')
    
    # Predict
    forecast = m.predict(future)
    return m, forecast

# ------------------------------------------------------------------------------
# 2. THE APP INTERFACE
# ------------------------------------------------------------------------------

st.title("🌊 Sea Level Predictor")
st.markdown("""
This web app uses **Facebook Prophet** to forecast sea level rises using historical data from CSIRO.
Use the controls below to adjust the prediction timeline.
""")

# Load data
try:
    df = load_data()
    st.success("Data loaded successfully!", icon="✅")
except FileNotFoundError:
    st.error("Error: 'epa-sea-level.csv' not found. Please place it in the same folder.")
    st.stop()

# Year Slider Controls
st.divider() # Adds a nice visual separator
st.subheader("Prediction Settings")
target_year = st.slider("Select Target Year", min_value=2024, max_value=2100, value=2050)
# ---------------------------------------------------


# Calculate how many years to predict from the last data point
last_year_in_data = df['Year'].max()
years_to_predict = target_year - last_year_in_data

if years_to_predict > 0:
    with st.spinner('Training model...'):
        model, forecast = train_prophet_model(df, years_to_predict)

    # --------------------------------------------------------------------------
    # 3. VISUALISATION
    # --------------------------------------------------------------------------
    
    # Extract the specific prediction for the target year
    # The forecast dataframe has a 'ds' column with dates. We find the row matching our target year.
    target_date = pd.Timestamp(year=target_year, month=1, day=1)
    prediction_row = forecast[forecast['ds'] == target_date]
    
    if not prediction_row.empty:
        predicted_level = prediction_row['yhat'].values[0]
        lower_bound = prediction_row['yhat_lower'].values[0]
        upper_bound = prediction_row['yhat_upper'].values[0]

        # Display Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Target Year", target_year)
        col2.metric("Predicted Rise", f"{predicted_level:.2f} inches")
        col3.metric("Uncertainty", f"± {(upper_bound - lower_bound)/2:.2f}")

    # Plotting
    st.subheader("Forecast Plot")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot actual data
    ax.scatter(df['Year'], df['CSIRO Adjusted Sea Level'], color='black', label='Observed Data', s=10)

    # Plot forecast
    # Convert the 'ds' column back to Year integers for the X-axis to match the scatter plot style
    forecast_years = forecast['ds'].dt.year
    ax.plot(forecast_years, forecast['yhat'], color='blue', label='Prophet Forecast')
    
    # Uncertainty Interval
    ax.fill_between(forecast_years, 
                    forecast['yhat_lower'], 
                    forecast['yhat_upper'], 
                    color='blue', alpha=0.2, label='Confidence Interval')

    ax.set_xlabel('Year')
    ax.set_ylabel('Sea Level (inches)')
    ax.set_title(f'Sea Level Rise Prediction to {target_year}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Pass the Matplotlib figure to Streamlit
    st.pyplot(fig)

    # Show raw data
    with st.expander("View Forecast Data"):
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

else:

    st.warning("Please select a year in the future.")

