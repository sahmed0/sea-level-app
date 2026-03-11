# -*- coding: utf-8 -*-

from scipy.stats import linregress, t
import numpy as np
import pandas as pd
import streamlit as st
import os

@st.cache_data
def getProphetForecast(df, maxPredictionYear):
    """
    If 'prophet_forecast.csv' exists, reads the pre-computed forecast.
    Otherwise, trains the Prophet model ONCE and forecasts up to the global maxPredictionYear.
    This function is cached, so it only runs the first time the app loads.
    """
    if os.path.exists("prophet_forecast.csv"):
        forecast = pd.read_csv("prophet_forecast.csv")
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        return forecast
    else:
        from prophet import Prophet
        
        # Prepare data
        data = df[['Year', 'CSIRO Adjusted Sea Level']].rename(columns={
            'Year': 'ds', 
            'CSIRO Adjusted Sea Level': 'y'
        })
        data['ds'] = pd.to_datetime(data['ds'], format='%Y').astype('datetime64[s]')

        # Fit Model
        m = Prophet(daily_seasonality=False, weekly_seasonality=False)
        m.fit(data)

        # Calculate years needed to reach 2200
        lastYear = df['Year'].max()
        yearsToPredict = maxPredictionYear - lastYear
        
        # Create forecast
        future = m.make_future_dataframe(periods=yearsToPredict, freq='YS')
        future['ds'] = future['ds'].astype('datetime64[s]')
        forecast = m.predict(future)
        
        # We only return the forecast dataframe because that's all we need for plotting
        return forecast

def getLinearPredictions(df, targetYear):
    """ Calculates linear regression points up to targetYear with 95% prediction interval."""
    x = df['Year'].values
    y = df['CSIRO Adjusted Sea Level'].values
    res = linregress(x, y)
    
    # Calculate Standard Error of the Estimate (Sigma)
    n = len(x)
    preds_hist = res.slope * x + res.intercept
    residuals = y - preds_hist
    s_err = np.sqrt(np.sum(residuals**2) / (n - 2))
    
    # Calculate uncertainty at targetYear
    # SE_pred = s_err * sqrt(1 + 1/n + (x_0 - mean_x)^2 / sum((x_i - mean_x)^2))
    x_mean = np.mean(x)
    sum_sq_x = np.sum((x - x_mean)**2)
    se_pred = s_err * np.sqrt(1 + 1/n + (targetYear - x_mean)**2 / sum_sq_x)
    
    # 95% Confidence (using T-distribution critical value)
    dof = n - 2
    t_val = t.ppf(0.975, dof)
    uncertainty = t_val * se_pred
    
    years = np.arange(x.min(), targetYear + 1)
    preds = res.slope * years + res.intercept
    targetVal = res.slope * targetYear + res.intercept
    
    return years, preds, targetVal, uncertainty

def getPolyPredictions(df, targetYear, degree=2):
    """ Calculates polynomial regression points up to targetYear with 95% prediction interval."""
    x = df['Year'].values
    y = df['CSIRO Adjusted Sea Level'].values
    
    # Fit with covariance matrix
    p, cov = np.polyfit(x, y, degree, cov=True)
    polyFunc = np.poly1d(p)
    
    # Calculate residual variance (MSE)
    n = len(x)
    preds_hist = polyFunc(x)
    residuals = y - preds_hist
    s_err = np.sqrt(np.sum(residuals**2) / (n - degree - 1))
    
    # Variance of the prediction at targetYear
    # V[y_0] = J * cov * J^T where J is the Jacobian [x_0^d, x_0^{d-1}, ..., 1]
    j = np.array([targetYear**i for i in range(degree, -1, -1)])
    v_pred = j @ cov @ j.T
    
    # Prediction uncertainty (includes noise term s_err^2)
    se_pred = np.sqrt(s_err**2 + v_pred)
    dof = n - degree - 1
    t_val = t.ppf(0.975, dof)
    uncertainty = t_val * se_pred
    
    years = np.arange(x.min(), targetYear + 1)
    preds = polyFunc(years)
    targetVal = polyFunc(targetYear)
    
    return years, preds, targetVal, uncertainty
