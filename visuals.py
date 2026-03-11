# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def generateForecastPlot(df, targetYear, showLinear, showPoly, showProphet, linYears, linPreds, polyYears, polyPreds, displayForecast, unitMultiplier=1.0, unitSuffix="\""):
    """
    Generates the forecast plot using Matplotlib with unit scaling.
    
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Observed Data
    ax.scatter(df['Year'], df['CSIRO Adjusted Sea Level'] * unitMultiplier, color='black', label='Observed Data', s=15, alpha=0.6)

    if showLinear:
        ax.plot(linYears, linPreds * unitMultiplier, color='red', linestyle='--', linewidth=2, label='Linear Regression')

    if showPoly:
        ax.plot(polyYears, polyPreds * unitMultiplier, color='green', linestyle='--', linewidth=2, label='Polynomial Regression')

    if showProphet:
        forecastYears = displayForecast['ds'].dt.year
        ax.plot(forecastYears, displayForecast['yhat'] * unitMultiplier, color='blue', linewidth=2, label='Prophet')
        ax.fill_between(forecastYears, 
                        displayForecast['yhat_lower'] * unitMultiplier, 
                        displayForecast['yhat_upper'] * unitMultiplier, 
                        color='blue', alpha=0.1)

    ax.set_xlabel('Year')
    ax.set_ylabel(f'Sea Level ({unitSuffix.strip()})')
    ax.set_title(f'Forecast Models up to {targetYear}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig
