# -*- coding: utf-8 -*-

"""Matplotlib chart generation for the Sea Level Predictor."""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def generateForecastPlot(
    df: pd.DataFrame,
    targetYear: int,
    showLinear: bool,
    showPoly: bool,
    showProphet: bool,
    linYears: np.ndarray,
    linPreds: np.ndarray,
    polyYears: np.ndarray,
    polyPreds: np.ndarray,
    displayForecast: pd.DataFrame,
    unitMultiplier: float = 1.0,
    unitSuffix: str = '"',
) -> matplotlib.figure.Figure:
    """Generate the model comparison plot.

    Args:
        df: The raw historical sea level DataFrame.
        targetYear: The forecast end year selected by the user.
        showLinear: Whether to render the Linear Regression trace.
        showPoly: Whether to render the Polynomial Regression trace.
        showProphet: Whether to render the Prophet forecast trace.
        linYears: x-values for the linear regression line.
        linPreds: y-values for the linear regression line.
        polyYears: x-values for the polynomial regression curve.
        polyPreds: y-values for the polynomial regression curve.
        displayForecast: Prophet forecast DataFrame filtered to targetYear.
        unitMultiplier: Scale factor to convert from inches to the chosen unit.
        unitSuffix: Unit string appended to axis labels.

    Returns:
        The rendered Matplotlib figure.
    """
    
    # 1. Initialize plot and set backgrounds to transparent
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_alpha(0.0) 
    ax.patch.set_alpha(0.0)

    # Observed historical data (Changed to semi-transparent white)
    ax.scatter(
        df['Year'],
        df['CSIRO Adjusted Sea Level'] * unitMultiplier,
        color='#ffffff', label='Observed Data', s=15, alpha=0.5, zorder=3,
    )

    if showLinear:
        ax.plot(
            linYears, linPreds * unitMultiplier,
            color='#ff3366', linestyle='--', linewidth=2, label='Linear Regression', # Updated Hex
        )

    if showPoly:
        ax.plot(
            polyYears, polyPreds * unitMultiplier,
            color='#00e676', linestyle='--', linewidth=2, label='Polynomial', # Updated Hex
        )

    if showProphet:
        forecastYears = displayForecast['ds'].dt.year
        ax.plot(
            forecastYears, displayForecast['yhat'] * unitMultiplier,
            color='#00f2fe', linewidth=2, label='Prophet', # Updated Hex
        )
        ax.fill_between(
            forecastYears,
            displayForecast['yhat_lower'] * unitMultiplier,
            displayForecast['yhat_upper'] * unitMultiplier,
            color='#00f2fe', alpha=0.15,
        )

    # 2. Update labels, title, and ticks to crisp white
    ax.set_xlabel('Year', color='white', fontsize=10)
    ax.set_ylabel(f'Sea Level ({unitSuffix.strip()})', color='white', fontsize=10)
    ax.set_title(f'Forecast Models up to {targetYear}', color='white', fontsize=12, pad=12)
    ax.tick_params(colors='white')
    
    # 3. Glass borders (Spines) and Grid
    for spine in ax.spines.values():
        spine.set_edgecolor('#ffffff')
        spine.set_alpha(0.2)

    ax.legend(
        framealpha=0.1, edgecolor=(1, 1, 1, 0.2),
        labelcolor='white', fontsize=9,
    )
    ax.grid(True, alpha=0.1, color='white', linestyle='--')

    fig.tight_layout()
    return fig
