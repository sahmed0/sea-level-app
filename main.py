# -*- coding: utf-8 -*-

"""PyScript entry point for the Sea Level Predictor application.

Orchestrates data loading, model predictions, and DOM updates in response
to user interaction events. Uses asyncio so the Pyodide event loop does not
block while importing heavy scientific modules.
"""

import js
import asyncio
from pyodide.ffi import create_proxy


def _log(message: str, level: str = 'info') -> None:
    """Write a prefixed log line to the browser console.

    Args:
        message: The text to log.
        level: One of 'info', 'error', or 'success'.
    """
    prefix = '[SEA-LEVEL]'
    if level == 'error':
        js.console.error(f'{prefix} ❌ {message}')
    else:
        js.console.log(f'{prefix} {message}')


_log('main.py started.')


async def init_app() -> None:
    """Initialise the application asynchronously.

    Imports all scientific modules after Pyodide is ready, wires up DOM event
    listeners, performs the initial render, and then hides the loading overlay.
    """
    from pyscript import document, display

    try:
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        _log('Scientific modules ready.')
    except Exception as exc:
        _log(f'Module import failure: {exc}', level='error')
        document.getElementById('loading-text').textContent = f'Error: {exc}'
        return

    try:
        from data import loadData
        from models import getProphetForecast, getLinearPredictions, getPolyPredictions
        from visuals import generateForecastPlot
        _log('Internal modules ready.')
    except Exception as exc:
        _log(f'Internal module failure: {exc}', level='error')
        document.getElementById('loading-text').textContent = f'Error: {exc}'
        return

    # Unit conversion constants from Inches (base unit in the dataset).
    UNIT_CONVERSIONS = {
        'Inches':      {'multiplier': 1.0,    'suffix': '"'},
        'Centimetres': {'multiplier': 2.54,   'suffix': ' cm'},
        'Metres':      {'multiplier': 0.0254, 'suffix': ' m'},
    }

    # -----------------------------------------------------------------------
    # Load data once on startup
    # -----------------------------------------------------------------------
    try:
        df = loadData()
        full_prophet_forecast = getProphetForecast()
        _log('Data loaded.', level='success' if True else 'info')
    except RuntimeError as exc:
        _log(str(exc), level='error')
        document.getElementById('loading-text').textContent = f'Error: {exc}'
        return

    # -----------------------------------------------------------------------
    # Inner helpers — defined inside init_app so they close over shared state
    # -----------------------------------------------------------------------

    def _get_settings() -> dict:
        """Read the current values of all UI controls.

        Returns:
            A dict with targetYear, unitMultiplier, unitSuffix, showProphet,
            showLinear, and showPoly.
        """
        target_year = int(document.getElementById('year-slider').value)
        unit_name = document.querySelector('input[name="unit"]:checked').value
        conv = UNIT_CONVERSIONS[unit_name]
        return {
            'targetYear':     target_year,
            'unitMultiplier': conv['multiplier'],
            'unitSuffix':     conv['suffix'],
            'showProphet':    document.getElementById('toggle-prophet').checked,
            'showLinear':     document.getElementById('toggle-linear').checked,
            'showPoly':       document.getElementById('toggle-poly').checked,
        }

    def _update_metric(element_id: str, value: float, uncertainty: float, suffix: str) -> None:
        """Update a single metric card's displayed value and uncertainty.

        Args:
            element_id: The DOM id of the metric value element.
            value: The central prediction value.
            uncertainty: Half-width of the 95% prediction interval.
            suffix: Unit suffix string to append to each number.
        """
        document.getElementById(element_id).innerHTML = (
            f'{value:.2f}{suffix}'
            f'<span class="metric-uncertainty">± {uncertainty:.2f}{suffix}</span>'
        )

    def update_dashboard(event=None) -> None:
        """Recalculate predictions and refresh all UI elements.

        This is the single reconciler triggered by every control change. It
        reads current settings atomically, runs the models, and redraws the
        chart and metric cards.

        Args:
            event: The DOM event object passed by addEventListener (unused).
        """
        s = _get_settings()
        target_year = s['targetYear']
        mult = s['unitMultiplier']
        suffix = s['unitSuffix']

        document.getElementById('year-display').textContent = str(target_year)

        # --- Prophet filtering ---
        target_limit = pd.Timestamp(year=target_year, month=12, day=31)
        display_forecast = full_prophet_forecast[full_prophet_forecast['ds'] <= target_limit]

        target_start = pd.Timestamp(year=target_year, month=1, day=1)
        prophet_row = full_prophet_forecast[full_prophet_forecast['ds'] == target_start]

        if not prophet_row.empty:
            prophet_pred = prophet_row['yhat'].values[0]
            prophet_unc = (prophet_row['yhat_upper'].values[0] - prophet_row['yhat_lower'].values[0]) / 2
        else:
            prophet_pred, prophet_unc = 0.0, 0.0

        # --- Regression models ---
        lin_years, lin_preds, lin_val, lin_unc = getLinearPredictions(df, target_year)
        poly_years, poly_preds, poly_val, poly_unc = getPolyPredictions(df, target_year)

        # --- Update metric cards ---
        _update_metric('metric-prophet', prophet_pred * mult, prophet_unc * mult, suffix)
        _update_metric('metric-linear',  lin_val * mult,      lin_unc * mult,     suffix)
        _update_metric('metric-poly',    poly_val * mult,     poly_unc * mult,    suffix)

        # --- Redraw chart ---
        plt.close('all')
        fig = generateForecastPlot(
            df, target_year,
            s['showLinear'], s['showPoly'], s['showProphet'],
            lin_years, lin_preds,
            poly_years, poly_preds,
            display_forecast,
            mult, suffix,
        )
        document.getElementById('chart-container').innerHTML = ''
        display(fig, target='chart-container', append=False)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Wire up event listeners
    # -----------------------------------------------------------------------
    proxy = create_proxy(update_dashboard)
    listener_ids = [
        'year-slider',
        'toggle-prophet',
        'toggle-linear',
        'toggle-poly',
    ]
    for element_id in listener_ids:
        document.getElementById(element_id).addEventListener('input', proxy)

    # Radio buttons need separate wiring as querySelector only returns one node.
    for radio in document.querySelectorAll('input[name="unit"]'):
        radio.addEventListener('change', proxy)

    # Initial render before revealing the app.
    update_dashboard()

    # Delegate loader hide to the JS bridge so the transition is handled by CSS.
    js.window.hideLoader()
    _log('Application ready.')


asyncio.ensure_future(init_app())
