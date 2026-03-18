# -*- coding: utf-8 -*-

import pandas as pd

def loadData() -> pd.DataFrame:
    """Load the CSIRO sea level CSV into a DataFrame.

    Returns:
        The loaded sea level data.

    Raises:
        RuntimeError: If the CSV file cannot be found.
    """
    try:
        return pd.read_csv('epa-sea-level.csv')
    except FileNotFoundError as exc:
        raise RuntimeError("'epa-sea-level.csv' not found. Ensure the file is served alongside index.html.") from exc
