# -*- coding: utf-8 -*-

import pandas as pd
import streamlit as st

@st.cache_data
def loadData():
    """
    Loads the CSV data.
    
    Returns:
        pandas.DataFrame: The loaded sea level data.
    """
    try:
        df = pd.read_csv('epa-sea-level.csv')
        return df
    except FileNotFoundError:
        st.error("Error: 'epa-sea-level.csv' not found.")
        st.stop()
