import pandas as pd
import numpy as np
import streamlit as st

def safe_round_int(series, default_value=0):
    """Safely convert a pandas Series to rounded integers"""
    try:
        numeric_series = pd.to_numeric(series, errors='coerce')
        return numeric_series.round(0).astype('Int64')
    except Exception:
        return pd.Series([default_value] * len(series), dtype='Int64')

def safe_round_float(series, decimals=1, default_value=0.0):
    """Safely convert a pandas Series to rounded floats"""
    try:
        numeric_series = pd.to_numeric(series, errors='coerce')
        return numeric_series.round(decimals)
    except Exception:
        return pd.Series([default_value] * len(series), dtype=float)

def safe_percentage(series, default_value=0.0):
    """Safely convert to percentage with 1 decimal place"""
    try:
        numeric_series = pd.to_numeric(series, errors='coerce')
        return (numeric_series * 100).round(1)
    except Exception:
        return pd.Series([default_value] * len(series), dtype=float)

def preprocess_numeric_columns(df, numeric_columns):
    """Convert specified columns to numeric, handling errors gracefully"""
    df = df.copy()
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# At the beginning of functions that display DataFrames
def safe_display(df, use_container_width=True):
    try:
        st.dataframe(df, use_container_width=True)
    except:
        # Clean and try again
        df_clean = df.copy()
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
            df_clean[col] = df_clean[col].astype(str)
        st.dataframe(df_clean, use_container_width=True)
