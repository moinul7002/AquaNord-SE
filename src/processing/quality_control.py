import pandas as pd
import numpy as np
from typing import Dict

def flag_physical_outliers(df: pd.DataFrame, param_limits: Dict[str, tuple]) -> pd.DataFrame:
    """
    Flags and removes values outside of physically plausible ranges.
    param_limits: {'temperature': (-60, 50), 'precip': (0, 1000)}
    """
    for param, (low, high) in param_limits.items():
        if param in df.columns:
            # Mark values outside limits as NaN
            mask = (df[param] < low) | (df[param] > high)
            df.loc[mask, param] = np.nan
    return df

def impute_missing_linear(df: pd.DataFrame, param: str) -> pd.DataFrame:
    """
    Applies linear interpolation for small gaps in time series data.
    """
    df[param] = df[param].interpolate(method='linear', limit=3)
    return df
