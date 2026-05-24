import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Union

def save_to_csv(df: pd.DataFrame, file_path: Union[str, Path]) -> None:
    """
    Saves a DataFrame to CSV, ensuring the directory exists.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def save_to_netcdf(ds: xr.Dataset, file_path: Union[str, Path]) -> None:
    """
    Saves an Xarray Dataset to NetCDF, ensuring compliance with CF Metadata Conventions.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Applying standard CF encoding for time units
    ds.to_netcdf(path, format='NETCDF4', engine='netcdf4')

def load_from_netcdf(file_path: Union[str, Path]) -> xr.Dataset:
    """
    Loads an Xarray Dataset from NetCDF.
    """
    return xr.open_dataset(file_path)
