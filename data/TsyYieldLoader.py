import pandas_datareader.data as web
from datetime import date as dt
import pandas as pd
import json
import os

def data_loader (start: dt, end: dt, decimals: bool=True) -> pd.DataFrame:
    """Load daily Treasury yield curve data from FRED."""
    fred_series_id = [
        'DGS1MO', 'DGS3MO', 'DGS6MO', 
        'DGS1', 'DGS2', 'DGS3', 'DGS5', 
        'DGS7', 'DGS10', 'DGS20', 'DGS30'
        ]
    source = 'fred'
    df = web.DataReader(fred_series_id, source, start, end)
    df = df.dropna() 
    if decimals: 
        df = df/100

    BASE_DIR = os.path.dirname(__file__)
    json_path = os.path.join(BASE_DIR, 'fred_tenor_map.json')
    with open(json_path, 'r') as f:
        rename_mapping = json.load(f)
    return df.rename(columns=rename_mapping)