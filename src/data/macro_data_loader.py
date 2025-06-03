import pandas as pd
import numpy as np
from fredapi import Fred
from typing import Optional, List, Dict
import logging
from .base_loader import BaseDataLoader

logger = logging.getLogger(__name__)

class MacroDataLoader(BaseDataLoader):
    """Loader for macroeconomic data from FRED and other sources."""
    
    FRED_SERIES_MAP = {
        'yield_10y': 'DGS10',
        'yield_2y': 'DGS2',
        'cpi': 'CPIAUCSL',
        'gdp': 'GDP',
        'unemployment': 'UNRATE',
        'vix': 'VIXCLS',
        'dollar_index': 'DTWEXBGS'
    }
    
    def __init__(self, fred_api_key: str, config: Optional[Dict] = None):
        super().__init__(config)
        self.fred = Fred(api_key=fred_api_key)
        
    def fetch_data(self, identifier: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch macro data from FRED."""
        start, end = self._validate_dates(start_date, end_date)
        
        # Check cache first
        cache_key = f"{identifier}_{start_date}_{end_date}"
        cached_data = self.get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            # Map identifier to FRED series if needed
            fred_series = self.FRED_SERIES_MAP.get(identifier, identifier)
            
            # Fetch from FRED
            data = self.fred.get_series(fred_series, start, end)
            df = pd.DataFrame(data, columns=[identifier])
            df.index.name = 'date'
            
            # Clean and cache
            df = self.clean_data(df)
            self.cache_data(cache_key, df)
            
            logger.info(f"Successfully fetched {identifier} from {start_date} to {end_date}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {identifier}: {e}")
            raise
    
    def fetch_multiple_series(self, identifiers: List[str], start_date: str, 
                            end_date: str) -> pd.DataFrame:
        """Fetch multiple macro series and combine them."""
        dfs = []
        for identifier in identifiers:
            df = self.fetch_data(identifier, start_date, end_date)
            dfs.append(df)
        
        # Combine all series
        combined = pd.concat(dfs, axis=1)
        return combined
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean macro data - handle missing values, outliers, etc."""
        # Forward fill missing values (common for macro data)
        df = df.ffill()
        
        # Remove any remaining NaN rows
        df = df.dropna()
        
        # Ensure datetime index
        df.index = pd.to_datetime(df.index)
        
        return df
    
    def resample_to_frequency(self, df: pd.DataFrame, freq: str = 'M') -> pd.DataFrame:
        """Resample data to specified frequency."""
        return df.resample(freq).last()
    
    def calculate_changes(self, df: pd.DataFrame, periods: int = 1, 
                         pct_change: bool = True) -> pd.DataFrame:
        """Calculate period-over-period changes."""
        if pct_change:
            return df.pct_change(periods)
        else:
            return df.diff(periods)