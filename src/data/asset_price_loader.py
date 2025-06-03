import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Optional, Dict
import logging
from .base_loader import BaseDataLoader

logger = logging.getLogger(__name__)

class AssetPriceLoader(BaseDataLoader):
    """Loader for asset price data from various sources."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
    def fetch_data(self, ticker: str, start_date: str, end_date: str,
                   data_type: str = 'Close') -> pd.DataFrame:
        """Fetch asset price data from Yahoo Finance."""
        start, end = self._validate_dates(start_date, end_date)
        
        # Check cache
        cache_key = f"{ticker}_{start_date}_{end_date}_{data_type}"
        cached_data = self.get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            # Download data
            data = yf.download(ticker, start=start, end=end, progress=False)
            
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Extract specific price type
            if data_type in data.columns:
                df = pd.DataFrame(data[data_type])
                df.columns = [ticker]
            else:
                df = pd.DataFrame(data['Close'])
                df.columns = [ticker]
            
            # Clean and cache
            df = self.clean_data(df)
            self.cache_data(cache_key, df)
            
            logger.info(f"Successfully fetched {ticker} from {start_date} to {end_date}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            raise
    
    def fetch_multiple_assets(self, tickers: List[str], start_date: str, 
                            end_date: str, data_type: str = 'Close') -> pd.DataFrame:
        """Fetch price data for multiple assets."""
        dfs = []
        for ticker in tickers:
            df = self.fetch_data(ticker, start_date, end_date, data_type)
            dfs.append(df)
        
        # Combine all assets
        combined = pd.concat(dfs, axis=1)
        return combined
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean price data."""
        # Forward fill missing values
        df = df.ffill()
        
        # Drop any remaining NaN rows
        df = df.dropna()
        
        # Ensure datetime index
        df.index = pd.to_datetime(df.index)
        
        return df
    
    def calculate_returns(self, prices: pd.DataFrame, method: str = 'simple') -> pd.DataFrame:
        """Calculate returns from prices."""
        if method == 'simple':
            return prices.pct_change()
        elif method == 'log':
            log_returns = np.log(prices / prices.shift(1))
            return pd.DataFrame(log_returns, index=prices.index, columns=prices.columns)
        else:
            raise ValueError(f"Unknown return method: {method}")
    
    def align_to_macro_data(self, price_data: pd.DataFrame, 
                           macro_data: pd.DataFrame) -> pd.DataFrame:
        """Align price data to macro data frequency and dates."""
        # Reindex to macro data index
        aligned = price_data.reindex(macro_data.index, method='ffill')
        return aligned