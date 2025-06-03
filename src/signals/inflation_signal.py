import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from .base_signal import BaseSignal
import logging

logger = logging.getLogger(__name__)

class InflationSurpriseSignal(BaseSignal):
    """Trading signal based on inflation surprises (CPI momentum)."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            'cpi_column': 'cpi',
            'lookback_period': 12,  # months for YoY calculation
            'ma_window': 12,  # moving average window for trend
            'zscore_window': 252,
            'signal_cap': 2.0
        }
        if params:
            default_params.update(params)
        super().__init__("inflation_surprise", default_params)
    
    def calculate_raw_signal(self, data: pd.DataFrame) -> pd.Series:
        """Calculate inflation surprise signal."""
        try:
            cpi = data[self.params['cpi_column']]
            
            # Calculate YoY change
            lookback = self.params['lookback_period']
            cpi_yoy = (cpi / cpi.shift(lookback) - 1) * 100
            
            # Calculate trend (moving average)
            ma_window = self.params['ma_window']
            cpi_trend = cpi_yoy.rolling(window=ma_window, min_periods=ma_window//2).mean()
            
            # Surprise = actual - trend
            surprise = cpi_yoy - cpi_trend
            
            logger.info(f"Calculated inflation surprise: mean={surprise.mean():.2f}, "
                       f"std={surprise.std():.2f}")
            
            return surprise
            
        except KeyError as e:
            logger.error(f"Missing required data for inflation signal: {e}")
            raise