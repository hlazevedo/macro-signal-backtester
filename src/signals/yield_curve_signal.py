import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from .base_signal import BaseSignal
import logging

logger = logging.getLogger(__name__)

class YieldCurveSignal(BaseSignal):
    """Trading signal based on yield curve slope (10Y - 2Y spread)."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            'long_yield': 'yield_10y',
            'short_yield': 'yield_2y',
            'zscore_window': 252,
            'smoothing_window': 21,
            'signal_cap': 2.0,
            'invert': False  # Set True if you want to go long when curve flattens
        }
        if params:
            default_params.update(params)
        super().__init__("yield_curve", default_params)
    
    def calculate_raw_signal(self, data: pd.DataFrame) -> pd.Series:
        """Calculate yield curve slope."""
        try:
            long_yield = data[self.params['long_yield']]
            short_yield = data[self.params['short_yield']]
            
            # Calculate spread
            spread = long_yield - short_yield
            
            # Invert if specified (useful for different interpretations)
            if self.params.get('invert', False):
                spread = -spread
            
            logger.info(f"Calculated yield curve spread: mean={spread.mean():.2f}, "
                       f"std={spread.std():.2f}")
            
            return spread
            
        except KeyError as e:
            logger.error(f"Missing required data for yield curve signal: {e}")
            raise