import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from .base_signal import BaseSignal
import logging

logger = logging.getLogger(__name__)

class GDPMomentumSignal(BaseSignal):
    """Trading signal based on GDP growth momentum."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            'gdp_column': 'gdp',
            'momentum_window': 4,  # quarters
            'zscore_window': 252,
            'signal_cap': 2.0
        }
        if params:
            default_params.update(params)
        super().__init__("gdp_momentum", default_params)
    
    def calculate_raw_signal(self, data: pd.DataFrame) -> pd.Series:
        """Calculate GDP momentum signal."""
        try:
            gdp = data[self.params['gdp_column']]
            
            # Calculate QoQ growth rate
            gdp_growth = gdp.pct_change()
            
            # Calculate momentum (acceleration)
            window = self.params['momentum_window']
            current_avg = gdp_growth.rolling(window=window//2).mean()
            previous_avg = gdp_growth.shift(window//2).rolling(window=window//2).mean()
            
            momentum = current_avg - previous_avg
            
            logger.info(f"Calculated GDP momentum: mean={momentum.mean():.4f}, "
                       f"std={momentum.std():.4f}")
            
            return momentum
            
        except KeyError as e:
            logger.error(f"Missing required data for GDP momentum signal: {e}")
            raise