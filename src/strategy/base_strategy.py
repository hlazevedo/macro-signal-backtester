from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, name: str, universe: List[str], 
                 params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.universe = universe
        self.params = params or {}
        self.weights_history = pd.DataFrame()
        
    @abstractmethod
    def calculate_weights(self, signals: pd.DataFrame, 
                         prices: pd.DataFrame) -> pd.Series:
        """Calculate portfolio weights based on signals."""
        pass
    
    def generate_weights(self, signals: pd.DataFrame, 
                        prices: pd.DataFrame) -> pd.DataFrame:
        """Generate portfolio weights over time."""
        weights_list = []
        
        # Ensure signals and prices are aligned
        common_dates = signals.index.intersection(prices.index)
        signals = signals.loc[common_dates]
        prices = prices.loc[common_dates]
        
        for date in signals.index:
            # Calculate weights for this date
            signal_row = signals.loc[date]
            price_row = prices.loc[date]
            
            weights = self.calculate_weights(signal_row, price_row)
            weights_list.append(weights)
        
        # Combine all weights
        self.weights_history = pd.DataFrame(weights_list, index=signals.index)
        
        # Apply any constraints
        self.weights_history = self._apply_constraints(self.weights_history)
        
        return self.weights_history
    
    def _apply_constraints(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Apply portfolio constraints (leverage, position limits, etc.)."""
        # Apply leverage constraint
        max_leverage = self.params.get('max_leverage', 1.0)
        leverage = weights.abs().sum(axis=1)
        scaling_factor = max_leverage / leverage.clip(lower=max_leverage)
        weights = weights.multiply(scaling_factor, axis=0)
        
        # Apply position limits
        if 'max_position_size' in self.params:
            max_size = self.params['max_position_size']
            weights = weights.clip(-max_size, max_size)
        
        return weights