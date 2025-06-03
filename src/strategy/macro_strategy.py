import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)

class MacroStrategy(BaseStrategy):
    """Strategy that trades based on macro signals."""
    
    def __init__(self, name: str, universe: List[str],
                 signal_weights: Dict[str, float],
                 params: Optional[Dict[str, Any]] = None):
        """
        Initialize macro strategy.
        
        Args:
            name: Strategy name
            universe: List of tradeable assets
            signal_weights: Dict mapping signal names to their weights
            params: Additional parameters
        """
        default_params = {
            'rebalance_frequency': 'M',
            'signal_threshold': 0.5,
            'max_leverage': 1.0,
            'max_position_size': 0.5
        }
        if params:
            default_params.update(params)
            
        super().__init__(name, universe, default_params)
        self.signal_weights = signal_weights
        
    def calculate_weights(self, signals: Union[pd.Series, float], 
                         prices: pd.Series) -> pd.Series:
        """Calculate portfolio weights based on combined signal."""
        # Combine multiple signals if we have them
        if isinstance(signals, pd.Series):
            # Weighted average of signals
            combined_signal = 0
            total_weight = 0
            
            for signal_name, weight in self.signal_weights.items():
                if signal_name in signals:
                    combined_signal += signals[signal_name] * weight
                    total_weight += weight
            
            if total_weight > 0:
                combined_signal /= total_weight
        else:
            # Single signal case
            combined_signal = signals
        
        # Generate weights based on signal strength
        weights = pd.Series(0.0, index=self.universe)
        
        # Simple threshold-based allocation
        threshold = self.params['signal_threshold']
        
        if combined_signal > threshold:
            # Risk-on: Long equities
            if 'SPY' in weights.index:
                weights['SPY'] = 1.0
        elif combined_signal < -threshold:
            # Risk-off: Long bonds
            if 'TLT' in weights.index:
                weights['TLT'] = 1.0
        else:
            # Neutral: Cash or mixed allocation
            if 'SPY' in weights.index and 'TLT' in weights.index:
                weights['SPY'] = 0.5
                weights['TLT'] = 0.5
        
        return weights