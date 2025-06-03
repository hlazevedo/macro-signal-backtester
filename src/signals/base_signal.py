from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class BaseSignal(ABC):
    """Abstract base class for all macro signals."""
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params or {}
        self._signal_cache = {}
        
    @abstractmethod
    def calculate_raw_signal(self, data: pd.DataFrame) -> pd.Series:
        """Calculate the raw signal values."""
        pass
    
    def generate_signal(self, data: pd.DataFrame, 
                       normalize: bool = True) -> pd.Series:
        """Generate trading signal from macro data."""
        # Calculate raw signal
        raw_signal = self.calculate_raw_signal(data)
        
        # Normalize if requested
        if normalize:
            signal = self._normalize_signal(raw_signal)
        else:
            signal = raw_signal
        
        # Apply any signal transformations
        signal = self._apply_transformations(signal)
        
        logger.info(f"Generated {self.name} signal")
        return signal
    
    def _normalize_signal(self, signal: pd.Series, 
                         method: str = 'z-score') -> pd.Series:
        """Normalize signal to standard scale."""
        if method == 'z-score':
            # Rolling z-score normalization
            window = self.params.get('zscore_window', 252)
            mean = signal.rolling(window=window, min_periods=window//2).mean()
            std = signal.rolling(window=window, min_periods=window//2).std()
            normalized = (signal - mean) / (std + 1e-8)
            return normalized.fillna(0)
        
        elif method == 'percentile':
            # Rolling percentile rank
            window = self.params.get('percentile_window', 252)
            normalized = signal.rolling(window=window).rank(pct=True)
            return normalized.fillna(0.5)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def _apply_transformations(self, signal: pd.Series) -> pd.Series:
        """Apply any signal transformations (smoothing, etc.)."""
        # Apply smoothing if specified
        if 'smoothing_window' in self.params:
            window = self.params['smoothing_window']
            signal = signal.rolling(window=window, min_periods=1).mean()
        
        # Apply signal bounds if specified
        if 'signal_cap' in self.params:
            cap = self.params['signal_cap']
            signal = signal.clip(-cap, cap)
        
        return signal