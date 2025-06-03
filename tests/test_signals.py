import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.signals.yield_curve_signal import YieldCurveSignal
from src.signals.inflation_signal import InflationSurpriseSignal

class TestYieldCurveSignal:
    def test_signal_generation(self):
        # Create sample data
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        data = pd.DataFrame({
            'yield_10y': np.random.normal(2.5, 0.5, len(dates)),
            'yield_2y': np.random.normal(1.5, 0.3, len(dates))
        }, index=dates)
        
        # Initialize signal
        signal = YieldCurveSignal()
        
        # Generate signal
        result = signal.generate_signal(data)
        
        # Assertions
        assert isinstance(result, pd.Series)
        assert len(result) == len(dates)
        assert not result.isna().all()

class TestInflationSignal:
    def test_signal_generation(self):
        # Create sample data
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
        cpi_values = 100 * (1 + np.random.normal(0.002, 0.003, len(dates))).cumprod()
        data = pd.DataFrame({
            'cpi': cpi_values
        }, index=dates)
        
        # Initialize signal
        signal = InflationSurpriseSignal()
        
        # Generate signal
        result = signal.generate_signal(data)
        
        # Assertions
        assert isinstance(result, pd.Series)
        assert len(result) == len(dates)