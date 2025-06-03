import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.portfolio.portfolio import Portfolio
from src.strategy.macro_strategy import MacroStrategy

class TestPortfolio:
    def test_portfolio_initialization(self):
        portfolio = Portfolio(initial_capital=1000000)
        assert portfolio.initial_capital == 1000000
        assert portfolio.transaction_cost == 0.001
        
    def test_nav_calculation(self):
        portfolio = Portfolio(initial_capital=100000)
        
        # Create sample data
        dates = pd.date_range('2023-01-01', '2023-01-05', freq='D')
        prices = pd.DataFrame({
            'SPY': [400, 405, 410, 408, 412],
            'TLT': [100, 99, 98, 99, 100]
        }, index=dates)
        
        # Update holdings
        weights = pd.Series({'SPY': 0.6, 'TLT': 0.4})
        portfolio.update_holdings(weights, prices.iloc[0], dates[0])
        
        # Calculate NAV
        nav = portfolio.calculate_nav(prices)
        
        assert len(nav) == 1
        assert nav.iloc[0] > 0

class TestStrategy:
    def test_macro_strategy_weights(self):
        strategy = MacroStrategy(
            name="TestStrategy",
            universe=["SPY", "TLT"],
            signal_weights={"test_signal": 1.0}
        )
        
        # Test different signal levels
        signals = pd.Series({"test_signal": 1.5})
        prices = pd.Series({"SPY": 400, "TLT": 100})
        
        weights = strategy.calculate_weights(signals, prices)
        
        assert isinstance(weights, pd.Series)
        assert weights.sum() <= 1.0 + 1e-6