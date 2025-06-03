import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from ..data.macro_data_loader import MacroDataLoader
from ..data.asset_price_loader import AssetPriceLoader
from ..signals.base_signal import BaseSignal
from ..strategy.base_strategy import BaseStrategy
from ..portfolio.portfolio import Portfolio
from .performance import PerformanceAnalyzer

logger = logging.getLogger(__name__)

class BacktestEngine:
    """Main backtesting engine that orchestrates the entire process."""
    
    def __init__(self, 
                 macro_loader: MacroDataLoader,
                 asset_loader: AssetPriceLoader,
                 signals: Dict[str, BaseSignal],
                 strategy: BaseStrategy,
                 config: Optional[Dict[str, Any]] = None):
        self.macro_loader = macro_loader
        self.asset_loader = asset_loader
        self.signals = signals
        self.strategy = strategy
        self.config = config or {}
        
        # Results storage
        self.portfolio = None
        self.signal_data = pd.DataFrame()
        self.price_data = pd.DataFrame()
        self.performance_analyzer = None
        
    def run(self, start_date: str, end_date: str, 
            rebalance_frequency: str = 'ME') -> Portfolio:  # Changed from 'M' to 'ME'
        """Run the backtest."""
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Step 1: Load macro data
        logger.info("Loading macro data...")
        # Get all required macro series from signals
        required_series = set()
        for signal in self.signals.values():
            # Extract series names from signal parameters
            for param_value in signal.params.values():
                if isinstance(param_value, str) and param_value in ['yield_10y', 'yield_2y', 'cpi', 'gdp', 'unemployment', 'vix', 'dollar_index']:
                    required_series.add(param_value)
        
        # Add default series if none found
        if not required_series:
            required_series = ['yield_10y', 'yield_2y', 'cpi', 'gdp']
        
        macro_data = self.macro_loader.fetch_multiple_series(
            list(required_series), start_date, end_date
        )
        
        # Step 2: Load asset price data
        logger.info("Loading asset price data...")
        self.price_data = self.asset_loader.fetch_multiple_assets(
            self.strategy.universe, start_date, end_date
        )
        
        # Step 3: Generate signals
        logger.info("Generating signals...")
        signal_values = {}
        for name, signal in self.signals.items():
            try:
                signal_values[name] = signal.generate_signal(macro_data)
            except Exception as e:
                logger.warning(f"Failed to generate signal {name}: {e}")
                # Create dummy signal if one fails
                signal_values[name] = pd.Series(0, index=macro_data.index, name=name)
        
        self.signal_data = pd.DataFrame(signal_values)
        
        # Step 4: Align all data to common dates
        logger.info("Aligning data...")
        common_dates = self.signal_data.index.intersection(self.price_data.index)
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates between signals and price data")
        
        self.signal_data = self.signal_data.loc[common_dates]
        self.price_data = self.price_data.loc[common_dates]
        
        # Step 5: Create rebalance dates and ensure they exist in data
        logger.info("Setting up rebalancing schedule...")
        rebalance_dates = pd.date_range(
            start=common_dates[0],
            end=common_dates[-1],
            freq=rebalance_frequency
        )
        
        # Find the closest actual dates in our data for each rebalance date
        actual_rebalance_dates = []
        for target_date in rebalance_dates:
            # Find the closest available date (forward fill approach)
            available_dates = common_dates[common_dates >= target_date]
            if len(available_dates) > 0:
                actual_rebalance_dates.append(available_dates[0])
            else:
                # If no future dates available, use the last available date
                if len(common_dates) > 0:
                    actual_rebalance_dates.append(common_dates[-1])
        
        # Remove duplicates while preserving order
        actual_rebalance_dates = pd.Index(actual_rebalance_dates).drop_duplicates()
        
        logger.info(f"Rebalancing on {len(actual_rebalance_dates)} dates")
        
        # Step 6: Initialize portfolio
        initial_capital = self.config.get('initial_capital', 1000000)
        transaction_cost = self.config.get('transaction_cost', 0.001)
        self.portfolio = Portfolio(initial_capital, transaction_cost)
        
        # Step 7: Run strategy and update portfolio
        logger.info("Running strategy...")
        for i, date in enumerate(actual_rebalance_dates):
            try:
                # Get signals and prices for this date
                current_signals = self.signal_data.loc[date]
                current_prices = self.price_data.loc[date]
                
                # Check for any NaN values and handle them
                if current_signals.isna().any():
                    logger.warning(f"NaN signals found on {date}, filling with 0")
                    current_signals = current_signals.fillna(0)
                
                if current_prices.isna().any():
                    logger.warning(f"NaN prices found on {date}, skipping rebalance")
                    continue
                
                # Calculate target weights
                weights = self.strategy.calculate_weights(current_signals, current_prices)
                
                # Update portfolio
                self.portfolio.update_holdings(weights, current_prices, date)
                
                if i % 10 == 0:  # Log progress every 10 rebalances
                    logger.info(f"Processed {i+1}/{len(actual_rebalance_dates)} rebalance dates")
                    
            except Exception as e:
                logger.error(f"Error processing date {date}: {e}")
                continue
        
        # Step 8: Calculate final NAV and returns
        logger.info("Calculating performance...")
        self.portfolio.calculate_nav(self.price_data)
        self.portfolio.calculate_returns()
        
        # Step 9: Initialize performance analyzer
        self.performance_analyzer = PerformanceAnalyzer(self.portfolio)
        
        logger.info("Backtest completed successfully")
        return self.portfolio
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate backtest performance."""
        if self.portfolio is None:
            raise ValueError("Must run backtest before evaluation")
        
        # Get basic metrics
        metrics = self.portfolio.get_performance_metrics()
        
        # Add signal statistics
        signal_stats = {}
        for name, signal_series in self.signal_data.items():
            clean_signal = signal_series.dropna()
            if len(clean_signal) > 0:
                signal_stats[f"{name}_mean"] = clean_signal.mean()
                signal_stats[f"{name}_std"] = clean_signal.std()
                signal_stats[f"{name}_skew"] = clean_signal.skew()
        
        metrics.update(signal_stats)
        
        # Add turnover statistics
        if not self.portfolio.trades.empty:
            turnover = self.portfolio.trades.abs().sum(axis=1).mean()
            metrics['avg_daily_turnover'] = turnover
        
        return metrics
    
    def get_results_summary(self) -> pd.DataFrame:
        """Get a summary DataFrame of results."""
        if self.portfolio is None:
            return pd.DataFrame()
            
        summary_data = {}
        
        # Start with common dates
        if not self.portfolio.nav.empty:
            common_dates = self.portfolio.nav.index
            
            summary_data['NAV'] = self.portfolio.nav
            summary_data['Returns'] = self.portfolio.returns
            summary_data['Cash'] = self.portfolio.cash
            
            # Add weights for each asset (reindex to common dates)
            if not self.portfolio.weights.empty:
                weights_aligned = self.portfolio.weights.reindex(common_dates, method='ffill')
                for asset in self.strategy.universe:
                    if asset in weights_aligned.columns:
                        summary_data[f'Weight_{asset}'] = weights_aligned[asset]
            
            # Add signal values (reindex to common dates)
            if not self.signal_data.empty:
                signals_aligned = self.signal_data.reindex(common_dates, method='ffill')
                for signal_name in signals_aligned.columns:
                    summary_data[f'Signal_{signal_name}'] = signals_aligned[signal_name]
        
        return pd.DataFrame(summary_data)