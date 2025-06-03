"""
strategy example: Trading SPY/TLT based on yield curve signal
"""
import os
import yaml
from datetime import datetime

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.macro_data_loader import MacroDataLoader
from src.data.asset_price_loader import AssetPriceLoader
from src.signals.yield_curve_signal import YieldCurveSignal
from src.strategy.macro_strategy import MacroStrategy
from src.backtest.engine import BacktestEngine
from src.utils.logger import setup_logger

# Setup logger
logger = setup_logger("backtest_example", "INFO")

def main():
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize data loaders
    logger.info("Initializing data loaders...")
    macro_loader = MacroDataLoader(
        fred_api_key=config['data']['fred_api_key']
    )
    asset_loader = AssetPriceLoader()
    
    # Initialize yield curve signal
    logger.info("Setting up yield curve signal...")
    yield_signal = YieldCurveSignal(params={
        'zscore_window': 252,
        'smoothing_window': 21,
        'signal_cap': 2.0
    })
    
    # Initialize strategy
    logger.info("Setting up strategy...")
    strategy = MacroStrategy(
        name="YieldCurveStrategy",
        universe=["SPY", "TLT"],
        signal_weights={"yield_curve": 1.0},
        params={
            'signal_threshold': 0.5,
            'max_leverage': 1.0
        }
    )
    
    # Initialize backtest engine
    engine = BacktestEngine(
        macro_loader=macro_loader,
        asset_loader=asset_loader,
        signals={"yield_curve": yield_signal},
        strategy=strategy,
        config={
            'initial_capital': 1000000,
            'transaction_cost': 0.001
        }
    )
    
    # Run backtest
    logger.info("Running backtest...")
    start_date = "2015-01-01"
    end_date = "2023-12-31"
    
    portfolio = engine.run(start_date, end_date, rebalance_frequency='M')
    
    # Evaluate results
    logger.info("Evaluating results...")
    metrics = engine.evaluate()
    
    # Print results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Strategy: {strategy.name}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${config['backtest']['initial_capital']:,.0f}")
    print("-"*60)
    
    for metric, value in metrics.items():
        if 'signal' not in metric.lower():  # Skip signal stats for clean output
            print(f"{metric.replace('_', ' ').title()}: {value:.2f}{'%' if 'return' in metric or 'rate' in metric else ''}")
    
    # Generate plots
    from src.backtest.performance import PerformanceAnalyzer
    analyzer = PerformanceAnalyzer(portfolio)
    
    analyzer.plot_nav()
    analyzer.plot_drawdown()
    analyzer.plot_weights()
    analyzer.plot_returns_distribution()

if __name__ == "__main__":
    main()