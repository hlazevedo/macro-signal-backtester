"""
Multi-signal example: Combining yield curve, inflation, and GDP signals
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
from src.signals.inflation_signal import InflationSurpriseSignal
from src.signals.gdp_momentum_signal import GDPMomentumSignal
from src.strategy.macro_strategy import MacroStrategy
from src.backtest.engine import BacktestEngine
from src.utils.logger import setup_logger

# Setup logger
logger = setup_logger("multi_signal_backtest", "INFO")

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
    
    # Initialize signals
    logger.info("Setting up multiple signals...")
    
    yield_signal = YieldCurveSignal(params={
        'zscore_window': 252,
        'smoothing_window': 21,
        'signal_cap': 2.0
    })
    
    inflation_signal = InflationSurpriseSignal(params={
        'lookback_period': 12,
        'ma_window': 12,
        'zscore_window': 252,
        'signal_cap': 2.0
    })
    
    gdp_signal = GDPMomentumSignal(params={
        'momentum_window': 4,
        'zscore_window': 252,
        'signal_cap': 2.0
    })
    
    # Initialize strategy with signal weights
    logger.info("Setting up multi-signal strategy...")
    strategy = MacroStrategy(
        name="MultiSignalStrategy",
        universe=["SPY", "TLT", "GLD"],  # Added GLD for more diversification
        signal_weights={
            "yield_curve": 0.4,
            "inflation_surprise": 0.3,
            "gdp_momentum": 0.3
        },
        params={
            'signal_threshold': 0.3,  # Lower threshold for combined signals
            'max_leverage': 1.0
        }
    )
    
    # Initialize backtest engine
    engine = BacktestEngine(
        macro_loader=macro_loader,
        asset_loader=asset_loader,
        signals={
            "yield_curve": yield_signal,
            "inflation_surprise": inflation_signal,
            "gdp_momentum": gdp_signal
        },
        strategy=strategy,
        config={
            'initial_capital': 1000000,
            'transaction_cost': 0.001
        }
    )
    
    # Run backtest
    logger.info("Running multi-signal backtest...")
    start_date = "2010-01-01"
    end_date = "2023-12-31"
    
    portfolio = engine.run(start_date, end_date, rebalance_frequency='M')
    
    # Evaluate results
    logger.info("Evaluating results...")
    metrics = engine.evaluate()
    
    # Print results
    print("\n" + "="*60)
    print("MULTI-SIGNAL BACKTEST RESULTS")
    print("="*60)
    print(f"Strategy: {strategy.name}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${config['backtest']['initial_capital']:,.0f}")
    print(f"Universe: {', '.join(strategy.universe)}")
    print("-"*60)
    
    # Print performance metrics
    performance_metrics = [
        'total_return', 'annualized_return', 'volatility', 
        'sharpe_ratio', 'max_drawdown', 'win_rate'
    ]
    
    for metric in performance_metrics:
        if metric in metrics:
            value = metrics[metric]
            print(f"{metric.replace('_', ' ').title()}: {value:.2f}{'%' if 'return' in metric or 'rate' in metric else ''}")
    
    print("-"*60)
    print("Signal Statistics:")
    for signal_name in ["yield_curve", "inflation_surprise", "gdp_momentum"]:
        mean_key = f"{signal_name}_mean"
        std_key = f"{signal_name}_std"
        if mean_key in metrics and std_key in metrics:
            print(f"{signal_name}: mean={metrics[mean_key]:.3f}, std={metrics[std_key]:.3f}")
    
    # Generate plots
    from src.backtest.performance import PerformanceAnalyzer
    analyzer = PerformanceAnalyzer(portfolio)
    
    analyzer.plot_nav()
    analyzer.plot_drawdown()
    analyzer.plot_weights()
    analyzer.plot_returns_distribution()
    
    # Save results summary
    summary_df = engine.get_results_summary()
    summary_df.to_csv('backtest_results.csv')
    logger.info("Results saved to backtest_results.csv")

if __name__ == "__main__":
    main()