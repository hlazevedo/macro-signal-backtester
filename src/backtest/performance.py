import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple, Any
from ..portfolio.portfolio import Portfolio

class PerformanceAnalyzer:
    """Analyze and visualize backtest performance."""
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        
    def plot_nav(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """Plot portfolio NAV over time."""
        fig, ax = plt.subplots(figsize=figsize)
        
        self.portfolio.nav.plot(ax=ax, linewidth=2, color='navy')
        ax.set_title('Portfolio Net Asset Value', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('NAV ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add initial capital line
        ax.axhline(y=self.portfolio.initial_capital, color='gray', 
                   linestyle='--', label='Initial Capital')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_returns_distribution(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """Plot returns distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        returns = self.portfolio.returns.dropna()
        
        # Histogram
        returns.hist(bins=50, ax=ax1, color='skyblue', edgecolor='black')
        ax1.set_title('Returns Distribution', fontsize=12)
        ax1.set_xlabel('Daily Returns', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.axvline(x=returns.mean(), color='red', linestyle='--', 
                    label=f'Mean: {returns.mean():.4f}')
        ax1.legend()
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_drawdown(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """Plot drawdown chart."""
        cumulative = (1 + self.portfolio.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig, ax = plt.subplots(figsize=figsize)
        drawdown.plot(ax=ax, linewidth=1, color='red', label='Drawdown')
        ax.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
        
        ax.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(top=0.05)
        
        # Mark maximum drawdown
        max_dd_date = drawdown.idxmin()
        max_dd_value = drawdown.min()
        ax.plot(max_dd_date, max_dd_value, 'ko', markersize=8)
        ax.annotate(f'Max DD: {max_dd_value:.2%}', 
                    xy=(max_dd_date, max_dd_value),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.show()
    
    def plot_weights(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot portfolio weights over time."""
        weights = self.portfolio.weights
        
        if weights.empty:
            print("No weights to plot")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create stacked area chart
        weights.plot.area(ax=ax, stacked=True, alpha=0.8)
        
        ax.set_title('Portfolio Weights Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Weight', fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_performance_report(self) -> Dict[str, Any]:
        """Create comprehensive performance report."""
        metrics = self.portfolio.get_performance_metrics()
        
        # Add additional metrics
        returns = self.portfolio.returns.dropna()
        
        # Risk metrics
        metrics['var_95'] = returns.quantile(0.05) * 100
        metrics['cvar_95'] = returns[returns <= returns.quantile(0.05)].mean() * 100
        metrics['kurtosis'] = returns.kurtosis()
        metrics['skewness'] = returns.skew()
        
        # Monthly aggregation
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        metrics['best_month'] = monthly_returns.max() * 100
        metrics['worst_month'] = monthly_returns.min() * 100
        metrics['positive_months'] = (monthly_returns > 0).mean() * 100
        
        return metrics