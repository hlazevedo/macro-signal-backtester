import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class Portfolio:
    """Portfolio tracking and performance calculation."""
    
    def __init__(self, initial_capital: float = 1000000, 
                 transaction_cost: float = 0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # Initialize tracking DataFrames with proper dtypes
        self.holdings = pd.DataFrame()
        self.weights = pd.DataFrame()
        self.trades = pd.DataFrame()
        self.cash = pd.Series(dtype=float, name='cash')
        self.nav = pd.Series(dtype=float, name='nav')
        self.returns = pd.Series(dtype=float, name='returns')
        
    def update_holdings(self, target_weights: pd.Series, prices: pd.Series, 
                       date: pd.Timestamp) -> None:
        """Update portfolio holdings based on target weights."""
        # Calculate current portfolio value
        if self.nav.empty:
            current_nav = self.initial_capital
        else:
            current_nav = self.nav.iloc[-1]
        
        # Calculate target holdings in shares
        target_value = target_weights * current_nav
        target_shares = target_value / prices
        
        # Calculate trades needed
        if self.holdings.empty:
            current_shares = pd.Series(0, index=target_shares.index, dtype=float)
        else:
            current_shares = self.holdings.iloc[-1].reindex(target_shares.index, fill_value=0)
        
        trades = target_shares - current_shares
        
        # Calculate transaction costs
        trade_value = (trades.abs() * prices).sum()
        costs = trade_value * self.transaction_cost
        
        # Update cash position
        cash_change = -(trades * prices).sum() - costs
        if self.cash.empty:
            new_cash = self.initial_capital + cash_change
        else:
            new_cash = self.cash.iloc[-1] + cash_change
        
        # Store updates using proper concatenation
        new_holdings = pd.DataFrame([target_shares], index=[date])
        new_weights = pd.DataFrame([target_weights], index=[date])
        new_trades = pd.DataFrame([trades], index=[date])
        new_cash_series = pd.Series([new_cash], index=[date], name='cash')
        
        # Concatenate properly
        if self.holdings.empty:
            self.holdings = new_holdings
        else:
            self.holdings = pd.concat([self.holdings, new_holdings])
            
        if self.weights.empty:
            self.weights = new_weights
        else:
            self.weights = pd.concat([self.weights, new_weights])
            
        if self.trades.empty:
            self.trades = new_trades
        else:
            self.trades = pd.concat([self.trades, new_trades])
            
        if self.cash.empty:
            self.cash = new_cash_series
        else:
            self.cash = pd.concat([self.cash, new_cash_series])
        
    def calculate_nav(self, prices: pd.DataFrame) -> pd.Series:
        """Calculate Net Asset Value over time."""
        if self.holdings.empty:
            return pd.Series(dtype=float)
            
        nav_list = []
        nav_dates = []
        
        for date in self.holdings.index:
            # Get holdings for this date
            holdings = self.holdings.loc[date]
            
            # Get prices - use forward fill for missing data
            if date in prices.index:
                price_row = prices.loc[date]
            else:
                # Find the closest available price date
                valid_dates = prices.index[prices.index <= date]
                if len(valid_dates) > 0:
                    price_row = prices.loc[valid_dates[-1]]
                else:
                    continue
            
            # Ensure holdings and prices are aligned
            common_assets = holdings.index.intersection(price_row.index)
            if len(common_assets) == 0:
                continue
                
            holdings_aligned = holdings.reindex(common_assets, fill_value=0)
            prices_aligned = price_row.reindex(common_assets, fill_value=0)
            
            # Calculate portfolio value
            portfolio_value = (holdings_aligned * prices_aligned).sum()
            
            # Get cash value
            if date in self.cash.index:
                cash_value = self.cash.loc[date]
            else:
                # Use most recent cash value
                cash_dates = self.cash.index[self.cash.index <= date]
                if len(cash_dates) > 0:
                    cash_value = self.cash.loc[cash_dates[-1]]
                else:
                    cash_value = self.initial_capital
            
            total_nav = portfolio_value + cash_value
            
            nav_list.append(total_nav)
            nav_dates.append(date)
        
        self.nav = pd.Series(nav_list, index=nav_dates, name='nav')
        return self.nav
    
    def calculate_returns(self) -> pd.Series:
        """Calculate portfolio returns."""
        if len(self.nav) > 1:
            self.returns = self.nav.pct_change().dropna()
        else:
            self.returns = pd.Series(dtype=float, name='returns')
        return self.returns
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        if self.returns.empty or len(self.returns) < 2:
            return {
                'total_return': 0,
                'annualized_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0
            }
            
        returns = self.returns.dropna()
        
        # Annualization factor (assuming daily returns)
        ann_factor = np.sqrt(252)
        
        # Basic metrics
        total_return = (self.nav.iloc[-1] / self.initial_capital - 1) * 100 if not self.nav.empty else 0
        annualized_return = returns.mean() * 252 * 100
        volatility = returns.std() * ann_factor * 100
        sharpe_ratio = (returns.mean() / returns.std()) * ann_factor if returns.std() > 0 else 0
        max_drawdown = self._calculate_max_drawdown() * 100
        
        # Win/loss metrics
        win_rate = (returns > 0).mean() * 100 if len(returns) > 0 else 0
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]
        avg_win = winning_returns.mean() * 100 if len(winning_returns) > 0 else 0
        avg_loss = losing_returns.mean() * 100 if len(losing_returns) > 0 else 0
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
        
        return metrics
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if self.returns.empty:
            return 0
            
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()