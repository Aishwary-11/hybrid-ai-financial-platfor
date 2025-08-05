"""
Core Portfolio Management Engine
Handles portfolio construction, optimization, and rebalancing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import yfinance as yf
from scipy.optimize import minimize


class PortfolioEngine:
    """Advanced portfolio management and construction engine"""
    
    def __init__(self):
        self.portfolios = {}
        self.market_data = {}
    
    def create_portfolio(self, name: str, initial_cash: float = 100000) -> Dict:
        """Create a new portfolio"""
        portfolio = {
            'name': name,
            'cash': initial_cash,
            'positions': {},
            'total_value': initial_cash,
            'created_at': datetime.now(),
            'performance': {
                'total_return': 0.0,
                'daily_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
        }
        self.portfolios[name] = portfolio
        return portfolio
    
    def add_position(self, portfolio_name: str, symbol: str, quantity: float, price: float):
        """Add a position to portfolio"""
        if portfolio_name not in self.portfolios:
            raise ValueError(f"Portfolio {portfolio_name} not found")
        
        portfolio = self.portfolios[portfolio_name]
        cost = quantity * price
        
        if cost > portfolio['cash']:
            raise ValueError("Insufficient cash for position")
        
        if symbol in portfolio['positions']:
            # Update existing position
            existing = portfolio['positions'][symbol]
            total_quantity = existing['quantity'] + quantity
            avg_price = (existing['avg_cost'] * existing['quantity'] + cost) / total_quantity
            portfolio['positions'][symbol] = {
                'quantity': total_quantity,
                'avg_cost': avg_price,
                'current_price': price,
                'market_value': total_quantity * price
            }
        else:
            # New position
            portfolio['positions'][symbol] = {
                'quantity': quantity,
                'avg_cost': price,
                'current_price': price,
                'market_value': cost
            }
        
        portfolio['cash'] -= cost
        self._update_portfolio_value(portfolio_name)
    
    def get_market_data(self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """Fetch market data for symbols"""
        try:
            if len(symbols) == 1:
                # Single symbol
                data = yf.download(symbols[0], period=period)
                return data
            else:
                # Multiple symbols
                data = yf.download(symbols, period=period, group_by='ticker')
                return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def calculate_portfolio_metrics(self, portfolio_name: str) -> Dict:
        """Calculate comprehensive portfolio metrics"""
        if portfolio_name not in self.portfolios:
            raise ValueError(f"Portfolio {portfolio_name} not found")
        
        portfolio = self.portfolios[portfolio_name]
        positions = portfolio['positions']
        
        if not positions:
            return portfolio['performance']
        
        # Get symbols and fetch data
        symbols = list(positions.keys())
        data = self.get_market_data(symbols, period="1y")
        
        if data.empty:
            return portfolio['performance']
        
        # Calculate returns
        returns = data['Close'].pct_change().dropna()
        
        # Portfolio weights
        total_value = sum(pos['market_value'] for pos in positions.values())
        weights = {symbol: positions[symbol]['market_value'] / total_value 
                  for symbol in symbols}
        
        # Portfolio returns
        portfolio_returns = sum(returns[symbol] * weights[symbol] for symbol in symbols)
        
        # Performance metrics
        total_return = (total_value / (portfolio['total_value'] - portfolio['cash'])) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = (portfolio_returns.mean() * 252) / volatility if volatility > 0 else 0
        
        # Max drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        metrics = {
            'total_return': total_return,
            'daily_return': portfolio_returns.iloc[-1] if len(portfolio_returns) > 0 else 0,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
        portfolio['performance'] = metrics
        return metrics
    
    def optimize_portfolio(self, symbols: List[str], method: str = "mean_variance") -> Dict:
        """Portfolio optimization using various methods"""
        data = self.get_market_data(symbols, period="2y")
        
        if data.empty:
            return {}
        
        # Handle different data structures from yfinance
        if len(symbols) == 1:
            # Single symbol
            returns = data['Close'].pct_change().dropna()
            returns = pd.DataFrame({symbols[0]: returns})
        else:
            # Multiple symbols - handle grouped data
            if 'Close' in data.columns:
                # Simple case
                returns = data['Close'].pct_change().dropna()
                if len(returns.shape) == 1:
                    returns = pd.DataFrame({symbols[0]: returns})
            else:
                # Grouped by ticker
                close_prices = pd.DataFrame()
                for symbol in symbols:
                    if (symbol, 'Close') in data.columns:
                        close_prices[symbol] = data[(symbol, 'Close')]
                    elif symbol in data.columns.get_level_values(0):
                        close_prices[symbol] = data[symbol]['Close']
                
                returns = close_prices.pct_change().dropna()
        
        if method == "mean_variance":
            return self._mean_variance_optimization(returns)
        elif method == "risk_parity":
            return self._risk_parity_optimization(returns)
        elif method == "black_litterman":
            return self._black_litterman_optimization(returns)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _mean_variance_optimization(self, returns: pd.DataFrame) -> Dict:
        """Mean-variance optimization (Markowitz)"""
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        num_assets = len(returns.columns)
        
        def objective(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            # Maximize Sharpe ratio (minimize negative Sharpe)
            return -portfolio_return / np.sqrt(portfolio_variance)
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        result = minimize(objective, num_assets * [1. / num_assets], 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        optimal_weights = dict(zip(returns.columns, result.x))
        
        return {
            'weights': optimal_weights,
            'expected_return': np.sum(mean_returns * result.x),
            'volatility': np.sqrt(np.dot(result.x.T, np.dot(cov_matrix, result.x))),
            'sharpe_ratio': -result.fun
        }
    
    def _risk_parity_optimization(self, returns: pd.DataFrame) -> Dict:
        """Risk parity optimization - equal risk contribution"""
        cov_matrix = returns.cov() * 252
        num_assets = len(returns.columns)
        
        def risk_budget_objective(weights):
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            marginal_contrib = np.dot(cov_matrix, weights)
            contrib = weights * marginal_contrib / portfolio_variance
            # Minimize sum of squared deviations from equal risk
            target_risk = 1.0 / num_assets
            return np.sum((contrib - target_risk) ** 2)
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.01, 1) for _ in range(num_assets))
        
        result = minimize(risk_budget_objective, num_assets * [1. / num_assets],
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        optimal_weights = dict(zip(returns.columns, result.x))
        
        return {
            'weights': optimal_weights,
            'volatility': np.sqrt(np.dot(result.x.T, np.dot(cov_matrix, result.x)))
        }
    
    def _black_litterman_optimization(self, returns: pd.DataFrame) -> Dict:
        """Black-Litterman optimization with market equilibrium"""
        # Simplified Black-Litterman implementation
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        # Market cap weights (simplified - equal weights)
        market_weights = np.array([1/len(returns.columns)] * len(returns.columns))
        
        # Risk aversion parameter
        risk_aversion = 3.0
        
        # Implied equilibrium returns
        pi = risk_aversion * np.dot(cov_matrix, market_weights)
        
        # Black-Litterman expected returns (no views for simplicity)
        bl_returns = pi
        
        # Optimization with Black-Litterman returns
        def objective(weights):
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_return = np.sum(bl_returns * weights)
            return -portfolio_return / np.sqrt(portfolio_variance)
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(returns.columns)))
        
        result = minimize(objective, market_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        optimal_weights = dict(zip(returns.columns, result.x))
        
        return {
            'weights': optimal_weights,
            'expected_return': np.sum(bl_returns * result.x),
            'volatility': np.sqrt(np.dot(result.x.T, np.dot(cov_matrix, result.x))),
            'sharpe_ratio': -result.fun
        }
    
    def _update_portfolio_value(self, portfolio_name: str):
        """Update total portfolio value"""
        portfolio = self.portfolios[portfolio_name]
        positions_value = sum(pos['market_value'] for pos in portfolio['positions'].values())
        portfolio['total_value'] = portfolio['cash'] + positions_value