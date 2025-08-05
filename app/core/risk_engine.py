"""
Risk Management Engine
Advanced risk analytics and monitoring
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats
import yfinance as yf


class RiskEngine:
    """Institutional-grade risk management system"""
    
    def __init__(self):
        self.risk_models = {}
        self.stress_scenarios = self._initialize_stress_scenarios()
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95, 
                     method: str = "historical") -> Dict:
        """Calculate Value at Risk using multiple methods"""
        
        if method == "historical":
            var = self._historical_var(returns, confidence_level)
        elif method == "parametric":
            var = self._parametric_var(returns, confidence_level)
        elif method == "monte_carlo":
            var = self._monte_carlo_var(returns, confidence_level)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        # Calculate Expected Shortfall (CVaR)
        cvar = self._calculate_cvar(returns, confidence_level)
        
        return {
            'var': var,
            'cvar': cvar,
            'confidence_level': confidence_level,
            'method': method,
            'calculated_at': datetime.now()
        }
    
    def _historical_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Historical simulation VaR"""
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def _parametric_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Parametric VaR assuming normal distribution"""
        mean = returns.mean()
        std = returns.std()
        z_score = stats.norm.ppf(1 - confidence_level)
        return mean + z_score * std
    
    def _monte_carlo_var(self, returns: pd.Series, confidence_level: float, 
                        simulations: int = 10000) -> float:
        """Monte Carlo simulation VaR"""
        mean = returns.mean()
        std = returns.std()
        
        # Generate random scenarios
        random_returns = np.random.normal(mean, std, simulations)
        return np.percentile(random_returns, (1 - confidence_level) * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self._historical_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def calculate_portfolio_risk_metrics(self, portfolio_returns: pd.Series) -> Dict:
        """Calculate comprehensive risk metrics for a portfolio"""
        
        # Basic risk metrics
        volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
        downside_deviation = self._downside_deviation(portfolio_returns)
        
        # VaR metrics
        var_95 = self.calculate_var(portfolio_returns, 0.95)
        var_99 = self.calculate_var(portfolio_returns, 0.99)
        
        # Drawdown analysis
        drawdown_metrics = self._calculate_drawdown_metrics(portfolio_returns)
        
        # Tail risk metrics
        skewness = stats.skew(portfolio_returns)
        kurtosis = stats.kurtosis(portfolio_returns)
        
        return {
            'volatility': volatility,
            'downside_deviation': downside_deviation,
            'var_95': var_95,
            'var_99': var_99,
            'drawdown_metrics': drawdown_metrics,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'calculated_at': datetime.now()
        }
    
    def _downside_deviation(self, returns: pd.Series, target_return: float = 0) -> float:
        """Calculate downside deviation"""
        downside_returns = returns[returns < target_return]
        return np.sqrt(np.mean((downside_returns - target_return) ** 2)) * np.sqrt(252)
    
    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict:
        """Calculate drawdown metrics"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1]
        
        # Drawdown duration
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        return {
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'avg_drawdown_duration': avg_drawdown_duration,
            'max_drawdown_duration': max_drawdown_duration
        }
    
    def run_stress_test(self, portfolio_weights: Dict, scenario: str) -> Dict:
        """Run stress test scenarios on portfolio"""
        
        if scenario not in self.stress_scenarios:
            raise ValueError(f"Unknown stress scenario: {scenario}")
        
        scenario_data = self.stress_scenarios[scenario]
        
        # Calculate portfolio impact
        portfolio_impact = 0
        position_impacts = {}
        
        for symbol, weight in portfolio_weights.items():
            if symbol in scenario_data['shocks']:
                shock = scenario_data['shocks'][symbol]
                impact = weight * shock
                portfolio_impact += impact
                position_impacts[symbol] = impact
        
        return {
            'scenario': scenario,
            'portfolio_impact': portfolio_impact,
            'position_impacts': position_impacts,
            'scenario_description': scenario_data['description'],
            'test_date': datetime.now()
        }
    
    def _initialize_stress_scenarios(self) -> Dict:
        """Initialize predefined stress test scenarios"""
        return {
            '2008_financial_crisis': {
                'description': '2008 Financial Crisis scenario',
                'shocks': {
                    'SPY': -0.37,  # S&P 500 decline
                    'QQQ': -0.42,  # NASDAQ decline
                    'IWM': -0.34,  # Small cap decline
                    'EFA': -0.43,  # International developed
                    'EEM': -0.54,  # Emerging markets
                    'TLT': 0.20,   # Long-term treasuries gain
                    'GLD': -0.05,  # Gold slight decline
                    'VNQ': -0.37   # REITs decline
                }
            },
            'covid_crash_2020': {
                'description': 'COVID-19 Market Crash (March 2020)',
                'shocks': {
                    'SPY': -0.34,
                    'QQQ': -0.30,
                    'IWM': -0.42,
                    'EFA': -0.35,
                    'EEM': -0.32,
                    'TLT': 0.15,
                    'GLD': 0.05,
                    'VNQ': -0.44
                }
            },
            'dot_com_bubble': {
                'description': 'Dot-com Bubble Burst (2000-2002)',
                'shocks': {
                    'SPY': -0.49,
                    'QQQ': -0.78,
                    'IWM': -0.27,
                    'EFA': -0.51,
                    'TLT': 0.17,
                    'GLD': 0.12
                }
            },
            'interest_rate_shock': {
                'description': 'Rapid Interest Rate Rise',
                'shocks': {
                    'SPY': -0.15,
                    'QQQ': -0.20,
                    'TLT': -0.25,
                    'VNQ': -0.30,
                    'GLD': -0.10
                }
            }
        }
    
    def calculate_correlation_matrix(self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """Calculate correlation matrix for given symbols"""
        try:
            data = yf.download(symbols, period=period)['Close']
            returns = data.pct_change().dropna()
            return returns.corr()
        except Exception as e:
            print(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()
    
    def detect_regime_change(self, returns: pd.Series, window: int = 60) -> Dict:
        """Detect market regime changes using volatility clustering"""
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=window).std()
        
        # Identify high and low volatility regimes
        vol_median = rolling_vol.median()
        high_vol_threshold = vol_median * 1.5
        low_vol_threshold = vol_median * 0.5
        
        # Classify regimes
        regimes = pd.Series(index=returns.index, dtype=str)
        regimes[rolling_vol > high_vol_threshold] = 'high_volatility'
        regimes[rolling_vol < low_vol_threshold] = 'low_volatility'
        regimes[(rolling_vol >= low_vol_threshold) & (rolling_vol <= high_vol_threshold)] = 'normal'
        
        # Detect regime changes
        regime_changes = regimes != regimes.shift(1)
        change_dates = regimes[regime_changes].index.tolist()
        
        return {
            'current_regime': regimes.iloc[-1],
            'regime_history': regimes.to_dict(),
            'regime_changes': change_dates,
            'volatility_threshold_high': high_vol_threshold,
            'volatility_threshold_low': low_vol_threshold
        }
    
    def calculate_beta(self, asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta coefficient"""
        covariance = np.cov(asset_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        return covariance / market_variance if market_variance != 0 else 0
    
    def calculate_tracking_error(self, portfolio_returns: pd.Series, 
                               benchmark_returns: pd.Series) -> float:
        """Calculate tracking error vs benchmark"""
        active_returns = portfolio_returns - benchmark_returns
        return active_returns.std() * np.sqrt(252)