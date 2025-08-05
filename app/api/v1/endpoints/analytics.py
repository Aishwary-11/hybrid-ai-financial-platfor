"""
Advanced Analytics Endpoints
Quantitative analysis and machine learning features
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.core.portfolio_engine import PortfolioEngine
from app.core.risk_engine import RiskEngine

router = APIRouter()

# Initialize engines
portfolio_engine = PortfolioEngine()
risk_engine = RiskEngine()


@router.post("/portfolio/create")
async def create_portfolio(name: str, initial_cash: float = 100000):
    """Create a new portfolio"""
    try:
        portfolio = portfolio_engine.create_portfolio(name, initial_cash)
        return {"message": f"Portfolio '{name}' created successfully", "portfolio": portfolio}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/portfolio/{portfolio_name}/add-position")
async def add_position(
    portfolio_name: str,
    symbol: str,
    quantity: float,
    price: float
):
    """Add a position to portfolio"""
    try:
        portfolio_engine.add_position(portfolio_name, symbol, quantity, price)
        return {"message": f"Added {quantity} shares of {symbol} to {portfolio_name}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/portfolio/{portfolio_name}/metrics")
async def get_portfolio_metrics(portfolio_name: str):
    """Get comprehensive portfolio metrics"""
    try:
        metrics = portfolio_engine.calculate_portfolio_metrics(portfolio_name)
        return {"portfolio": portfolio_name, "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/portfolio/{portfolio_name}")
async def get_portfolio(portfolio_name: str):
    """Get portfolio details"""
    if portfolio_name not in portfolio_engine.portfolios:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    portfolio = portfolio_engine.portfolios[portfolio_name]
    return {"portfolio": portfolio}


@router.post("/optimize")
async def optimize_portfolio(
    symbols: List[str] = Query(...),
    method: str = "mean_variance"
):
    """Optimize portfolio allocation"""
    try:
        optimization = portfolio_engine.optimize_portfolio(symbols, method)
        return {
            "method": method,
            "symbols": symbols,
            "optimization": optimization
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/risk/var")
async def calculate_var(
    symbols: List[str] = Query(...),
    confidence_level: float = 0.95,
    method: str = "historical"
):
    """Calculate Value at Risk for given symbols"""
    try:
        # Get market data
        data = portfolio_engine.get_market_data(symbols, period="1y")
        if data.empty:
            raise HTTPException(status_code=400, detail="Unable to fetch market data")
        
        # Calculate portfolio returns (equal weights for simplicity)
        if len(symbols) == 1:
            # Single symbol
            returns = data['Close'].pct_change().dropna()
            portfolio_returns = returns
        else:
            # Multiple symbols - handle grouped data
            if 'Close' in data.columns:
                # Simple case
                returns = data['Close'].pct_change().dropna()
                portfolio_returns = returns.mean(axis=1) if len(returns.shape) > 1 else returns
            else:
                # Grouped by ticker
                close_prices = pd.DataFrame()
                for symbol in symbols:
                    if (symbol, 'Close') in data.columns:
                        close_prices[symbol] = data[(symbol, 'Close')]
                    elif symbol in data.columns.get_level_values(0):
                        close_prices[symbol] = data[symbol]['Close']
                
                returns = close_prices.pct_change().dropna()
                portfolio_returns = returns.mean(axis=1)  # Equal weighted
        
        var_result = risk_engine.calculate_var(portfolio_returns, confidence_level, method)
        
        return {
            "symbols": symbols,
            "var_analysis": var_result
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/risk/metrics")
async def get_risk_metrics(symbols: List[str] = Query(...)):
    """Get comprehensive risk metrics"""
    try:
        # Get market data
        data = portfolio_engine.get_market_data(symbols, period="2y")
        if data.empty:
            raise HTTPException(status_code=400, detail="Unable to fetch market data")
        
        # Calculate portfolio returns
        if len(symbols) == 1:
            # Single symbol
            returns = data['Close'].pct_change().dropna()
            portfolio_returns = returns
        else:
            # Multiple symbols - handle grouped data
            if 'Close' in data.columns:
                # Simple case
                returns = data['Close'].pct_change().dropna()
                portfolio_returns = returns.mean(axis=1) if len(returns.shape) > 1 else returns
            else:
                # Grouped by ticker
                close_prices = pd.DataFrame()
                for symbol in symbols:
                    if (symbol, 'Close') in data.columns:
                        close_prices[symbol] = data[(symbol, 'Close')]
                    elif symbol in data.columns.get_level_values(0):
                        close_prices[symbol] = data[symbol]['Close']
                
                returns = close_prices.pct_change().dropna()
                portfolio_returns = returns.mean(axis=1)  # Equal weighted
        
        risk_metrics = risk_engine.calculate_portfolio_risk_metrics(portfolio_returns)
        
        return {
            "symbols": symbols,
            "risk_metrics": risk_metrics
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/risk/stress-test")
async def run_stress_test(request_data: dict):
    """Run stress test on portfolio"""
    try:
        portfolio_weights = request_data.get("portfolio_weights", {})
        scenario = request_data.get("scenario", "2008_financial_crisis")
        stress_result = risk_engine.run_stress_test(portfolio_weights, scenario)
        return stress_result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/risk/correlation")
async def get_correlation_matrix(symbols: List[str] = Query(...)):
    """Get correlation matrix for symbols"""
    try:
        correlation_matrix = risk_engine.calculate_correlation_matrix(symbols)
        if correlation_matrix.empty:
            raise HTTPException(status_code=400, detail="Unable to calculate correlation matrix")
        
        return {
            "symbols": symbols,
            "correlation_matrix": correlation_matrix.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/market-data/{symbol}")
async def get_market_data(
    symbol: str,
    period: str = "1y",
    interval: str = "1d"
):
    """Get market data for a symbol"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        # Convert to dict for JSON serialization
        data_dict = {
            'dates': data.index.strftime('%Y-%m-%d').tolist(),
            'open': data['Open'].tolist(),
            'high': data['High'].tolist(),
            'low': data['Low'].tolist(),
            'close': data['Close'].tolist(),
            'volume': data['Volume'].tolist()
        }
        
        return {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "data": data_dict
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/analysis/regime-detection")
async def detect_market_regime(symbol: str = "SPY", window: int = 60):
    """Detect market regime changes"""
    try:
        # Get market data
        data = portfolio_engine.get_market_data([symbol], period="2y")
        if data.empty:
            raise HTTPException(status_code=400, detail="Unable to fetch market data")
        
        returns = data['Close'].pct_change().dropna()
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]  # Get first column if DataFrame
        
        regime_analysis = risk_engine.detect_regime_change(returns, window)
        
        return {
            "symbol": symbol,
            "regime_analysis": regime_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/portfolios")
async def list_portfolios():
    """List all portfolios"""
    portfolios = {}
    for name, portfolio in portfolio_engine.portfolios.items():
        portfolios[name] = {
            'name': portfolio['name'],
            'total_value': portfolio['total_value'],
            'cash': portfolio['cash'],
            'positions_count': len(portfolio['positions']),
            'created_at': portfolio['created_at'].isoformat()
        }
    
    return {"portfolios": portfolios}