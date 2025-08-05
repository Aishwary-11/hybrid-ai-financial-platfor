"""
Personal Aladdin - Simplified Working Version
Institutional-grade investment management platform
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import uvicorn
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json

app = FastAPI(
    title="Personal Aladdin - Investment Platform",
    description="Institutional-grade investment management with AI predictions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for portfolios and predictions
portfolios = {}
prediction_cache = {}

@app.get("/")
async def root():
    return {
        "message": "Personal Aladdin - Investment Management Platform",
        "version": "1.0.0",
        "status": "operational",
        "features": [
            "Portfolio Management",
            "Live Market Data",
            "AI-Powered Predictions",
            "Risk Analytics",
            "Technical Analysis"
        ],
        "endpoints": {
            "create_portfolio": "/portfolio/create",
            "live_quote": "/market/quote/{symbol}",
            "predict_price": "/ai/predict/{symbol}",
            "market_analysis": "/analysis/{symbol}",
            "portfolio_optimize": "/portfolio/optimize"
        }
    }

@app.post("/portfolio/create")
async def create_portfolio(name: str, initial_cash: float = 100000):
    """Create a new investment portfolio"""
    try:
        portfolio = {
            "name": name,
            "cash": initial_cash,
            "positions": {},
            "total_value": initial_cash,
            "created_at": datetime.now().isoformat(),
            "performance": {
                "total_return": 0.0,
                "daily_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0
            }
        }
        portfolios[name] = portfolio
        return {"message": f"Portfolio '{name}' created successfully", "portfolio": portfolio}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/portfolio/{portfolio_name}/add")
async def add_position(portfolio_name: str, symbol: str, quantity: float, price: float):
    """Add a stock position to portfolio"""
    try:
        if portfolio_name not in portfolios:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        portfolio = portfolios[portfolio_name]
        cost = quantity * price
        
        if cost > portfolio["cash"]:
            raise HTTPException(status_code=400, detail="Insufficient cash")
        
        # Add position
        if symbol in portfolio["positions"]:
            existing = portfolio["positions"][symbol]
            total_qty = existing["quantity"] + quantity
            avg_price = (existing["avg_price"] * existing["quantity"] + cost) / total_qty
            portfolio["positions"][symbol] = {
                "quantity": total_qty,
                "avg_price": avg_price,
                "current_price": price,
                "market_value": total_qty * price
            }
        else:
            portfolio["positions"][symbol] = {
                "quantity": quantity,
                "avg_price": price,
                "current_price": price,
                "market_value": cost
            }
        
        portfolio["cash"] -= cost
        portfolio["total_value"] = portfolio["cash"] + sum(pos["market_value"] for pos in portfolio["positions"].values())
        
        return {"message": f"Added {quantity} shares of {symbol}", "portfolio": portfolio}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/portfolio/{portfolio_name}")
async def get_portfolio(portfolio_name: str):
    """Get portfolio details"""
    if portfolio_name not in portfolios:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return {"portfolio": portfolios[portfolio_name]}

@app.get("/market/quote/{symbol}")
async def get_live_quote(symbol: str):
    """Get live market quote for a symbol"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="2d")
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        current_price = hist['Close'].iloc[-1]
        previous_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change = current_price - previous_price
        change_pct = (change / previous_price) * 100 if previous_price != 0 else 0
        
        return {
            "symbol": symbol,
            "price": round(current_price, 2),
            "change": round(change, 2),
            "change_percent": round(change_pct, 2),
            "volume": int(hist['Volume'].iloc[-1]),
            "high": round(hist['High'].iloc[-1], 2),
            "low": round(hist['Low'].iloc[-1], 2),
            "open": round(hist['Open'].iloc[-1], 2),
            "market_cap": info.get('marketCap', 0),
            "pe_ratio": info.get('trailingPE', 0),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to get quote: {str(e)}")

@app.get("/ai/predict/{symbol}")
async def predict_stock_price(symbol: str, days: int = 5):
    """AI-powered stock price prediction"""
    try:
        # Check cache first
        cache_key = f"{symbol}_{days}"
        if cache_key in prediction_cache:
            cached = prediction_cache[cache_key]
            if (datetime.now() - datetime.fromisoformat(cached["timestamp"])).seconds < 3600:  # 1 hour cache
                return cached
        
        # Get historical data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y")
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Simple prediction using moving averages and momentum
        current_price = hist['Close'].iloc[-1]
        
        # Calculate technical indicators
        sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        rsi = calculate_rsi(hist['Close']).iloc[-1]
        
        # Calculate returns and volatility
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Simple prediction logic
        momentum_score = (current_price - sma_20) / sma_20
        trend_score = (sma_20 - sma_50) / sma_50
        
        # Predict direction and magnitude
        if momentum_score > 0.02 and trend_score > 0.01 and rsi < 70:
            direction = "bullish"
            predicted_return = min(0.05, abs(momentum_score) * 2)  # Cap at 5%
        elif momentum_score < -0.02 and trend_score < -0.01 and rsi > 30:
            direction = "bearish"
            predicted_return = -min(0.05, abs(momentum_score) * 2)
        else:
            direction = "neutral"
            predicted_return = np.random.normal(0, 0.01)  # Small random walk
        
        # Calculate predicted price
        predicted_price = current_price * (1 + predicted_return)
        
        # Calculate confidence based on various factors
        confidence = min(0.95, 0.5 + abs(momentum_score) * 10 + abs(trend_score) * 5)
        
        prediction = {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "predicted_price": round(predicted_price, 2),
            "predicted_return": round(predicted_return * 100, 2),
            "direction": direction,
            "confidence": round(confidence, 2),
            "days_ahead": days,
            "technical_indicators": {
                "sma_20": round(sma_20, 2),
                "sma_50": round(sma_50, 2),
                "rsi": round(rsi, 2),
                "volatility": round(volatility, 2)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache the prediction
        prediction_cache[cache_key] = prediction
        
        return prediction
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/analysis/{symbol}")
async def comprehensive_analysis(symbol: str):
    """Comprehensive stock analysis"""
    try:
        # Get quote and prediction
        quote = await get_live_quote(symbol)
        prediction = await predict_stock_price(symbol)
        
        # Get additional data
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="1y")
        
        # Calculate additional metrics
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        
        # Risk metrics
        var_95 = np.percentile(returns, 5) * 100  # 95% VaR
        max_drawdown = calculate_max_drawdown(hist['Close'])
        
        # Generate recommendation
        recommendation = generate_recommendation(prediction, quote, volatility)
        
        analysis = {
            "symbol": symbol,
            "company_name": info.get('longName', symbol),
            "sector": info.get('sector', 'Unknown'),
            "current_data": quote,
            "prediction": prediction,
            "risk_metrics": {
                "volatility": round(volatility, 3),
                "var_95": round(var_95, 2),
                "max_drawdown": round(max_drawdown, 2),
                "beta": info.get('beta', 1.0)
            },
            "recommendation": recommendation,
            "analysis_time": datetime.now().isoformat()
        }
        
        return analysis
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {str(e)}")

@app.post("/portfolio/optimize")
async def optimize_portfolio(symbols: List[str], method: str = "equal_weight"):
    """Portfolio optimization"""
    try:
        if not symbols:
            raise HTTPException(status_code=400, detail="No symbols provided")
        
        # Get data for all symbols
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1y")
                if not hist.empty:
                    data[symbol] = hist['Close']
            except:
                continue
        
        if not data:
            raise HTTPException(status_code=400, detail="No valid data found")
        
        # Create DataFrame
        df = pd.DataFrame(data)
        returns = df.pct_change().dropna()
        
        if method == "equal_weight":
            weights = {symbol: 1/len(symbols) for symbol in symbols}
        elif method == "risk_parity":
            # Simple risk parity - inverse volatility weighting
            volatilities = returns.std()
            inv_vol = 1 / volatilities
            weights = (inv_vol / inv_vol.sum()).to_dict()
        else:
            # Mean reversion strategy
            recent_returns = returns.tail(20).mean()
            weights = ((-recent_returns) / (-recent_returns).sum()).to_dict()
        
        # Calculate portfolio metrics
        portfolio_return = (returns * pd.Series(weights)).sum(axis=1)
        annual_return = portfolio_return.mean() * 252
        annual_vol = portfolio_return.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        return {
            "method": method,
            "symbols": symbols,
            "optimal_weights": {k: round(v, 3) for k, v in weights.items()},
            "expected_annual_return": round(annual_return, 3),
            "expected_volatility": round(annual_vol, 3),
            "sharpe_ratio": round(sharpe_ratio, 3),
            "optimization_date": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Optimization failed: {str(e)}")

@app.get("/market/movers")
async def get_market_movers():
    """Get top market movers"""
    try:
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX"]
        movers = []
        
        for symbol in symbols:
            try:
                quote = await get_live_quote(symbol)
                movers.append({
                    "symbol": symbol,
                    "price": quote["price"],
                    "change_percent": quote["change_percent"]
                })
            except:
                continue
        
        # Sort by absolute change
        movers.sort(key=lambda x: abs(x["change_percent"]), reverse=True)
        
        return {
            "top_movers": movers[:5],
            "biggest_gainers": [m for m in movers if m["change_percent"] > 0][:3],
            "biggest_losers": [m for m in movers if m["change_percent"] < 0][:3],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to get movers: {str(e)}")

# Helper functions
def calculate_rsi(prices, window=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_max_drawdown(prices):
    """Calculate maximum drawdown"""
    cumulative = (prices / prices.iloc[0])
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()

def generate_recommendation(prediction, quote, volatility):
    """Generate investment recommendation"""
    pred_return = prediction["predicted_return"]
    confidence = prediction["confidence"]
    change_pct = quote["change_percent"]
    
    # Simple recommendation logic
    if pred_return > 3 and confidence > 0.7:
        action = "Strong Buy"
    elif pred_return > 1 and confidence > 0.6:
        action = "Buy"
    elif pred_return < -3 and confidence > 0.7:
        action = "Strong Sell"
    elif pred_return < -1 and confidence > 0.6:
        action = "Sell"
    else:
        action = "Hold"
    
    risk_level = "High" if volatility > 0.3 else "Medium" if volatility > 0.2 else "Low"
    
    return {
        "action": action,
        "risk_level": risk_level,
        "reasoning": f"Based on {pred_return:.1f}% predicted return with {confidence:.0%} confidence",
        "time_horizon": "Short-term (1-5 days)"
    }

if __name__ == "__main__":
    print("🚀 Starting Personal Aladdin Investment Platform...")
    print("📊 Features: Portfolio Management, AI Predictions, Risk Analytics")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    
    uvicorn.run("simple_main:app", host="0.0.0.0", port=8000, reload=True)