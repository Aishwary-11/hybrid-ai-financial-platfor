"""
Indian Stock Market API Endpoints
Live data, recommendations, and analysis for Indian equities
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime

from app.core.indian_market_service import IndianMarketService
from app.core.stock_recommender import StockRecommender

router = APIRouter()

# Initialize services
market_service = IndianMarketService()
recommender = StockRecommender()


@router.get("/live-data/{symbol}")
async def get_live_stock_data(symbol: str):
    """Get live data for an Indian stock"""
    try:
        data = market_service.get_live_data([symbol])
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Get latest data point
        latest = data.iloc[-1]
        
        return {
            "symbol": symbol,
            "current_price": float(latest['Close']),
            "open": float(latest['Open']),
            "high": float(latest['High']),
            "low": float(latest['Low']),
            "volume": int(latest['Volume']),
            "timestamp": latest.name.isoformat(),
            "exchange": "NSE"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/stock-info/{symbol}")
async def get_stock_info(symbol: str):
    """Get detailed information about an Indian stock"""
    try:
        info = market_service.get_stock_info(symbol)
        if not info:
            raise HTTPException(status_code=404, detail=f"Stock info not found for {symbol}")
        
        return {
            "symbol": symbol,
            "company_name": info.get('longName', 'N/A'),
            "sector": info.get('sector', 'Unknown'),
            "industry": info.get('industry', 'Unknown'),
            "market_cap": info.get('marketCap', 0),
            "pe_ratio": info.get('trailingPE', 0),
            "pb_ratio": info.get('priceToBook', 0),
            "dividend_yield": info.get('dividendYield', 0),
            "52_week_high": info.get('fiftyTwoWeekHigh', 0),
            "52_week_low": info.get('fiftyTwoWeekLow', 0),
            "beta": info.get('beta', 0),
            "currency": "INR",
            "exchange": "NSE"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/technical-analysis/{symbol}")
async def get_technical_analysis(symbol: str, period: str = "1y"):
    """Get technical analysis for an Indian stock"""
    try:
        # Get historical data
        data = market_service.get_historical_data([symbol], period=period)
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Calculate technical indicators
        indicators = market_service.calculate_technical_indicators(data, symbol)
        if not indicators:
            raise HTTPException(status_code=400, detail="Unable to calculate technical indicators")
        
        # Generate signals
        signals = {}
        
        # RSI Signal
        rsi = indicators.get('RSI', 50)
        if rsi < 30:
            signals['RSI'] = "Oversold - Potential Buy"
        elif rsi > 70:
            signals['RSI'] = "Overbought - Potential Sell"
        else:
            signals['RSI'] = "Neutral"
        
        # MACD Signal
        macd = indicators.get('MACD', 0)
        macd_signal = indicators.get('MACD_Signal', 0)
        if macd > macd_signal:
            signals['MACD'] = "Bullish"
        else:
            signals['MACD'] = "Bearish"
        
        # Moving Average Signal
        price_vs_sma20 = indicators.get('Price_vs_SMA20', 0)
        if price_vs_sma20 > 2:
            signals['Moving_Average'] = "Above SMA20 - Bullish"
        elif price_vs_sma20 < -2:
            signals['Moving_Average'] = "Below SMA20 - Bearish"
        else:
            signals['Moving_Average'] = "Near SMA20 - Neutral"
        
        return {
            "symbol": symbol,
            "technical_indicators": indicators,
            "signals": signals,
            "analysis_date": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/recommendations")
async def get_stock_recommendations(
    investment_amount: float = Query(100000, description="Investment amount in INR"),
    risk_tolerance: str = Query("moderate", description="Risk tolerance: conservative, moderate, aggressive")
):
    """Get personalized stock recommendations for Indian market"""
    try:
        if risk_tolerance not in ["conservative", "moderate", "aggressive"]:
            raise HTTPException(status_code=400, detail="Risk tolerance must be: conservative, moderate, or aggressive")
        
        recommendations = recommender.get_buy_recommendations(investment_amount, risk_tolerance)
        
        if not recommendations:
            return {
                "message": "No recommendations available at this time",
                "recommendations": [],
                "total_recommendations": 0
            }
        
        return {
            "investment_amount": investment_amount,
            "risk_tolerance": risk_tolerance,
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/screen-stocks")
async def screen_stocks(
    max_pe: Optional[float] = Query(None, description="Maximum P/E ratio"),
    max_pb: Optional[float] = Query(None, description="Maximum P/B ratio"),
    min_roe: Optional[float] = Query(None, description="Minimum ROE percentage"),
    min_rsi: Optional[float] = Query(None, description="Minimum RSI"),
    max_rsi: Optional[float] = Query(None, description="Maximum RSI")
):
    """Screen Indian stocks based on criteria"""
    try:
        criteria = {}
        if max_pe is not None:
            criteria['max_pe'] = max_pe
        if max_pb is not None:
            criteria['max_pb'] = max_pb
        if min_roe is not None:
            criteria['min_roe'] = min_roe
        if min_rsi is not None:
            criteria['min_rsi'] = min_rsi
        if max_rsi is not None:
            criteria['max_rsi'] = max_rsi
        
        screened_stocks = recommender.screen_stocks(criteria)
        
        return {
            "screening_criteria": criteria,
            "screened_stocks": screened_stocks,
            "total_stocks": len(screened_stocks),
            "screened_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/market-sentiment")
async def get_market_sentiment():
    """Get current Indian market sentiment"""
    try:
        sentiment = market_service.get_market_sentiment()
        if not sentiment:
            raise HTTPException(status_code=400, detail="Unable to fetch market sentiment")
        
        return sentiment
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/top-movers")
async def get_top_movers(limit: int = Query(10, description="Number of stocks to return")):
    """Get top gainers and losers in Indian market"""
    try:
        movers = market_service.get_top_gainers_losers(limit)
        if not movers:
            raise HTTPException(status_code=400, detail="Unable to fetch top movers")
        
        return movers
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/sector-analysis")
async def get_sector_analysis():
    """Get sector-wise analysis of Indian market"""
    try:
        analysis = recommender.get_sector_analysis()
        if not analysis:
            raise HTTPException(status_code=400, detail="Unable to perform sector analysis")
        
        return analysis
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/market-outlook")
async def get_market_outlook():
    """Get overall market outlook and recommendations"""
    try:
        outlook = recommender.get_market_outlook()
        if not outlook:
            raise HTTPException(status_code=400, detail="Unable to generate market outlook")
        
        return outlook
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/available-stocks")
async def get_available_stocks():
    """Get list of available Indian stocks"""
    try:
        symbols = market_service.get_available_symbols()
        
        # Add sector information
        stocks_with_sectors = []
        for symbol in symbols:
            stocks_with_sectors.append({
                "symbol": symbol,
                "sector": market_service.sector_mapping.get(symbol, "Unknown"),
                "yahoo_symbol": market_service.nse_symbols.get(symbol, f"{symbol}.NS")
            })
        
        return {
            "available_stocks": stocks_with_sectors,
            "total_stocks": len(stocks_with_sectors),
            "exchanges": ["NSE"],
            "currency": "INR"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/compare-stocks")
async def compare_stocks(symbols: List[str] = Query(..., description="List of stock symbols to compare")):
    """Compare multiple Indian stocks"""
    try:
        if len(symbols) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 stocks can be compared at once")
        
        comparison = []
        
        for symbol in symbols:
            try:
                # Get stock info
                info = market_service.get_stock_info(symbol)
                
                # Get technical indicators
                data = market_service.get_historical_data([symbol], period="1y")
                tech_indicators = market_service.calculate_technical_indicators(data, symbol)
                
                # Get fundamental score
                fundamental_score = market_service.calculate_fundamental_score(symbol)
                
                stock_comparison = {
                    "symbol": symbol,
                    "sector": market_service.sector_mapping.get(symbol, "Unknown"),
                    "current_price": tech_indicators.get('Current_Price', 0),
                    "pe_ratio": info.get('trailingPE', 0),
                    "pb_ratio": info.get('priceToBook', 0),
                    "market_cap": info.get('marketCap', 0),
                    "rsi": tech_indicators.get('RSI', 0),
                    "price_vs_sma20": tech_indicators.get('Price_vs_SMA20', 0),
                    "fundamental_score": fundamental_score.get('Score_Percentage', 0),
                    "roe": fundamental_score.get('ROE', 0)
                }
                
                comparison.append(stock_comparison)
                
            except Exception as e:
                continue
        
        return {
            "stock_comparison": comparison,
            "compared_stocks": len(comparison),
            "comparison_date": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))