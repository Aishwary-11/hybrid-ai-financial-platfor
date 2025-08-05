"""
Live Predictions API Endpoints
Real-time market predictions and analysis
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import List, Dict, Optional
import asyncio
from datetime import datetime

from app.core.live_prediction_engine import LivePredictionEngine
from app.core.realtime_market_service import RealTimeMarketService

router = APIRouter()

# Initialize engines
prediction_engine = LivePredictionEngine()
market_service = RealTimeMarketService()


@router.post("/train/{symbol}")
async def train_prediction_model(symbol: str):
    """Train ML models for a specific symbol"""
    try:
        result = prediction_engine.train_prediction_models(symbol)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/predict/{symbol}")
async def get_live_prediction(symbol: str):
    """Get live prediction for a symbol"""
    try:
        prediction = prediction_engine.make_live_prediction(symbol)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict/batch")
async def batch_predictions(symbols: List[str] = Query(...)):
    """Get predictions for multiple symbols"""
    try:
        results = prediction_engine.batch_predict(symbols)
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/sentiment/market")
async def get_market_sentiment(symbols: List[str] = Query(default=["SPY", "QQQ", "IWM", "VTI"])):
    """Get overall market sentiment analysis"""
    try:
        sentiment = prediction_engine.get_market_sentiment(symbols)
        return sentiment
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/accuracy/{symbol}")
async def get_prediction_accuracy(symbol: str):
    """Get prediction accuracy metrics for a symbol"""
    try:
        accuracy = prediction_engine.get_prediction_accuracy(symbol)
        return accuracy
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/quote/{symbol}")
async def get_live_quote(symbol: str):
    """Get live quote for a symbol"""
    try:
        quote = await market_service.get_live_quote(symbol)
        return quote
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/quotes")
async def get_live_quotes(symbols: List[str] = Query(...)):
    """Get live quotes for multiple symbols"""
    try:
        quotes = await market_service.get_live_quotes(symbols)
        return quotes
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/intraday/{symbol}")
async def get_intraday_data(
    symbol: str,
    interval: str = "5m"  # 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
):
    """Get intraday data with technical indicators"""
    try:
        data = market_service.get_intraday_data(symbol, interval)
        return data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/market/overview")
async def get_market_overview():
    """Get overall market overview"""
    try:
        overview = await market_service.get_market_overview()
        return overview
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/market/movers")
async def get_market_movers(market: str = "US"):
    """Get top market movers (gainers/losers)"""
    try:
        movers = await market_service.get_market_movers(market)
        return movers
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/market/sectors")
async def get_sector_performance():
    """Get sector performance data"""
    try:
        sectors = await market_service.get_sector_performance()
        return sectors
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/options/{symbol}")
async def get_options_data(symbol: str):
    """Get options data and sentiment for a symbol"""
    try:
        options = await market_service.get_options_data(symbol)
        return options
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/earnings/calendar")
async def get_earnings_calendar(days_ahead: int = 7):
    """Get upcoming earnings calendar"""
    try:
        earnings = await market_service.get_earnings_calendar(days_ahead)
        return earnings
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/analysis/comprehensive/{symbol}")
async def get_comprehensive_analysis(symbol: str):
    """Get comprehensive analysis combining predictions, quotes, and technical data"""
    try:
        # Get all data concurrently
        tasks = [
            prediction_engine.make_live_prediction(symbol),
            market_service.get_live_quote(symbol),
            market_service.get_intraday_data(symbol, "5m"),
            market_service.get_options_data(symbol)
        ]
        
        # Execute tasks concurrently
        prediction_result = prediction_engine.make_live_prediction(symbol)
        quote_result = await market_service.get_live_quote(symbol)
        intraday_result = market_service.get_intraday_data(symbol, "5m")
        options_result = await market_service.get_options_data(symbol)
        
        # Combine results
        comprehensive_analysis = {
            'symbol': symbol,
            'analysis_time': datetime.now().isoformat(),
            'live_quote': quote_result,
            'predictions': prediction_result,
            'intraday_data': intraday_result,
            'options_data': options_result,
            'summary': {
                'current_price': quote_result.get('price', 0),
                'predicted_1d_return': prediction_result.get('predictions', {}).get('target_1d', {}).get('predicted_return_pct', 0),
                'predicted_5d_return': prediction_result.get('predictions', {}).get('target_5d', {}).get('predicted_return_pct', 0),
                'current_rsi': intraday_result.get('summary', {}).get('current_rsi', 50),
                'options_sentiment': options_result.get('sentiment', 'neutral'),
                'recommendation': 'hold'  # Default recommendation
            }
        }
        
        # Generate recommendation based on analysis
        comprehensive_analysis['summary']['recommendation'] = _generate_recommendation(comprehensive_analysis)
        
        return comprehensive_analysis
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/watchlist/analyze")
async def analyze_watchlist(symbols: List[str] = Query(...)):
    """Analyze a watchlist of symbols"""
    try:
        watchlist_analysis = {}
        
        for symbol in symbols:
            # Get prediction and quote
            prediction = prediction_engine.make_live_prediction(symbol)
            quote = await market_service.get_live_quote(symbol)
            
            if "error" not in prediction and "error" not in quote:
                watchlist_analysis[symbol] = {
                    'current_price': quote['price'],
                    'change_percent': quote['change_percent'],
                    'predicted_1d_return': prediction.get('predictions', {}).get('target_1d', {}).get('predicted_return_pct', 0),
                    'predicted_5d_return': prediction.get('predictions', {}).get('target_5d', {}).get('predicted_return_pct', 0),
                    'confidence_1d': prediction.get('predictions', {}).get('target_1d', {}).get('confidence', 0),
                    'volume_ratio': quote.get('volume_ratio', 1),
                    'recommendation': _get_simple_recommendation(
                        prediction.get('predictions', {}).get('target_1d', {}).get('predicted_return_pct', 0),
                        quote.get('change_percent', 0)
                    )
                }
        
        # Sort by predicted 1-day return
        sorted_watchlist = dict(sorted(watchlist_analysis.items(), 
                                     key=lambda x: x[1]['predicted_1d_return'], reverse=True))
        
        return {
            'watchlist_analysis': sorted_watchlist,
            'top_picks': list(sorted_watchlist.keys())[:3],
            'symbols_analyzed': len(watchlist_analysis),
            'analysis_time': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/alerts/generate/{symbol}")
async def generate_trading_alerts(symbol: str):
    """Generate trading alerts based on technical and prediction analysis"""
    try:
        # Get comprehensive data
        prediction = prediction_engine.make_live_prediction(symbol)
        quote = await market_service.get_live_quote(symbol)
        intraday = market_service.get_intraday_data(symbol, "5m")
        
        alerts = []
        
        if "error" not in prediction and "error" not in quote and "error" not in intraday:
            # Price-based alerts
            if abs(quote['change_percent']) > 5:
                alerts.append({
                    'type': 'price_movement',
                    'severity': 'high',
                    'message': f"{symbol} moved {quote['change_percent']:+.2f}% today",
                    'action': 'monitor'
                })
            
            # Volume alerts
            if quote.get('volume_ratio', 1) > 2:
                alerts.append({
                    'type': 'volume_spike',
                    'severity': 'medium',
                    'message': f"{symbol} volume is {quote['volume_ratio']:.1f}x above average",
                    'action': 'investigate'
                })
            
            # Prediction alerts
            pred_1d = prediction.get('predictions', {}).get('target_1d', {}).get('predicted_return_pct', 0)
            if abs(pred_1d) > 3:
                alerts.append({
                    'type': 'prediction_alert',
                    'severity': 'medium',
                    'message': f"Model predicts {pred_1d:+.2f}% move in 1 day",
                    'action': 'consider_position'
                })
            
            # RSI alerts
            current_rsi = intraday.get('summary', {}).get('current_rsi', 50)
            if current_rsi > 70:
                alerts.append({
                    'type': 'technical_overbought',
                    'severity': 'medium',
                    'message': f"{symbol} RSI at {current_rsi:.1f} - potentially overbought",
                    'action': 'consider_sell'
                })
            elif current_rsi < 30:
                alerts.append({
                    'type': 'technical_oversold',
                    'severity': 'medium',
                    'message': f"{symbol} RSI at {current_rsi:.1f} - potentially oversold",
                    'action': 'consider_buy'
                })
        
        return {
            'symbol': symbol,
            'alerts': alerts,
            'alert_count': len(alerts),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _generate_recommendation(analysis: Dict) -> str:
    """Generate trading recommendation based on comprehensive analysis"""
    try:
        pred_1d = analysis['predictions'].get('predictions', {}).get('target_1d', {}).get('predicted_return_pct', 0)
        pred_5d = analysis['predictions'].get('predictions', {}).get('target_5d', {}).get('predicted_return_pct', 0)
        current_change = analysis['live_quote'].get('change_percent', 0)
        rsi = analysis['intraday_data'].get('summary', {}).get('current_rsi', 50)
        options_sentiment = analysis['options_data'].get('sentiment', 'neutral')
        
        # Scoring system
        score = 0
        
        # Prediction score
        if pred_1d > 2:
            score += 2
        elif pred_1d > 0:
            score += 1
        elif pred_1d < -2:
            score -= 2
        elif pred_1d < 0:
            score -= 1
        
        # Medium-term prediction
        if pred_5d > 3:
            score += 1
        elif pred_5d < -3:
            score -= 1
        
        # RSI score
        if rsi < 30:
            score += 1  # Oversold, potential buy
        elif rsi > 70:
            score -= 1  # Overbought, potential sell
        
        # Options sentiment
        if options_sentiment == 'bullish':
            score += 1
        elif options_sentiment == 'bearish':
            score -= 1
        
        # Generate recommendation
        if score >= 3:
            return 'strong_buy'
        elif score >= 1:
            return 'buy'
        elif score <= -3:
            return 'strong_sell'
        elif score <= -1:
            return 'sell'
        else:
            return 'hold'
            
    except:
        return 'hold'


def _get_simple_recommendation(predicted_return: float, current_change: float) -> str:
    """Simple recommendation based on predicted return"""
    if predicted_return > 3:
        return 'strong_buy'
    elif predicted_return > 1:
        return 'buy'
    elif predicted_return < -3:
        return 'strong_sell'
    elif predicted_return < -1:
        return 'sell'
    else:
        return 'hold'