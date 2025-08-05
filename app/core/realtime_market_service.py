"""
Real-time Market Data Service
Live market data streaming and processing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import yfinance as yf
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
import time


class RealTimeMarketService:
    """Real-time market data service with live streaming capabilities"""
    
    def __init__(self):
        self.live_data = {}
        self.subscribers = {}
        self.data_cache = {}
        self.last_update = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    async def get_live_quote(self, symbol: str) -> Dict:
        """Get live quote for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get latest price data
            hist = ticker.history(period="2d", interval="1m")
            if hist.empty:
                return {"error": f"No data available for {symbol}"}
            
            latest = hist.iloc[-1]
            previous = hist.iloc[-2] if len(hist) > 1 else latest
            
            # Calculate metrics
            change = latest['Close'] - previous['Close']
            change_pct = (change / previous['Close']) * 100 if previous['Close'] != 0 else 0
            
            # Get additional info
            volume_avg = hist['Volume'].tail(20).mean()
            volume_ratio = latest['Volume'] / volume_avg if volume_avg > 0 else 1
            
            quote_data = {
                'symbol': symbol,
                'price': latest['Close'],
                'change': change,
                'change_percent': change_pct,
                'volume': latest['Volume'],
                'volume_ratio': volume_ratio,
                'high': latest['High'],
                'low': latest['Low'],
                'open': latest['Open'],
                'previous_close': previous['Close'],
                'timestamp': datetime.now().isoformat(),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0)
            }
            
            # Cache the data
            self.live_data[symbol] = quote_data
            self.last_update[symbol] = datetime.now()
            
            return quote_data
            
        except Exception as e:
            return {"error": f"Failed to get quote for {symbol}: {str(e)}"}
    
    async def get_live_quotes(self, symbols: List[str]) -> Dict:
        """Get live quotes for multiple symbols"""
        tasks = [self.get_live_quote(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        quotes = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                quotes[symbol] = {"error": str(result)}
            else:
                quotes[symbol] = result
        
        return {
            'quotes': quotes,
            'timestamp': datetime.now().isoformat(),
            'symbols_count': len(symbols)
        }
    
    def get_intraday_data(self, symbol: str, interval: str = "5m") -> Dict:
        """Get intraday data for technical analysis"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval=interval)
            
            if data.empty:
                return {"error": f"No intraday data for {symbol}"}
            
            # Calculate technical indicators
            data['sma_20'] = data['Close'].rolling(window=20).mean()
            data['rsi'] = self._calculate_rsi(data['Close'])
            data['volume_sma'] = data['Volume'].rolling(window=20).mean()
            
            # Convert to dict for JSON serialization
            intraday_data = {
                'symbol': symbol,
                'interval': interval,
                'data_points': len(data),
                'latest_price': data['Close'].iloc[-1],
                'price_data': {
                    'timestamps': [ts.isoformat() for ts in data.index],
                    'open': data['Open'].tolist(),
                    'high': data['High'].tolist(),
                    'low': data['Low'].tolist(),
                    'close': data['Close'].tolist(),
                    'volume': data['Volume'].tolist(),
                    'sma_20': data['sma_20'].fillna(0).tolist(),
                    'rsi': data['rsi'].fillna(50).tolist()
                },
                'summary': {
                    'high_of_day': data['High'].max(),
                    'low_of_day': data['Low'].min(),
                    'volume_total': data['Volume'].sum(),
                    'avg_volume': data['Volume'].mean(),
                    'price_range': data['High'].max() - data['Low'].min(),
                    'current_rsi': data['rsi'].iloc[-1] if not data['rsi'].isna().all() else 50
                }
            }
            
            return intraday_data
            
        except Exception as e:
            return {"error": f"Failed to get intraday data for {symbol}: {str(e)}"}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    async def get_market_movers(self, market: str = "US") -> Dict:
        """Get top market movers (gainers/losers)"""
        try:
            # Popular symbols for different markets
            symbols_map = {
                "US": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX", "AMD", "INTC"],
                "CRYPTO": ["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "LINK-USD"],
                "FOREX": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"]
            }
            
            symbols = symbols_map.get(market, symbols_map["US"])
            
            movers_data = []
            
            for symbol in symbols:
                quote = await self.get_live_quote(symbol)
                if "error" not in quote:
                    movers_data.append({
                        'symbol': symbol,
                        'price': quote['price'],
                        'change': quote['change'],
                        'change_percent': quote['change_percent'],
                        'volume': quote['volume']
                    })
            
            # Sort by change percentage
            gainers = sorted([m for m in movers_data if m['change_percent'] > 0], 
                           key=lambda x: x['change_percent'], reverse=True)[:5]
            losers = sorted([m for m in movers_data if m['change_percent'] < 0], 
                          key=lambda x: x['change_percent'])[:5]
            
            return {
                'market': market,
                'gainers': gainers,
                'losers': losers,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to get market movers: {str(e)}"}
    
    async def get_market_overview(self) -> Dict:
        """Get overall market overview"""
        try:
            # Major indices
            indices = {
                'S&P 500': '^GSPC',
                'Dow Jones': '^DJI',
                'NASDAQ': '^IXIC',
                'Russell 2000': '^RUT',
                'VIX': '^VIX'
            }
            
            overview_data = {}
            
            for name, symbol in indices.items():
                quote = await self.get_live_quote(symbol)
                if "error" not in quote:
                    overview_data[name] = {
                        'symbol': symbol,
                        'price': quote['price'],
                        'change': quote['change'],
                        'change_percent': quote['change_percent']
                    }
            
            # Market sentiment indicators
            vix_level = overview_data.get('VIX', {}).get('price', 20)
            market_sentiment = self._interpret_vix(vix_level)
            
            return {
                'indices': overview_data,
                'market_sentiment': market_sentiment,
                'vix_level': vix_level,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to get market overview: {str(e)}"}
    
    def _interpret_vix(self, vix_level: float) -> str:
        """Interpret VIX level for market sentiment"""
        if vix_level < 12:
            return "Very Low Volatility - Complacent"
        elif vix_level < 20:
            return "Low Volatility - Calm"
        elif vix_level < 30:
            return "Normal Volatility - Neutral"
        elif vix_level < 40:
            return "High Volatility - Fearful"
        else:
            return "Very High Volatility - Panic"
    
    async def get_sector_performance(self) -> Dict:
        """Get sector performance data"""
        try:
            # Sector ETFs
            sectors = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financials': 'XLF',
                'Consumer Discretionary': 'XLY',
                'Communication Services': 'XLC',
                'Industrials': 'XLI',
                'Consumer Staples': 'XLP',
                'Energy': 'XLE',
                'Utilities': 'XLU',
                'Real Estate': 'XLRE',
                'Materials': 'XLB'
            }
            
            sector_data = {}
            
            for sector_name, etf_symbol in sectors.items():
                quote = await self.get_live_quote(etf_symbol)
                if "error" not in quote:
                    sector_data[sector_name] = {
                        'symbol': etf_symbol,
                        'price': quote['price'],
                        'change_percent': quote['change_percent']
                    }
            
            # Sort by performance
            sorted_sectors = sorted(sector_data.items(), 
                                  key=lambda x: x[1]['change_percent'], reverse=True)
            
            return {
                'sector_performance': dict(sorted_sectors),
                'best_performer': sorted_sectors[0] if sorted_sectors else None,
                'worst_performer': sorted_sectors[-1] if sorted_sectors else None,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to get sector performance: {str(e)}"}
    
    async def get_options_data(self, symbol: str) -> Dict:
        """Get options data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get options expiration dates
            exp_dates = ticker.options
            if not exp_dates:
                return {"error": f"No options data available for {symbol}"}
            
            # Get options for nearest expiration
            nearest_exp = exp_dates[0]
            options_chain = ticker.option_chain(nearest_exp)
            
            calls = options_chain.calls
            puts = options_chain.puts
            
            # Calculate put/call ratio
            total_call_volume = calls['volume'].fillna(0).sum()
            total_put_volume = puts['volume'].fillna(0).sum()
            put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
            
            # Find at-the-money options
            current_price = await self.get_live_quote(symbol)
            if "error" in current_price:
                return current_price
            
            price = current_price['price']
            
            # Find closest strikes to current price
            calls['distance'] = abs(calls['strike'] - price)
            puts['distance'] = abs(puts['strike'] - price)
            
            atm_call = calls.loc[calls['distance'].idxmin()] if not calls.empty else None
            atm_put = puts.loc[puts['distance'].idxmin()] if not puts.empty else None
            
            return {
                'symbol': symbol,
                'current_price': price,
                'expiration_date': nearest_exp,
                'put_call_ratio': put_call_ratio,
                'total_call_volume': total_call_volume,
                'total_put_volume': total_put_volume,
                'atm_call': {
                    'strike': atm_call['strike'] if atm_call is not None else 0,
                    'last_price': atm_call['lastPrice'] if atm_call is not None else 0,
                    'implied_volatility': atm_call['impliedVolatility'] if atm_call is not None else 0
                } if atm_call is not None else None,
                'atm_put': {
                    'strike': atm_put['strike'] if atm_put is not None else 0,
                    'last_price': atm_put['lastPrice'] if atm_put is not None else 0,
                    'implied_volatility': atm_put['impliedVolatility'] if atm_put is not None else 0
                } if atm_put is not None else None,
                'sentiment': 'bullish' if put_call_ratio < 0.8 else 'bearish' if put_call_ratio > 1.2 else 'neutral',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to get options data for {symbol}: {str(e)}"}
    
    async def get_earnings_calendar(self, days_ahead: int = 7) -> Dict:
        """Get upcoming earnings calendar"""
        try:
            # This is a simplified version - in production, you'd use a dedicated earnings API
            popular_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX"]
            
            earnings_data = []
            
            for symbol in popular_stocks:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Get earnings date if available
                earnings_date = info.get('earningsDate')
                if earnings_date:
                    earnings_data.append({
                        'symbol': symbol,
                        'company_name': info.get('longName', symbol),
                        'earnings_date': earnings_date,
                        'estimated_eps': info.get('forwardEps', 0),
                        'previous_eps': info.get('trailingEps', 0)
                    })
            
            return {
                'earnings_calendar': earnings_data,
                'days_ahead': days_ahead,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to get earnings calendar: {str(e)}"}
    
    async def start_live_stream(self, symbols: List[str], interval: int = 60):
        """Start live data streaming for symbols"""
        print(f"🔴 Starting live stream for {symbols} (update every {interval}s)")
        
        while True:
            try:
                # Update live data for all symbols
                for symbol in symbols:
                    quote = await self.get_live_quote(symbol)
                    if "error" not in quote:
                        print(f"📊 {symbol}: ${quote['price']:.2f} ({quote['change_percent']:+.2f}%)")
                
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                print("🛑 Live stream stopped")
                break
            except Exception as e:
                print(f"❌ Stream error: {e}")
                await asyncio.sleep(interval)