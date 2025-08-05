"""
Indian Stock Market Data Service
Real-time data and analysis for Indian equities
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json


class IndianMarketService:
    """Service for Indian stock market data and analysis"""
    
    def __init__(self):
        self.nse_symbols = self._load_nse_symbols()
        self.sector_mapping = self._load_sector_mapping()
        
    def _load_nse_symbols(self) -> Dict[str, str]:
        """Load popular NSE symbols with their Yahoo Finance tickers"""
        return {
            # Large Cap Stocks
            "RELIANCE": "RELIANCE.NS",
            "TCS": "TCS.NS", 
            "HDFCBANK": "HDFCBANK.NS",
            "INFY": "INFY.NS",
            "HINDUNILVR": "HINDUNILVR.NS",
            "ICICIBANK": "ICICIBANK.NS",
            "KOTAKBANK": "KOTAKBANK.NS",
            "BHARTIARTL": "BHARTIARTL.NS",
            "ITC": "ITC.NS",
            "SBIN": "SBIN.NS",
            "ASIANPAINT": "ASIANPAINT.NS",
            "MARUTI": "MARUTI.NS",
            "BAJFINANCE": "BAJFINANCE.NS",
            "HCLTECH": "HCLTECH.NS",
            "WIPRO": "WIPRO.NS",
            "ULTRACEMCO": "ULTRACEMCO.NS",
            "TITAN": "TITAN.NS",
            "NESTLEIND": "NESTLEIND.NS",
            "POWERGRID": "POWERGRID.NS",
            "NTPC": "NTPC.NS",
            
            # Mid Cap Stocks
            "ADANIPORTS": "ADANIPORTS.NS",
            "TECHM": "TECHM.NS",
            "SUNPHARMA": "SUNPHARMA.NS",
            "JSWSTEEL": "JSWSTEEL.NS",
            "TATAMOTORS": "TATAMOTORS.NS",
            "INDUSINDBK": "INDUSINDBK.NS",
            "BAJAJFINSV": "BAJAJFINSV.NS",
            "DRREDDY": "DRREDDY.NS",
            "CIPLA": "CIPLA.NS",
            "DIVISLAB": "DIVISLAB.NS",
            
            # ETFs and Indices
            "NIFTY50": "^NSEI",
            "SENSEX": "^BSESN",
            "NIFTYNEXT50": "NIFTYNEXT50.NS",
            "NIFTYBANK": "NIFTYBANK.NS"
        }
    
    def _load_sector_mapping(self) -> Dict[str, str]:
        """Map stocks to their sectors"""
        return {
            "RELIANCE": "Energy",
            "TCS": "Information Technology", 
            "HDFCBANK": "Financial Services",
            "INFY": "Information Technology",
            "HINDUNILVR": "FMCG",
            "ICICIBANK": "Financial Services",
            "KOTAKBANK": "Financial Services",
            "BHARTIARTL": "Telecom",
            "ITC": "FMCG",
            "SBIN": "Financial Services",
            "ASIANPAINT": "Paints",
            "MARUTI": "Automobile",
            "BAJFINANCE": "Financial Services",
            "HCLTECH": "Information Technology",
            "WIPRO": "Information Technology",
            "ULTRACEMCO": "Cement",
            "TITAN": "Consumer Discretionary",
            "NESTLEIND": "FMCG",
            "POWERGRID": "Power",
            "NTPC": "Power",
            "ADANIPORTS": "Infrastructure",
            "TECHM": "Information Technology",
            "SUNPHARMA": "Pharmaceuticals",
            "JSWSTEEL": "Steel",
            "TATAMOTORS": "Automobile",
            "INDUSINDBK": "Financial Services",
            "BAJAJFINSV": "Financial Services",
            "DRREDDY": "Pharmaceuticals",
            "CIPLA": "Pharmaceuticals",
            "DIVISLAB": "Pharmaceuticals"
        }
    
    def get_live_data(self, symbols: List[str]) -> pd.DataFrame:
        """Get live market data for Indian stocks"""
        try:
            # Convert to Yahoo Finance format
            yf_symbols = []
            for symbol in symbols:
                if symbol in self.nse_symbols:
                    yf_symbols.append(self.nse_symbols[symbol])
                elif symbol.endswith('.NS'):
                    yf_symbols.append(symbol)
                else:
                    yf_symbols.append(f"{symbol}.NS")
            
            # Fetch data
            data = yf.download(yf_symbols, period="1d", interval="1m")
            return data
            
        except Exception as e:
            print(f"Error fetching live data: {e}")
            return pd.DataFrame()
    
    def get_historical_data(self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """Get historical data for Indian stocks"""
        try:
            # Convert to Yahoo Finance format
            yf_symbols = []
            for symbol in symbols:
                if symbol in self.nse_symbols:
                    yf_symbols.append(self.nse_symbols[symbol])
                elif symbol.endswith('.NS'):
                    yf_symbols.append(symbol)
                else:
                    yf_symbols.append(f"{symbol}.NS")
            
            # Fetch historical data
            data = yf.download(yf_symbols, period=period)
            return data
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Get detailed stock information"""
        try:
            yf_symbol = self.nse_symbols.get(symbol, f"{symbol}.NS")
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info
            
            # Add Indian market specific info
            info['sector'] = self.sector_mapping.get(symbol, "Unknown")
            info['exchange'] = "NSE"
            info['currency'] = "INR"
            
            return info
            
        except Exception as e:
            print(f"Error fetching stock info: {e}")
            return {}
    
    def calculate_technical_indicators(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Calculate technical indicators for a stock"""
        try:
            if data.empty:
                return {}
            
            # Get close prices
            if len(data.columns.levels) > 1:  # Multi-symbol data
                close = data['Close'][symbol] if symbol in data['Close'].columns else data['Close'].iloc[:, 0]
            else:
                close = data['Close']
            
            indicators = {}
            
            # Moving Averages
            indicators['SMA_20'] = close.rolling(window=20).mean().iloc[-1]
            indicators['SMA_50'] = close.rolling(window=50).mean().iloc[-1]
            indicators['SMA_200'] = close.rolling(window=200).mean().iloc[-1]
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['RSI'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # MACD
            exp1 = close.ewm(span=12).mean()
            exp2 = close.ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            indicators['MACD'] = macd.iloc[-1]
            indicators['MACD_Signal'] = signal.iloc[-1]
            indicators['MACD_Histogram'] = (macd - signal).iloc[-1]
            
            # Bollinger Bands
            sma_20 = close.rolling(window=20).mean()
            std_20 = close.rolling(window=20).std()
            indicators['BB_Upper'] = (sma_20 + (std_20 * 2)).iloc[-1]
            indicators['BB_Lower'] = (sma_20 - (std_20 * 2)).iloc[-1]
            indicators['BB_Middle'] = sma_20.iloc[-1]
            
            # Current Price
            indicators['Current_Price'] = close.iloc[-1]
            
            # Price vs Moving Averages
            current_price = close.iloc[-1]
            indicators['Price_vs_SMA20'] = (current_price / indicators['SMA_20'] - 1) * 100
            indicators['Price_vs_SMA50'] = (current_price / indicators['SMA_50'] - 1) * 100
            indicators['Price_vs_SMA200'] = (current_price / indicators['SMA_200'] - 1) * 100
            
            return indicators
            
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            return {}
    
    def calculate_fundamental_score(self, symbol: str) -> Dict:
        """Calculate fundamental analysis score"""
        try:
            info = self.get_stock_info(symbol)
            
            if not info:
                return {}
            
            score = 0
            max_score = 10
            factors = {}
            
            # P/E Ratio (lower is better, but not too low)
            pe_ratio = info.get('trailingPE', 0)
            if 10 <= pe_ratio <= 25:
                pe_score = 2
            elif 5 <= pe_ratio < 10 or 25 < pe_ratio <= 35:
                pe_score = 1
            else:
                pe_score = 0
            
            factors['PE_Ratio'] = pe_ratio
            factors['PE_Score'] = pe_score
            score += pe_score
            
            # P/B Ratio (lower is better)
            pb_ratio = info.get('priceToBook', 0)
            if pb_ratio <= 1.5:
                pb_score = 2
            elif pb_ratio <= 3:
                pb_score = 1
            else:
                pb_score = 0
            
            factors['PB_Ratio'] = pb_ratio
            factors['PB_Score'] = pb_score
            score += pb_score
            
            # ROE (higher is better)
            roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
            if roe >= 15:
                roe_score = 2
            elif roe >= 10:
                roe_score = 1
            else:
                roe_score = 0
            
            factors['ROE'] = roe
            factors['ROE_Score'] = roe_score
            score += roe_score
            
            # Debt to Equity (lower is better)
            debt_equity = info.get('debtToEquity', 0)
            if debt_equity <= 0.5:
                de_score = 2
            elif debt_equity <= 1:
                de_score = 1
            else:
                de_score = 0
            
            factors['Debt_to_Equity'] = debt_equity
            factors['DE_Score'] = de_score
            score += de_score
            
            # Revenue Growth
            revenue_growth = info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0
            if revenue_growth >= 15:
                growth_score = 2
            elif revenue_growth >= 5:
                growth_score = 1
            else:
                growth_score = 0
            
            factors['Revenue_Growth'] = revenue_growth
            factors['Growth_Score'] = growth_score
            score += growth_score
            
            # Final score
            factors['Total_Score'] = score
            factors['Max_Score'] = max_score
            factors['Score_Percentage'] = (score / max_score) * 100
            
            return factors
            
        except Exception as e:
            print(f"Error calculating fundamental score: {e}")
            return {}
    
    def get_market_sentiment(self) -> Dict:
        """Get overall market sentiment indicators"""
        try:
            # Get Nifty 50 data
            nifty_data = yf.download("^NSEI", period="5d")
            
            if nifty_data.empty:
                return {}
            
            current_price = nifty_data['Close'].iloc[-1]
            prev_close = nifty_data['Close'].iloc[-2]
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            
            # Simple sentiment based on market movement
            if change_pct > 1:
                sentiment = "Very Bullish"
            elif change_pct > 0.5:
                sentiment = "Bullish"
            elif change_pct > -0.5:
                sentiment = "Neutral"
            elif change_pct > -1:
                sentiment = "Bearish"
            else:
                sentiment = "Very Bearish"
            
            return {
                'nifty_current': current_price,
                'nifty_change': change,
                'nifty_change_pct': change_pct,
                'market_sentiment': sentiment,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting market sentiment: {e}")
            return {}
    
    def get_top_gainers_losers(self, limit: int = 10) -> Dict:
        """Get top gainers and losers from tracked stocks"""
        try:
            symbols = list(self.nse_symbols.keys())[:20]  # Top 20 stocks
            data = self.get_historical_data(symbols, period="2d")
            
            if data.empty:
                return {}
            
            gainers = []
            losers = []
            
            for symbol in symbols:
                try:
                    yf_symbol = self.nse_symbols[symbol]
                    
                    if len(data.columns.levels) > 1:  # Multi-symbol data
                        if symbol in data['Close'].columns:
                            close_prices = data['Close'][symbol]
                        else:
                            continue
                    else:
                        close_prices = data['Close']
                    
                    if len(close_prices) >= 2:
                        current = close_prices.iloc[-1]
                        previous = close_prices.iloc[-2]
                        change_pct = ((current - previous) / previous) * 100
                        
                        stock_data = {
                            'symbol': symbol,
                            'current_price': current,
                            'change_pct': change_pct,
                            'sector': self.sector_mapping.get(symbol, "Unknown")
                        }
                        
                        if change_pct > 0:
                            gainers.append(stock_data)
                        else:
                            losers.append(stock_data)
                            
                except Exception as e:
                    continue
            
            # Sort and limit
            gainers = sorted(gainers, key=lambda x: x['change_pct'], reverse=True)[:limit]
            losers = sorted(losers, key=lambda x: x['change_pct'])[:limit]
            
            return {
                'top_gainers': gainers,
                'top_losers': losers,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting gainers/losers: {e}")
            return {}
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available Indian stock symbols"""
        return list(self.nse_symbols.keys())