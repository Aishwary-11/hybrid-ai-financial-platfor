"""
AI-Powered Stock Recommendation Engine
Intelligent stock screening and recommendations for Indian market
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

from .indian_market_service import IndianMarketService


class StockRecommender:
    """AI-powered stock recommendation system"""
    
    def __init__(self):
        self.market_service = IndianMarketService()
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def screen_stocks(self, criteria: Dict) -> List[Dict]:
        """Screen stocks based on multiple criteria"""
        try:
            symbols = self.market_service.get_available_symbols()
            screened_stocks = []
            
            for symbol in symbols:
                try:
                    # Get stock data
                    data = self.market_service.get_historical_data([symbol], period="1y")
                    if data.empty:
                        continue
                    
                    # Calculate technical indicators
                    tech_indicators = self.market_service.calculate_technical_indicators(data, symbol)
                    if not tech_indicators:
                        continue
                    
                    # Calculate fundamental score
                    fundamental_score = self.market_service.calculate_fundamental_score(symbol)
                    
                    # Apply screening criteria
                    if self._meets_criteria(tech_indicators, fundamental_score, criteria):
                        stock_info = {
                            'symbol': symbol,
                            'sector': self.market_service.sector_mapping.get(symbol, "Unknown"),
                            'current_price': tech_indicators.get('Current_Price', 0),
                            'rsi': tech_indicators.get('RSI', 0),
                            'price_vs_sma20': tech_indicators.get('Price_vs_SMA20', 0),
                            'price_vs_sma50': tech_indicators.get('Price_vs_SMA50', 0),
                            'fundamental_score': fundamental_score.get('Score_Percentage', 0),
                            'pe_ratio': fundamental_score.get('PE_Ratio', 0),
                            'pb_ratio': fundamental_score.get('PB_Ratio', 0),
                            'roe': fundamental_score.get('ROE', 0),
                            'debt_equity': fundamental_score.get('Debt_to_Equity', 0)
                        }
                        screened_stocks.append(stock_info)
                        
                except Exception as e:
                    continue
            
            return screened_stocks
            
        except Exception as e:
            print(f"Error screening stocks: {e}")
            return []
    
    def _meets_criteria(self, tech_indicators: Dict, fundamental_score: Dict, criteria: Dict) -> bool:
        """Check if stock meets screening criteria"""
        try:
            # RSI criteria
            rsi = tech_indicators.get('RSI', 50)
            if criteria.get('min_rsi') and rsi < criteria['min_rsi']:
                return False
            if criteria.get('max_rsi') and rsi > criteria['max_rsi']:
                return False
            
            # Price vs Moving Average criteria
            price_vs_sma20 = tech_indicators.get('Price_vs_SMA20', 0)
            if criteria.get('min_price_vs_sma20') and price_vs_sma20 < criteria['min_price_vs_sma20']:
                return False
            
            # Fundamental criteria
            pe_ratio = fundamental_score.get('PE_Ratio', 0)
            if criteria.get('max_pe') and pe_ratio > criteria['max_pe']:
                return False
            
            pb_ratio = fundamental_score.get('PB_Ratio', 0)
            if criteria.get('max_pb') and pb_ratio > criteria['max_pb']:
                return False
            
            roe = fundamental_score.get('ROE', 0)
            if criteria.get('min_roe') and roe < criteria['min_roe']:
                return False
            
            return True
            
        except Exception as e:
            return False
    
    def get_buy_recommendations(self, investment_amount: float = 100000, 
                              risk_tolerance: str = "moderate") -> List[Dict]:
        """Get personalized buy recommendations"""
        try:
            # Define screening criteria based on risk tolerance
            if risk_tolerance == "conservative":
                criteria = {
                    'max_pe': 20,
                    'max_pb': 2,
                    'min_roe': 12,
                    'min_rsi': 30,
                    'max_rsi': 70
                }
            elif risk_tolerance == "aggressive":
                criteria = {
                    'max_pe': 40,
                    'max_pb': 5,
                    'min_roe': 8,
                    'min_rsi': 20,
                    'max_rsi': 80,
                    'min_price_vs_sma20': -5  # Allow some momentum stocks
                }
            else:  # moderate
                criteria = {
                    'max_pe': 30,
                    'max_pb': 3,
                    'min_roe': 10,
                    'min_rsi': 25,
                    'max_rsi': 75
                }
            
            # Screen stocks
            screened_stocks = self.screen_stocks(criteria)
            
            if not screened_stocks:
                return []
            
            # Calculate recommendation scores
            recommendations = []
            for stock in screened_stocks:
                score = self._calculate_recommendation_score(stock, risk_tolerance)
                stock['recommendation_score'] = score
                stock['recommendation_reason'] = self._get_recommendation_reason(stock)
                recommendations.append(stock)
            
            # Sort by recommendation score
            recommendations = sorted(recommendations, key=lambda x: x['recommendation_score'], reverse=True)
            
            # Limit to top 10 and add allocation suggestions
            top_recommendations = recommendations[:10]
            
            # Suggest allocation
            total_score = sum(stock['recommendation_score'] for stock in top_recommendations)
            for stock in top_recommendations:
                allocation_pct = (stock['recommendation_score'] / total_score) * 100
                stock['suggested_allocation_pct'] = round(allocation_pct, 2)
                stock['suggested_amount'] = round(investment_amount * allocation_pct / 100, 2)
            
            return top_recommendations
            
        except Exception as e:
            print(f"Error getting buy recommendations: {e}")
            return []
    
    def _calculate_recommendation_score(self, stock: Dict, risk_tolerance: str) -> float:
        """Calculate recommendation score for a stock"""
        try:
            score = 0
            
            # Fundamental score (40% weight)
            fundamental_score = stock.get('fundamental_score', 0)
            score += (fundamental_score / 100) * 40
            
            # Technical score (30% weight)
            rsi = stock.get('rsi', 50)
            # Prefer RSI between 30-70 (not overbought/oversold)
            if 30 <= rsi <= 70:
                rsi_score = 30
            elif 20 <= rsi < 30 or 70 < rsi <= 80:
                rsi_score = 20
            else:
                rsi_score = 10
            
            score += rsi_score
            
            # Price momentum (20% weight)
            price_vs_sma20 = stock.get('price_vs_sma20', 0)
            if price_vs_sma20 > 5:  # Strong uptrend
                momentum_score = 20
            elif price_vs_sma20 > 0:  # Mild uptrend
                momentum_score = 15
            elif price_vs_sma20 > -5:  # Sideways
                momentum_score = 10
            else:  # Downtrend
                momentum_score = 5
            
            score += momentum_score
            
            # Sector diversification bonus (10% weight)
            sector = stock.get('sector', '')
            if sector in ['Information Technology', 'Financial Services', 'FMCG']:
                score += 10  # Bonus for stable sectors
            elif sector in ['Pharmaceuticals', 'Automobile']:
                score += 8
            else:
                score += 5
            
            # Risk adjustment
            if risk_tolerance == "conservative":
                # Penalize high PE stocks
                pe_ratio = stock.get('pe_ratio', 0)
                if pe_ratio > 25:
                    score -= 10
            elif risk_tolerance == "aggressive":
                # Bonus for growth stocks
                if price_vs_sma20 > 10:
                    score += 5
            
            return max(0, min(100, score))  # Ensure score is between 0-100
            
        except Exception as e:
            return 0
    
    def _get_recommendation_reason(self, stock: Dict) -> str:
        """Generate recommendation reason"""
        try:
            reasons = []
            
            # Fundamental reasons
            fundamental_score = stock.get('fundamental_score', 0)
            if fundamental_score > 70:
                reasons.append("Strong fundamentals")
            elif fundamental_score > 50:
                reasons.append("Good fundamentals")
            
            # Technical reasons
            rsi = stock.get('rsi', 50)
            if rsi < 35:
                reasons.append("Oversold condition")
            elif rsi > 65:
                reasons.append("Strong momentum")
            
            price_vs_sma20 = stock.get('price_vs_sma20', 0)
            if price_vs_sma20 > 5:
                reasons.append("Above 20-day MA")
            elif price_vs_sma20 < -5:
                reasons.append("Below 20-day MA (potential value)")
            
            # Valuation reasons
            pe_ratio = stock.get('pe_ratio', 0)
            if pe_ratio < 15:
                reasons.append("Attractive valuation")
            
            roe = stock.get('roe', 0)
            if roe > 15:
                reasons.append("High ROE")
            
            return "; ".join(reasons) if reasons else "Meets screening criteria"
            
        except Exception as e:
            return "Analysis available"
    
    def get_sector_analysis(self) -> Dict:
        """Get sector-wise analysis and recommendations"""
        try:
            symbols = self.market_service.get_available_symbols()
            sector_data = {}
            
            for symbol in symbols:
                sector = self.market_service.sector_mapping.get(symbol, "Unknown")
                if sector not in sector_data:
                    sector_data[sector] = {
                        'stocks': [],
                        'avg_pe': 0,
                        'avg_roe': 0,
                        'performance': 0
                    }
                
                # Get stock data
                data = self.market_service.get_historical_data([symbol], period="1m")
                if not data.empty:
                    # Calculate 1-month performance
                    if len(data['Close']) > 1:
                        performance = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
                        sector_data[sector]['stocks'].append({
                            'symbol': symbol,
                            'performance': performance
                        })
            
            # Calculate sector averages
            for sector in sector_data:
                if sector_data[sector]['stocks']:
                    performances = [stock['performance'] for stock in sector_data[sector]['stocks']]
                    sector_data[sector]['avg_performance'] = np.mean(performances)
                    sector_data[sector]['stock_count'] = len(sector_data[sector]['stocks'])
            
            # Sort sectors by performance
            sorted_sectors = sorted(sector_data.items(), 
                                  key=lambda x: x[1].get('avg_performance', 0), 
                                  reverse=True)
            
            return {
                'sector_analysis': dict(sorted_sectors),
                'top_performing_sector': sorted_sectors[0][0] if sorted_sectors else None,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error in sector analysis: {e}")
            return {}
    
    def get_market_outlook(self) -> Dict:
        """Get overall market outlook and recommendations"""
        try:
            # Get market sentiment
            sentiment = self.market_service.get_market_sentiment()
            
            # Get top gainers/losers
            gainers_losers = self.market_service.get_top_gainers_losers()
            
            # Generate outlook
            nifty_change = sentiment.get('nifty_change_pct', 0)
            
            if nifty_change > 1:
                outlook = "Bullish"
                recommendation = "Consider increasing equity allocation"
            elif nifty_change > 0:
                outlook = "Cautiously Optimistic"
                recommendation = "Maintain current allocation, look for quality stocks"
            elif nifty_change > -1:
                outlook = "Neutral"
                recommendation = "Focus on stock selection, avoid momentum plays"
            else:
                outlook = "Bearish"
                recommendation = "Consider defensive stocks, reduce risk"
            
            return {
                'market_outlook': outlook,
                'recommendation': recommendation,
                'market_sentiment': sentiment,
                'top_movers': gainers_losers,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting market outlook: {e}")
            return {}