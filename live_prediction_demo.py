"""
Live Prediction Demo Script
Demonstrates the advanced live prediction capabilities
"""

import requests
import json
import time
from typing import Dict, List
from datetime import datetime

BASE_URL = "http://localhost:8000/api/v1/live"

class LivePredictionDemo:
    """Demo client for live prediction features"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
    
    def train_and_predict_demo(self):
        """Demonstrate model training and prediction"""
        print("🤖 LIVE PREDICTION ENGINE DEMO")
        print("=" * 50)
        
        symbols = ["AAPL", "GOOGL", "TSLA", "MSFT", "NVDA"]
        
        for symbol in symbols:
            print(f"\n🔧 Training models for {symbol}...")
            
            # Train models
            response = requests.post(f"{self.base_url}/train/{symbol}")
            if response.status_code == 200:
                training_result = response.json()
                print(f"✅ Training completed: {training_result['models_trained']}")
                print(f"   Data points: {training_result['data_points']}")
                print(f"   Features: {training_result['features']}")
            else:
                print(f"❌ Training failed: {response.json()}")
                continue
            
            # Make prediction
            print(f"🔮 Making live prediction for {symbol}...")
            response = requests.get(f"{self.base_url}/predict/{symbol}")
            if response.status_code == 200:
                prediction = response.json()
                self._display_prediction(prediction)
            else:
                print(f"❌ Prediction failed: {response.json()}")
    
    def _display_prediction(self, prediction: Dict):
        """Display prediction results in a formatted way"""
        symbol = prediction['symbol']
        current_price = prediction['current_price']
        
        print(f"\n📊 PREDICTION RESULTS for {symbol}")
        print(f"   Current Price: ${current_price:.2f}")
        print(f"   Current Return: {prediction['current_return']:+.2f}%")
        
        if 'predictions' in prediction:
            preds = prediction['predictions']
            
            if 'target_1d' in preds:
                pred_1d = preds['target_1d']
                print(f"   1-Day Prediction: {pred_1d['predicted_return_pct']:+.2f}% (Confidence: {pred_1d['confidence']:.1f}%)")
                print(f"   Predicted Price (1d): ${prediction['predicted_prices'].get('1_day_price', 0):.2f}")
            
            if 'target_5d' in preds:
                pred_5d = preds['target_5d']
                print(f"   5-Day Prediction: {pred_5d['predicted_return_pct']:+.2f}% (Confidence: {pred_5d['confidence']:.1f}%)")
                print(f"   Predicted Price (5d): ${prediction['predicted_prices'].get('5_day_price', 0):.2f}")
            
            if 'target_20d' in preds:
                pred_20d = preds['target_20d']
                print(f"   20-Day Prediction: {pred_20d['predicted_return_pct']:+.2f}% (Confidence: {pred_20d['confidence']:.1f}%)")
                print(f"   Predicted Price (20d): ${prediction['predicted_prices'].get('20_day_price', 0):.2f}")
    
    def live_quotes_demo(self):
        """Demonstrate live quotes functionality"""
        print("\n📈 LIVE QUOTES DEMO")
        print("=" * 30)
        
        symbols = ["AAPL", "GOOGL", "TSLA", "SPY", "QQQ"]
        
        # Get multiple quotes
        response = requests.get(f"{self.base_url}/quotes", params={"symbols": symbols})
        if response.status_code == 200:
            quotes_data = response.json()
            quotes = quotes_data['quotes']
            
            print(f"📊 Live Quotes ({quotes_data['timestamp']})")
            print("-" * 60)
            print(f"{'Symbol':<8} {'Price':<10} {'Change':<10} {'Change%':<10} {'Volume':<12}")
            print("-" * 60)
            
            for symbol, quote in quotes.items():
                if 'error' not in quote:
                    print(f"{symbol:<8} ${quote['price']:<9.2f} {quote['change']:+<9.2f} {quote['change_percent']:+<9.2f}% {quote['volume']:<12,.0f}")
                else:
                    print(f"{symbol:<8} Error: {quote['error']}")
        else:
            print(f"❌ Failed to get quotes: {response.json()}")
    
    def market_sentiment_demo(self):
        """Demonstrate market sentiment analysis"""
        print("\n🌊 MARKET SENTIMENT ANALYSIS")
        print("=" * 35)
        
        symbols = ["SPY", "QQQ", "IWM", "VTI", "DIA"]
        
        response = requests.get(f"{self.base_url}/sentiment/market", params={"symbols": symbols})
        if response.status_code == 200:
            sentiment_data = response.json()
            
            overall = sentiment_data['overall_sentiment']
            individual = sentiment_data['individual_sentiment']
            
            print(f"📊 Overall Market Sentiment: {overall['overall_sentiment'].upper()}")
            print(f"   Short-term Bullish: {overall['short_term_bullish_pct']:.1f}%")
            print(f"   Medium-term Bullish: {overall['medium_term_bullish_pct']:.1f}%")
            print(f"   Symbols Analyzed: {overall['symbols_analyzed']}")
            
            print("\n📈 Individual Symbol Sentiment:")
            print("-" * 50)
            for symbol, data in individual.items():
                print(f"   {symbol}: {data['short_term_sentiment'].title()} (1d: {data['predicted_1d_return']:+.2f}%)")
        else:
            print(f"❌ Failed to get sentiment: {response.json()}")
    
    def comprehensive_analysis_demo(self):
        """Demonstrate comprehensive analysis"""
        print("\n🎯 COMPREHENSIVE ANALYSIS DEMO")
        print("=" * 40)
        
        symbol = "AAPL"
        
        print(f"🔍 Analyzing {symbol} comprehensively...")
        response = requests.get(f"{self.base_url}/analysis/comprehensive/{symbol}")
        
        if response.status_code == 200:
            analysis = response.json()
            
            print(f"\n📊 COMPREHENSIVE ANALYSIS for {symbol}")
            print("=" * 45)
            
            # Live quote info
            if 'live_quote' in analysis and 'error' not in analysis['live_quote']:
                quote = analysis['live_quote']
                print(f"💰 Current Price: ${quote['price']:.2f} ({quote['change_percent']:+.2f}%)")
                print(f"📊 Volume Ratio: {quote.get('volume_ratio', 1):.2f}x")
                print(f"📈 52W High: ${quote.get('52_week_high', 0):.2f}")
                print(f"📉 52W Low: ${quote.get('52_week_low', 0):.2f}")
            
            # Predictions
            if 'predictions' in analysis and 'error' not in analysis['predictions']:
                preds = analysis['predictions'].get('predictions', {})
                if 'target_1d' in preds:
                    print(f"🔮 1-Day Prediction: {preds['target_1d']['predicted_return_pct']:+.2f}%")
                if 'target_5d' in preds:
                    print(f"🔮 5-Day Prediction: {preds['target_5d']['predicted_return_pct']:+.2f}%")
            
            # Technical indicators
            if 'intraday_data' in analysis and 'error' not in analysis['intraday_data']:
                intraday = analysis['intraday_data']
                summary = intraday.get('summary', {})
                print(f"📊 RSI: {summary.get('current_rsi', 50):.1f}")
                print(f"📊 Day High: ${summary.get('high_of_day', 0):.2f}")
                print(f"📊 Day Low: ${summary.get('low_of_day', 0):.2f}")
            
            # Options sentiment
            if 'options_data' in analysis and 'error' not in analysis['options_data']:
                options = analysis['options_data']
                print(f"📊 Options Sentiment: {options.get('sentiment', 'neutral').title()}")
                print(f"📊 Put/Call Ratio: {options.get('put_call_ratio', 0):.2f}")
            
            # Final recommendation
            summary = analysis.get('summary', {})
            recommendation = summary.get('recommendation', 'hold').upper()
            print(f"\n🎯 RECOMMENDATION: {recommendation}")
            
        else:
            print(f"❌ Analysis failed: {response.json()}")
    
    def watchlist_analysis_demo(self):
        """Demonstrate watchlist analysis"""
        print("\n👀 WATCHLIST ANALYSIS DEMO")
        print("=" * 35)
        
        watchlist = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META"]
        
        print(f"📋 Analyzing watchlist: {', '.join(watchlist)}")
        
        response = requests.post(f"{self.base_url}/watchlist/analyze", params={"symbols": watchlist})
        
        if response.status_code == 200:
            analysis = response.json()
            
            print(f"\n📊 WATCHLIST ANALYSIS RESULTS")
            print("=" * 45)
            print(f"Symbols Analyzed: {analysis['symbols_analyzed']}")
            print(f"Top Picks: {', '.join(analysis['top_picks'])}")
            
            print("\n📈 Detailed Analysis:")
            print("-" * 80)
            print(f"{'Symbol':<8} {'Price':<10} {'Change%':<10} {'Pred 1D%':<10} {'Pred 5D%':<10} {'Rec':<12}")
            print("-" * 80)
            
            for symbol, data in analysis['watchlist_analysis'].items():
                print(f"{symbol:<8} ${data['current_price']:<9.2f} {data['change_percent']:+<9.2f}% "
                      f"{data['predicted_1d_return']:+<9.2f}% {data['predicted_5d_return']:+<9.2f}% "
                      f"{data['recommendation']:<12}")
        else:
            print(f"❌ Watchlist analysis failed: {response.json()}")
    
    def trading_alerts_demo(self):
        """Demonstrate trading alerts"""
        print("\n🚨 TRADING ALERTS DEMO")
        print("=" * 30)
        
        symbols = ["AAPL", "TSLA", "NVDA"]
        
        for symbol in symbols:
            print(f"\n🔍 Generating alerts for {symbol}...")
            
            response = requests.get(f"{self.base_url}/alerts/generate/{symbol}")
            
            if response.status_code == 200:
                alerts_data = response.json()
                alerts = alerts_data['alerts']
                
                if alerts:
                    print(f"🚨 {len(alerts)} alerts generated for {symbol}:")
                    for alert in alerts:
                        severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(alert['severity'], "⚪")
                        print(f"   {severity_emoji} {alert['type'].upper()}: {alert['message']}")
                        print(f"      Action: {alert['action']}")
                else:
                    print(f"✅ No alerts for {symbol}")
            else:
                print(f"❌ Failed to generate alerts: {response.json()}")
    
    def market_overview_demo(self):
        """Demonstrate market overview"""
        print("\n🌍 MARKET OVERVIEW DEMO")
        print("=" * 30)
        
        response = requests.get(f"{self.base_url}/market/overview")
        
        if response.status_code == 200:
            overview = response.json()
            
            if 'error' not in overview:
                print("📊 Major Indices:")
                print("-" * 40)
                
                indices = overview.get('indices', {})
                for name, data in indices.items():
                    print(f"{name:<15}: ${data['price']:<10.2f} ({data['change_percent']:+.2f}%)")
                
                print(f"\n🌊 Market Sentiment: {overview.get('market_sentiment', 'Unknown')}")
                print(f"📊 VIX Level: {overview.get('vix_level', 0):.2f}")
            else:
                print(f"❌ Market overview error: {overview['error']}")
        else:
            print(f"❌ Failed to get market overview: {response.json()}")
    
    def run_full_demo(self):
        """Run the complete live prediction demo"""
        print("🚀" * 20)
        print("🏛️  PERSONAL ALADDIN - LIVE PREDICTION PLATFORM")
        print("🚀" * 20)
        
        try:
            # Market overview first
            self.market_overview_demo()
            
            # Live quotes
            self.live_quotes_demo()
            
            # Market sentiment
            self.market_sentiment_demo()
            
            # Train models and make predictions (limited to save time)
            print("\n🤖 TRAINING MODELS (Sample)")
            print("=" * 30)
            symbol = "AAPL"
            print(f"🔧 Training model for {symbol}...")
            response = requests.post(f"{self.base_url}/train/{symbol}")
            if response.status_code == 200:
                print("✅ Training completed")
                
                # Make prediction
                response = requests.get(f"{self.base_url}/predict/{symbol}")
                if response.status_code == 200:
                    self._display_prediction(response.json())
            
            # Comprehensive analysis
            self.comprehensive_analysis_demo()
            
            # Watchlist analysis
            self.watchlist_analysis_demo()
            
            # Trading alerts
            self.trading_alerts_demo()
            
            print("\n" + "🎉" * 20)
            print("✅ LIVE PREDICTION DEMO COMPLETED!")
            print("🎯 Key Features Demonstrated:")
            print("   • 🤖 Machine Learning Price Predictions")
            print("   • 📈 Real-time Market Data & Quotes")
            print("   • 🌊 Market Sentiment Analysis")
            print("   • 🎯 Comprehensive Stock Analysis")
            print("   • 👀 Watchlist Analysis & Recommendations")
            print("   • 🚨 Automated Trading Alerts")
            print("   • 🌍 Market Overview & Indices")
            print("🎉" * 20)
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            print("Make sure the server is running: python main.py")


if __name__ == "__main__":
    demo = LivePredictionDemo()
    demo.run_full_demo()