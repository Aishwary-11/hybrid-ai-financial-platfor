"""
Live Streaming Demo
Real-time market data streaming demonstration
"""

import asyncio
import requests
import time
from datetime import datetime
from typing import List

BASE_URL = "http://localhost:8000/api/v1/live"

class LiveStreamDemo:
    """Live streaming demonstration"""
    
    def __init__(self):
        self.base_url = BASE_URL
        self.running = True
    
    async def stream_quotes(self, symbols: List[str], interval: int = 10):
        """Stream live quotes for symbols"""
        print(f"🔴 LIVE STREAMING: {', '.join(symbols)}")
        print(f"📊 Update interval: {interval} seconds")
        print("=" * 60)
        print(f"{'Time':<10} {'Symbol':<8} {'Price':<10} {'Change':<10} {'Change%':<10}")
        print("=" * 60)
        
        while self.running:
            try:
                # Get live quotes
                response = requests.get(f"{self.base_url}/quotes", 
                                      params={"symbols": symbols})
                
                if response.status_code == 200:
                    quotes_data = response.json()
                    quotes = quotes_data['quotes']
                    current_time = datetime.now().strftime("%H:%M:%S")
                    
                    for symbol, quote in quotes.items():
                        if 'error' not in quote:
                            print(f"{current_time:<10} {symbol:<8} ${quote['price']:<9.2f} "
                                  f"{quote['change']:+<9.2f} {quote['change_percent']:+<9.2f}%")
                        else:
                            print(f"{current_time:<10} {symbol:<8} ERROR: {quote['error']}")
                else:
                    print(f"❌ API Error: {response.status_code}")
                
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                print("\n🛑 Streaming stopped by user")
                self.running = False
                break
            except Exception as e:
                print(f"❌ Stream error: {e}")
                await asyncio.sleep(interval)
    
    async def stream_predictions(self, symbols: List[str], interval: int = 30):
        """Stream live predictions for symbols"""
        print(f"🔮 LIVE PREDICTION STREAMING: {', '.join(symbols)}")
        print(f"📊 Update interval: {interval} seconds")
        print("=" * 80)
        print(f"{'Time':<10} {'Symbol':<8} {'Price':<10} {'Pred 1D%':<10} {'Pred 5D%':<10} {'Confidence':<12}")
        print("=" * 80)
        
        while self.running:
            try:
                current_time = datetime.now().strftime("%H:%M:%S")
                
                for symbol in symbols:
                    # Get prediction
                    response = requests.get(f"{self.base_url}/predict/{symbol}")
                    
                    if response.status_code == 200:
                        prediction = response.json()
                        
                        if 'error' not in prediction:
                            price = prediction.get('current_price', 0)
                            preds = prediction.get('predictions', {})
                            
                            pred_1d = preds.get('target_1d', {}).get('predicted_return_pct', 0)
                            pred_5d = preds.get('target_5d', {}).get('predicted_return_pct', 0)
                            conf_1d = preds.get('target_1d', {}).get('confidence', 0)
                            
                            print(f"{current_time:<10} {symbol:<8} ${price:<9.2f} "
                                  f"{pred_1d:+<9.2f}% {pred_5d:+<9.2f}% {conf_1d:<11.1f}%")
                        else:
                            print(f"{current_time:<10} {symbol:<8} ERROR: {prediction['error']}")
                    else:
                        print(f"{current_time:<10} {symbol:<8} API ERROR: {response.status_code}")
                
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                print("\n🛑 Prediction streaming stopped by user")
                self.running = False
                break
            except Exception as e:
                print(f"❌ Prediction stream error: {e}")
                await asyncio.sleep(interval)
    
    async def stream_market_sentiment(self, interval: int = 60):
        """Stream market sentiment updates"""
        print("🌊 LIVE MARKET SENTIMENT STREAMING")
        print(f"📊 Update interval: {interval} seconds")
        print("=" * 50)
        
        while self.running:
            try:
                current_time = datetime.now().strftime("%H:%M:%S")
                
                # Get market sentiment
                response = requests.get(f"{self.base_url}/sentiment/market")
                
                if response.status_code == 200:
                    sentiment_data = response.json()
                    overall = sentiment_data.get('overall_sentiment', {})
                    
                    sentiment = overall.get('overall_sentiment', 'unknown').upper()
                    bullish_pct = overall.get('short_term_bullish_pct', 0)
                    
                    print(f"{current_time} | Market Sentiment: {sentiment} | Bullish: {bullish_pct:.1f}%")
                else:
                    print(f"{current_time} | API ERROR: {response.status_code}")
                
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                print("\n🛑 Sentiment streaming stopped by user")
                self.running = False
                break
            except Exception as e:
                print(f"❌ Sentiment stream error: {e}")
                await asyncio.sleep(interval)
    
    def run_quote_stream(self):
        """Run live quote streaming"""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
        asyncio.run(self.stream_quotes(symbols, interval=5))
    
    def run_prediction_stream(self):
        """Run live prediction streaming"""
        symbols = ["AAPL", "TSLA", "NVDA"]
        
        # First train models
        print("🤖 Training models for streaming...")
        for symbol in symbols:
            print(f"Training {symbol}...")
            response = requests.post(f"{self.base_url}/train/{symbol}")
            if response.status_code == 200:
                print(f"✅ {symbol} trained")
            else:
                print(f"❌ {symbol} training failed")
        
        print("\nStarting prediction stream...\n")
        asyncio.run(self.stream_predictions(symbols, interval=15))
    
    def run_sentiment_stream(self):
        """Run market sentiment streaming"""
        asyncio.run(self.stream_market_sentiment(interval=30))


def main():
    """Main demo function"""
    demo = LiveStreamDemo()
    
    print("🚀 PERSONAL ALADDIN - LIVE STREAMING DEMO")
    print("=" * 45)
    print("Choose streaming mode:")
    print("1. Live Quotes (5s updates)")
    print("2. Live Predictions (15s updates)")
    print("3. Market Sentiment (30s updates)")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == "1":
                print("\n🔴 Starting live quote stream...")
                print("Press Ctrl+C to stop\n")
                demo.run_quote_stream()
                break
            elif choice == "2":
                print("\n🔮 Starting live prediction stream...")
                print("Press Ctrl+C to stop\n")
                demo.run_prediction_stream()
                break
            elif choice == "3":
                print("\n🌊 Starting market sentiment stream...")
                print("Press Ctrl+C to stop\n")
                demo.run_sentiment_stream()
                break
            elif choice == "4":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-4.")
        except KeyboardInterrupt:
            print("\n👋 Demo stopped by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()