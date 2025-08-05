"""
Personal Aladdin - Simple Demo
Demonstrates core investment management features
"""

import requests
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

class PersonalAladdinDemo:
    """Simple demo for Personal Aladdin platform"""
    
    def __init__(self):
        self.base_url = BASE_URL
    
    def test_server(self):
        """Test if server is running"""
        try:
            response = requests.get(f"{self.base_url}/")
            if response.status_code == 200:
                print("✅ Server is running!")
                return True
            else:
                print("❌ Server responded with error")
                return False
        except:
            print("❌ Server is not running. Please start it with: python simple_main.py")
            return False
    
    def portfolio_demo(self):
        """Demonstrate portfolio management"""
        print("\n💼 PORTFOLIO MANAGEMENT DEMO")
        print("=" * 40)
        
        # Create portfolio
        print("📁 Creating investment portfolio...")
        response = requests.post(f"{self.base_url}/portfolio/create?name=Growth Portfolio&initial_cash=100000")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Portfolio created: {result['portfolio']['name']}")
            print(f"💰 Initial cash: ${result['portfolio']['cash']:,.2f}")
        else:
            print(f"❌ Portfolio creation failed: {response.json()}")
            return
        
        # Add positions
        positions = [
            {"symbol": "AAPL", "quantity": 50, "price": 150},
            {"symbol": "GOOGL", "quantity": 20, "price": 140},
            {"symbol": "MSFT", "quantity": 30, "price": 300},
            {"symbol": "TSLA", "quantity": 25, "price": 200}
        ]
        
        print("\n📈 Adding stock positions...")
        for pos in positions:
            response = requests.post(
                f"{self.base_url}/portfolio/Growth%20Portfolio/add?symbol={pos['symbol']}&quantity={pos['quantity']}&price={pos['price']}"
            )
            if response.status_code == 200:
                print(f"✅ Added {pos['quantity']} shares of {pos['symbol']}")
            else:
                print(f"❌ Failed to add {pos['symbol']}: {response.json()}")
        
        # Get portfolio status
        print("\n📊 Portfolio Summary:")
        response = requests.get(f"{self.base_url}/portfolio/Growth%20Portfolio")
        if response.status_code == 200:
            portfolio = response.json()["portfolio"]
            print(f"💰 Total Value: ${portfolio['total_value']:,.2f}")
            print(f"💵 Cash: ${portfolio['cash']:,.2f}")
            print(f"📈 Positions: {len(portfolio['positions'])}")
            
            print("\n🏢 Holdings:")
            for symbol, pos in portfolio['positions'].items():
                print(f"   {symbol}: {pos['quantity']} shares @ ${pos['avg_price']:.2f} = ${pos['market_value']:,.2f}")
    
    def live_quotes_demo(self):
        """Demonstrate live market quotes"""
        print("\n📊 LIVE MARKET QUOTES")
        print("=" * 30)
        
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
        
        print("🔄 Fetching live quotes...")
        print(f"{'Symbol':<8} {'Price':<10} {'Change':<10} {'Change%':<10} {'Volume':<12}")
        print("-" * 60)
        
        for symbol in symbols:
            try:
                response = requests.get(f"{self.base_url}/market/quote/{symbol}")
                if response.status_code == 200:
                    quote = response.json()
                    print(f"{symbol:<8} ${quote['price']:<9.2f} {quote['change']:+<9.2f} {quote['change_percent']:+<9.2f}% {quote['volume']:<12,}")
                else:
                    print(f"{symbol:<8} Error fetching data")
            except Exception as e:
                print(f"{symbol:<8} Connection error")
    
    def ai_predictions_demo(self):
        """Demonstrate AI-powered predictions"""
        print("\n🤖 AI-POWERED STOCK PREDICTIONS")
        print("=" * 40)
        
        symbols = ["AAPL", "GOOGL", "TSLA"]
        
        for symbol in symbols:
            print(f"\n🔮 Analyzing {symbol}...")
            try:
                response = requests.get(f"{self.base_url}/ai/predict/{symbol}")
                if response.status_code == 200:
                    pred = response.json()
                    
                    print(f"📊 Current Price: ${pred['current_price']:.2f}")
                    print(f"🎯 Predicted Price: ${pred['predicted_price']:.2f}")
                    print(f"📈 Expected Return: {pred['predicted_return']:+.2f}%")
                    print(f"🎪 Direction: {pred['direction'].upper()}")
                    print(f"🎯 Confidence: {pred['confidence']:.0%}")
                    
                    # Technical indicators
                    tech = pred['technical_indicators']
                    print(f"📊 Technical Indicators:")
                    print(f"   SMA 20: ${tech['sma_20']:.2f}")
                    print(f"   SMA 50: ${tech['sma_50']:.2f}")
                    print(f"   RSI: {tech['rsi']:.1f}")
                    print(f"   Volatility: {tech['volatility']:.1%}")
                    
                else:
                    print(f"❌ Prediction failed for {symbol}")
            except Exception as e:
                print(f"❌ Error predicting {symbol}: {e}")
    
    def comprehensive_analysis_demo(self):
        """Demonstrate comprehensive stock analysis"""
        print("\n🎯 COMPREHENSIVE STOCK ANALYSIS")
        print("=" * 40)
        
        symbol = "AAPL"
        print(f"🔍 Analyzing {symbol} comprehensively...")
        
        try:
            response = requests.get(f"{self.base_url}/analysis/{symbol}")
            if response.status_code == 200:
                analysis = response.json()
                
                print(f"\n🏢 Company: {analysis['company_name']}")
                print(f"🏭 Sector: {analysis['sector']}")
                
                # Current data
                current = analysis['current_data']
                print(f"\n💰 Current Price: ${current['price']:.2f} ({current['change_percent']:+.2f}%)")
                print(f"📊 Volume: {current['volume']:,}")
                print(f"📈 Day Range: ${current['low']:.2f} - ${current['high']:.2f}")
                
                # Prediction
                pred = analysis['prediction']
                print(f"\n🤖 AI Prediction:")
                print(f"   Direction: {pred['direction'].upper()}")
                print(f"   Expected Return: {pred['predicted_return']:+.2f}%")
                print(f"   Confidence: {pred['confidence']:.0%}")
                
                # Risk metrics
                risk = analysis['risk_metrics']
                print(f"\n⚠️ Risk Metrics:")
                print(f"   Volatility: {risk['volatility']:.1%}")
                print(f"   VaR (95%): {risk['var_95']:.2f}%")
                print(f"   Max Drawdown: {risk['max_drawdown']:.2f}%")
                print(f"   Beta: {risk['beta']:.2f}")
                
                # Recommendation
                rec = analysis['recommendation']
                print(f"\n🎯 RECOMMENDATION: {rec['action']}")
                print(f"   Risk Level: {rec['risk_level']}")
                print(f"   Reasoning: {rec['reasoning']}")
                print(f"   Time Horizon: {rec['time_horizon']}")
                
            else:
                print(f"❌ Analysis failed for {symbol}")
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
    
    def portfolio_optimization_demo(self):
        """Demonstrate portfolio optimization"""
        print("\n🎯 PORTFOLIO OPTIMIZATION")
        print("=" * 35)
        
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        methods = ["equal_weight", "risk_parity", "mean_reversion"]
        
        for method in methods:
            print(f"\n📊 {method.replace('_', ' ').title()} Optimization:")
            try:
                response = requests.post(f"{self.base_url}/portfolio/optimize",
                                       json=symbols,
                                       params={"method": method})
                
                if response.status_code == 200:
                    result = response.json()
                    
                    print(f"   Expected Return: {result['expected_annual_return']:.1%}")
                    print(f"   Expected Volatility: {result['expected_volatility']:.1%}")
                    print(f"   Sharpe Ratio: {result['sharpe_ratio']:.2f}")
                    
                    print(f"   Optimal Weights:")
                    for symbol, weight in result['optimal_weights'].items():
                        print(f"     {symbol}: {weight:.1%}")
                else:
                    print(f"   ❌ Optimization failed")
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    def market_movers_demo(self):
        """Demonstrate market movers"""
        print("\n🚀 MARKET MOVERS")
        print("=" * 20)
        
        try:
            response = requests.get(f"{self.base_url}/market/movers")
            if response.status_code == 200:
                movers = response.json()
                
                print("📈 Biggest Gainers:")
                for stock in movers['biggest_gainers']:
                    print(f"   {stock['symbol']}: ${stock['price']:.2f} ({stock['change_percent']:+.2f}%)")
                
                print("\n📉 Biggest Losers:")
                for stock in movers['biggest_losers']:
                    print(f"   {stock['symbol']}: ${stock['price']:.2f} ({stock['change_percent']:+.2f}%)")
                
                print("\n🔥 Top Movers (by volatility):")
                for stock in movers['top_movers']:
                    print(f"   {stock['symbol']}: ${stock['price']:.2f} ({stock['change_percent']:+.2f}%)")
            else:
                print("❌ Failed to get market movers")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def run_full_demo(self):
        """Run complete demonstration"""
        print("🏛️" * 20)
        print("🏛️  PERSONAL ALADDIN - INVESTMENT MANAGEMENT PLATFORM")
        print("🏛️" * 20)
        print("\n🎯 Institutional-grade investment management for personal investors")
        
        # Test server connection
        if not self.test_server():
            return
        
        try:
            # Run all demos
            self.portfolio_demo()
            time.sleep(1)
            
            self.live_quotes_demo()
            time.sleep(1)
            
            self.ai_predictions_demo()
            time.sleep(1)
            
            self.comprehensive_analysis_demo()
            time.sleep(1)
            
            self.portfolio_optimization_demo()
            time.sleep(1)
            
            self.market_movers_demo()
            
            print("\n" + "🎉" * 20)
            print("✅ PERSONAL ALADDIN DEMO COMPLETED!")
            print("🎯 Features Demonstrated:")
            print("   • 💼 Portfolio Management & Tracking")
            print("   • 📊 Live Market Data & Quotes")
            print("   • 🤖 AI-Powered Price Predictions")
            print("   • 🎯 Comprehensive Stock Analysis")
            print("   • 📈 Portfolio Optimization Strategies")
            print("   • 🚀 Market Movers & Trends")
            print("\n🏆 INSTITUTIONAL-GRADE INVESTMENT MANAGEMENT")
            print("🎉" * 20)
            
        except KeyboardInterrupt:
            print("\n🛑 Demo stopped by user")
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")


if __name__ == "__main__":
    demo = PersonalAladdinDemo()
    demo.run_full_demo()