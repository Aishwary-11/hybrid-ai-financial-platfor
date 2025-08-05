"""
Personal Aladdin - Indian Market Demo
Demonstrates AI-powered stock recommendations and live market analysis
"""

import requests
import json
from typing import Dict, List
import time

BASE_URL = "http://localhost:8000/api/v1/indian-market"

class IndianMarketDemo:
    """Demo client for Indian market features"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
    
    def show_available_stocks(self):
        """Show available Indian stocks"""
        print("📋 Available Indian Stocks:")
        
        response = requests.get(f"{self.base_url}/available-stocks")
        if response.status_code == 200:
            data = response.json()
            stocks = data["available_stocks"]
            
            # Group by sector
            sectors = {}
            for stock in stocks:
                sector = stock["sector"]
                if sector not in sectors:
                    sectors[sector] = []
                sectors[sector].append(stock["symbol"])
            
            for sector, symbols in sectors.items():
                print(f"\n🏢 {sector}:")
                print(f"   {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
            
            print(f"\n✅ Total: {data['total_stocks']} stocks available")
        else:
            print("❌ Failed to fetch available stocks")
    
    def get_live_market_data(self):
        """Get live market data for popular stocks"""
        print("\n📊 Live Market Data:")
        
        popular_stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC"]
        
        for symbol in popular_stocks:
            try:
                response = requests.get(f"{self.base_url}/live-data/{symbol}")
                if response.status_code == 200:
                    data = response.json()
                    print(f"📈 {symbol}: ₹{data['current_price']:.2f} "
                          f"(Vol: {data['volume']:,})")
                else:
                    print(f"❌ {symbol}: Data not available")
            except Exception as e:
                print(f"❌ {symbol}: Error fetching data")
    
    def get_market_sentiment(self):
        """Get current market sentiment"""
        print("\n🌊 Market Sentiment Analysis:")
        
        response = requests.get(f"{self.base_url}/market-sentiment")
        if response.status_code == 200:
            data = response.json()
            print(f"📊 Nifty 50: {data['nifty_current']:.2f}")
            print(f"📈 Change: {data['nifty_change']:+.2f} ({data['nifty_change_pct']:+.2f}%)")
            print(f"🎯 Sentiment: {data['market_sentiment']}")
        else:
            print("❌ Failed to fetch market sentiment")
    
    def get_top_movers(self):
        """Get top gainers and losers"""
        print("\n🚀 Top Market Movers:")
        
        response = requests.get(f"{self.base_url}/top-movers?limit=5")
        if response.status_code == 200:
            data = response.json()
            
            print("\n📈 Top Gainers:")
            for stock in data.get("top_gainers", []):
                print(f"   {stock['symbol']}: +{stock['change_pct']:.2f}% "
                      f"({stock['sector']})")
            
            print("\n📉 Top Losers:")
            for stock in data.get("top_losers", []):
                print(f"   {stock['symbol']}: {stock['change_pct']:.2f}% "
                      f"({stock['sector']})")
        else:
            print("❌ Failed to fetch top movers")
    
    def get_technical_analysis(self, symbol: str = "RELIANCE"):
        """Get technical analysis for a stock"""
        print(f"\n🔍 Technical Analysis - {symbol}:")
        
        response = requests.get(f"{self.base_url}/technical-analysis/{symbol}")
        if response.status_code == 200:
            data = response.json()
            indicators = data["technical_indicators"]
            signals = data["signals"]
            
            print(f"💰 Current Price: ₹{indicators.get('Current_Price', 0):.2f}")
            print(f"📊 RSI: {indicators.get('RSI', 0):.1f}")
            print(f"📈 Price vs SMA20: {indicators.get('Price_vs_SMA20', 0):+.2f}%")
            print(f"📉 Price vs SMA50: {indicators.get('Price_vs_SMA50', 0):+.2f}%")
            
            print("\n🎯 Trading Signals:")
            for indicator, signal in signals.items():
                print(f"   {indicator}: {signal}")
        else:
            print(f"❌ Failed to get technical analysis for {symbol}")
    
    def get_stock_recommendations(self, investment_amount: float = 100000, 
                                risk_tolerance: str = "moderate"):
        """Get AI-powered stock recommendations"""
        print(f"\n🤖 AI Stock Recommendations:")
        print(f"💰 Investment Amount: ₹{investment_amount:,.0f}")
        print(f"⚖️ Risk Tolerance: {risk_tolerance.title()}")
        
        response = requests.get(
            f"{self.base_url}/recommendations",
            params={
                "investment_amount": investment_amount,
                "risk_tolerance": risk_tolerance
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            recommendations = data.get("recommendations", [])
            
            if not recommendations:
                print("❌ No recommendations available")
                return
            
            print(f"\n📋 Top {len(recommendations)} Recommendations:")
            print("-" * 80)
            
            for i, stock in enumerate(recommendations, 1):
                print(f"{i}. {stock['symbol']} ({stock['sector']})")
                print(f"   💰 Current Price: ₹{stock['current_price']:.2f}")
                print(f"   📊 Recommendation Score: {stock['recommendation_score']:.1f}/100")
                print(f"   💵 Suggested Allocation: {stock['suggested_allocation_pct']:.1f}% "
                      f"(₹{stock['suggested_amount']:,.0f})")
                print(f"   📈 PE Ratio: {stock['pe_ratio']:.1f}")
                print(f"   📊 ROE: {stock['roe']:.1f}%")
                print(f"   💡 Reason: {stock['recommendation_reason']}")
                print("-" * 80)
        else:
            print("❌ Failed to get recommendations")
    
    def screen_stocks(self):
        """Demonstrate stock screening"""
        print("\n🔍 Stock Screening Demo:")
        
        # Screen for value stocks
        print("\n📊 Screening for Value Stocks (PE < 20, PB < 2, ROE > 12):")
        
        response = requests.get(
            f"{self.base_url}/screen-stocks",
            params={
                "max_pe": 20,
                "max_pb": 2,
                "min_roe": 12
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            stocks = data.get("screened_stocks", [])
            
            print(f"✅ Found {len(stocks)} stocks matching criteria:")
            for stock in stocks[:5]:  # Show top 5
                print(f"   {stock['symbol']}: PE={stock['pe_ratio']:.1f}, "
                      f"PB={stock['pb_ratio']:.1f}, ROE={stock['roe']:.1f}%")
        else:
            print("❌ Failed to screen stocks")
    
    def get_sector_analysis(self):
        """Get sector-wise analysis"""
        print("\n🏢 Sector Analysis:")
        
        response = requests.get(f"{self.base_url}/sector-analysis")
        if response.status_code == 200:
            data = response.json()
            sector_analysis = data.get("sector_analysis", {})
            top_sector = data.get("top_performing_sector")
            
            print(f"🏆 Top Performing Sector: {top_sector}")
            print("\n📊 Sector Performance:")
            
            for sector, info in list(sector_analysis.items())[:5]:
                avg_perf = info.get('avg_performance', 0)
                stock_count = info.get('stock_count', 0)
                print(f"   {sector}: {avg_perf:+.2f}% ({stock_count} stocks)")
        else:
            print("❌ Failed to get sector analysis")
    
    def get_market_outlook(self):
        """Get market outlook"""
        print("\n🔮 Market Outlook:")
        
        response = requests.get(f"{self.base_url}/market-outlook")
        if response.status_code == 200:
            data = response.json()
            print(f"📈 Outlook: {data.get('market_outlook', 'N/A')}")
            print(f"💡 Recommendation: {data.get('recommendation', 'N/A')}")
        else:
            print("❌ Failed to get market outlook")
    
    def compare_stocks(self):
        """Compare popular stocks"""
        print("\n⚖️ Stock Comparison - IT Giants:")
        
        symbols = ["TCS", "INFY", "HCLTECH", "WIPRO"]
        
        response = requests.get(
            f"{self.base_url}/compare-stocks",
            params={"symbols": symbols}
        )
        
        if response.status_code == 200:
            data = response.json()
            comparison = data.get("stock_comparison", [])
            
            print(f"{'Stock':<10} {'Price':<10} {'PE':<8} {'ROE':<8} {'RSI':<8} {'Score':<8}")
            print("-" * 60)
            
            for stock in comparison:
                print(f"{stock['symbol']:<10} "
                      f"₹{stock['current_price']:<9.2f} "
                      f"{stock['pe_ratio']:<7.1f} "
                      f"{stock['roe']:<7.1f}% "
                      f"{stock['rsi']:<7.1f} "
                      f"{stock['fundamental_score']:<7.1f}")
        else:
            print("❌ Failed to compare stocks")
    
    def run_full_demo(self):
        """Run complete Indian market demo"""
        print("=" * 80)
        print("🇮🇳 PERSONAL ALADDIN - INDIAN MARKET EDITION")
        print("🤖 AI-Powered Stock Recommendations & Live Market Analysis")
        print("=" * 80)
        
        try:
            # Show available stocks
            self.show_available_stocks()
            
            # Live market data
            self.get_live_market_data()
            
            # Market sentiment
            self.get_market_sentiment()
            
            # Top movers
            self.get_top_movers()
            
            # Technical analysis
            self.get_technical_analysis("RELIANCE")
            
            # Stock recommendations
            print("\n" + "="*50)
            print("🎯 PERSONALIZED INVESTMENT RECOMMENDATIONS")
            print("="*50)
            
            # Conservative recommendations
            self.get_stock_recommendations(100000, "conservative")
            
            # Moderate recommendations  
            self.get_stock_recommendations(200000, "moderate")
            
            # Stock screening
            self.screen_stocks()
            
            # Sector analysis
            self.get_sector_analysis()
            
            # Market outlook
            self.get_market_outlook()
            
            # Stock comparison
            self.compare_stocks()
            
            print("\n" + "=" * 80)
            print("✅ Indian Market Demo Completed Successfully!")
            print("\n🎯 Key Features Demonstrated:")
            print("   🇮🇳 Live Indian stock market data")
            print("   🤖 AI-powered personalized recommendations")
            print("   📊 Technical analysis with trading signals")
            print("   🔍 Advanced stock screening capabilities")
            print("   📈 Sector analysis and market sentiment")
            print("   ⚖️ Multi-stock comparison tools")
            print("   🔮 Market outlook and investment guidance")
            print("=" * 80)
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            print("Make sure the server is running: python main.py")


if __name__ == "__main__":
    demo = IndianMarketDemo()
    demo.run_full_demo()