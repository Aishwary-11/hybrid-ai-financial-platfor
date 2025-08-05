"""
Personal Aladdin Demo Script
Demonstrates key platform capabilities
"""

import requests
import json
from typing import Dict, List

BASE_URL = "http://localhost:8000/api/v1/analytics"

class PersonalAladdinDemo:
    """Demo client for Personal Aladdin platform"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
    
    def create_sample_portfolio(self):
        """Create a sample diversified portfolio"""
        print("🚀 Creating sample portfolio...")
        
        # Create portfolio
        response = requests.post(f"{self.base_url}/portfolio/create", 
                               params={"name": "Diversified Growth", "initial_cash": 100000})
        print(f"✅ Portfolio created: {response.json()}")
        
        # Add positions
        positions = [
            {"symbol": "SPY", "quantity": 100, "price": 450},    # S&P 500 ETF
            {"symbol": "QQQ", "quantity": 50, "price": 380},     # NASDAQ ETF
            {"symbol": "VTI", "quantity": 80, "price": 240},     # Total Stock Market
            {"symbol": "BND", "quantity": 200, "price": 75},     # Bond ETF
            {"symbol": "VNQ", "quantity": 30, "price": 90},      # REIT ETF
            {"symbol": "GLD", "quantity": 20, "price": 180},     # Gold ETF
        ]
        
        for pos in positions:
            response = requests.post(
                f"{self.base_url}/portfolio/Diversified Growth/add-position",
                params=pos
            )
            print(f"✅ Added position: {pos['symbol']} - {response.json()}")
        
        return "Diversified Growth"
    
    def analyze_portfolio_risk(self, portfolio_name: str):
        """Analyze portfolio risk metrics"""
        print(f"\n📊 Analyzing risk for portfolio: {portfolio_name}")
        
        # Get portfolio details
        response = requests.get(f"{self.base_url}/portfolio/{portfolio_name}")
        portfolio = response.json()["portfolio"]
        
        symbols = list(portfolio["positions"].keys())
        print(f"Portfolio symbols: {symbols}")
        
        # Calculate VaR
        response = requests.get(f"{self.base_url}/risk/var", 
                              params={"symbols": symbols, "confidence_level": 0.95})
        var_analysis = response.json()
        print(f"📉 VaR Analysis: {var_analysis['var_analysis']}")
        
        # Get comprehensive risk metrics
        response = requests.get(f"{self.base_url}/risk/metrics", 
                              params={"symbols": symbols})
        risk_metrics = response.json()
        print(f"⚠️ Risk Metrics: {risk_metrics['risk_metrics']}")
        
        return risk_metrics
    
    def run_stress_tests(self):
        """Run stress tests on sample portfolio"""
        print(f"\n🧪 Running stress tests...")
        
        # Sample portfolio weights
        portfolio_weights = {
            "SPY": 0.35,
            "QQQ": 0.20,
            "VTI": 0.20,
            "BND": 0.15,
            "VNQ": 0.05,
            "GLD": 0.05
        }
        
        scenarios = ["2008_financial_crisis", "covid_crash_2020", "dot_com_bubble"]
        
        for scenario in scenarios:
            response = requests.post(f"{self.base_url}/risk/stress-test",
                                   json={"portfolio_weights": portfolio_weights, 
                                        "scenario": scenario})
            result = response.json()
            print(f"💥 {scenario}: Portfolio Impact = {result['portfolio_impact']:.2%}")
    
    def optimize_portfolio(self):
        """Demonstrate portfolio optimization"""
        print(f"\n🎯 Running portfolio optimization...")
        
        symbols = ["SPY", "QQQ", "VTI", "BND", "VNQ", "GLD", "EFA", "EEM"]
        methods = ["mean_variance", "risk_parity"]
        
        for method in methods:
            response = requests.post(f"{self.base_url}/optimize",
                                   params={"symbols": symbols, "method": method})
            result = response.json()
            
            print(f"\n📈 {method.upper()} Optimization:")
            if "optimization" in result and "weights" in result["optimization"]:
                weights = result["optimization"]["weights"]
                for symbol, weight in weights.items():
                    print(f"  {symbol}: {weight:.2%}")
                
                if "sharpe_ratio" in result["optimization"]:
                    print(f"  Sharpe Ratio: {result['optimization']['sharpe_ratio']:.3f}")
    
    def analyze_correlations(self):
        """Analyze asset correlations"""
        print(f"\n🔗 Analyzing asset correlations...")
        
        symbols = ["SPY", "QQQ", "BND", "GLD", "VNQ"]
        response = requests.get(f"{self.base_url}/risk/correlation",
                              params={"symbols": symbols})
        
        correlation_data = response.json()
        print("Correlation Matrix:")
        correlation_matrix = correlation_data["correlation_matrix"]
        
        # Print correlation matrix
        print(f"{'':>6}", end="")
        for symbol in symbols:
            print(f"{symbol:>8}", end="")
        print()
        
        for symbol1 in symbols:
            print(f"{symbol1:>6}", end="")
            for symbol2 in symbols:
                corr = correlation_matrix[symbol1][symbol2]
                print(f"{corr:>8.3f}", end="")
            print()
    
    def detect_market_regime(self):
        """Detect current market regime"""
        print(f"\n🌊 Detecting market regime...")
        
        response = requests.get(f"{self.base_url}/analysis/regime-detection",
                              params={"symbol": "SPY", "window": 60})
        
        regime_data = response.json()
        regime_analysis = regime_data["regime_analysis"]
        
        print(f"Current Market Regime: {regime_analysis['current_regime']}")
        print(f"Recent Regime Changes: {len(regime_analysis['regime_changes'])} detected")
    
    def run_full_demo(self):
        """Run complete demonstration"""
        print("=" * 60)
        print("🏛️  PERSONAL ALADDIN - INVESTMENT MANAGEMENT PLATFORM")
        print("=" * 60)
        
        try:
            # Create and analyze portfolio
            portfolio_name = self.create_sample_portfolio()
            self.analyze_portfolio_risk(portfolio_name)
            
            # Run various analyses
            self.run_stress_tests()
            self.optimize_portfolio()
            self.analyze_correlations()
            self.detect_market_regime()
            
            print("\n" + "=" * 60)
            print("✅ Demo completed successfully!")
            print("🎯 Key Features Demonstrated:")
            print("   • Portfolio creation and management")
            print("   • Advanced risk analytics (VaR, stress tests)")
            print("   • Portfolio optimization (Mean-Variance, Risk Parity)")
            print("   • Correlation analysis")
            print("   • Market regime detection")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            print("Make sure the server is running: python main.py")


if __name__ == "__main__":
    demo = PersonalAladdinDemo()
    demo.run_full_demo()