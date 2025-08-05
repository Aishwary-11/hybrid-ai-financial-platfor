"""
Risk Prediction Model Demo
Test the specialized AI-based risk prediction capabilities
"""

import asyncio
import json
from datetime import datetime
from app.core.risk_prediction_model import (
    RiskPredictionModel,
    quick_risk_assessment,
    portfolio_stress_test,
    RiskType,
    RiskHorizon
)


async def demo_risk_prediction():
    """Demonstrate risk prediction model capabilities"""
    
    print("⚠️  AI-Based Risk Prediction Model Demo")
    print("=" * 50)
    
    # Initialize the risk prediction model
    risk_model = RiskPredictionModel()
    
    # Test cases with different portfolio configurations
    test_cases = [
        {
            "name": "Tech Portfolio",
            "symbols": ["AAPL", "GOOGL", "MSFT", "NVDA"],
            "portfolio_data": {
                "total_value": 2000000,
                "positions": {
                    "AAPL": 500000,
                    "GOOGL": 600000,
                    "MSFT": 500000,
                    "NVDA": 400000
                }
            },
            "risk_types": ["market_risk", "volatility_risk", "systemic_risk"],
            "time_horizon": "monthly"
        },
        {
            "name": "Diversified Portfolio",
            "symbols": ["SPY", "BND", "GLD", "VTI", "VXUS"],
            "portfolio_data": {
                "total_value": 1500000,
                "positions": {
                    "SPY": 600000,
                    "BND": 300000,
                    "GLD": 200000,
                    "VTI": 250000,
                    "VXUS": 150000
                }
            },
            "risk_types": ["market_risk", "credit_risk", "liquidity_risk"],
            "time_horizon": "quarterly"
        },
        {
            "name": "High-Risk Growth Portfolio",
            "symbols": ["TSLA", "ARKK", "COIN", "PLTR"],
            "portfolio_data": {
                "total_value": 1000000,
                "positions": {
                    "TSLA": 400000,
                    "ARKK": 300000,
                    "COIN": 200000,
                    "PLTR": 100000
                }
            },
            "risk_types": ["market_risk", "volatility_risk", "tail_risk"],
            "time_horizon": "weekly"
        }
    ]
    
    print("\n📊 Running Risk Prediction Tests")
    print("-" * 40)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Symbols: {', '.join(test_case['symbols'])}")
        print(f"   Portfolio Value: ${test_case['portfolio_data']['total_value']:,}")
        print(f"   Time Horizon: {test_case['time_horizon']}")
        
        # Prepare input data
        input_data = {
            "symbols": test_case["symbols"],
            "risk_types": test_case["risk_types"],
            "time_horizon": test_case["time_horizon"],
            "portfolio_data": test_case["portfolio_data"],
            "market_conditions": {"regime": "normal", "volatility": "moderate"}
        }
        
        # Run risk prediction
        try:
            result = risk_model.predict(input_data)
            
            print(f"   ✅ Risk analysis completed")
            print(f"   📈 Overall Risk Score: {result.result['overall_risk_score']:.3f}")
            print(f"   🎯 Risk Level: {result.result['risk_level']}")
            print(f"   🔍 Confidence: {result.confidence:.3f}")
            print(f"   ⚡ Market Regime: {result.result.get('market_regime', 'N/A')}")
            
            # Show portfolio risk metrics
            portfolio_metrics = result.result.get("portfolio_risk_metrics", {})
            if portfolio_metrics:
                var_95 = portfolio_metrics.get("var_95", 0)
                expected_shortfall = portfolio_metrics.get("expected_shortfall", 0)
                print(f"   💹 VaR (95%): ${var_95:,.0f}")
                print(f"   📉 Expected Shortfall: ${expected_shortfall:,.0f}")
                print(f"   🎲 Diversification Ratio: {portfolio_metrics.get('diversification_ratio', 0):.2f}")
            
            # Show top risk factors
            risk_factors = result.result.get("risk_factors", [])
            if risk_factors:
                print(f"   🔍 Top Risk Factors:")
                for rf in risk_factors[:3]:  # Show top 3
                    print(f"      - {rf['name']}: {rf['value']:.2f} ({rf['severity']})")
            
            # Show risk scenarios
            scenarios = result.result.get("risk_scenarios", [])
            if scenarios:
                print(f"   📋 Risk Scenarios: {len(scenarios)} generated")
                for scenario in scenarios[:2]:  # Show top 2
                    print(f"      - {scenario['scenario_name']}: "
                          f"{scenario['probability']:.1%} probability, "
                          f"${scenario['expected_loss']:,.0f} expected loss")
            
            # Show early warnings
            warnings = result.result.get("early_warning_signals", [])
            if warnings:
                print(f"   ⚠️  Early Warnings: {len(warnings)} detected")
                for warning in warnings:
                    print(f"      - {warning['signal']} ({warning['severity']})")
            
        except Exception as e:
            print(f"   ❌ Risk analysis failed: {e}")
    
    # Test quick risk assessment
    print(f"\n🚀 Testing Quick Risk Assessment")
    print("-" * 35)
    
    quick_test_symbols = ["AAPL", "TSLA", "SPY"]
    try:
        quick_result = quick_risk_assessment(quick_test_symbols, 500000)
        print(f"   ✅ Quick assessment completed")
        print(f"   📊 Overall Risk: {quick_result['overall_risk_score']:.3f}")
        print(f"   🎯 Risk Level: {quick_result['risk_level']}")
        print(f"   💰 Portfolio Value: $500,000")
        
    except Exception as e:
        print(f"   ❌ Quick assessment failed: {e}")
    
    # Test stress testing
    print(f"\n🧪 Testing Portfolio Stress Testing")
    print("-" * 35)
    
    stress_test_data = {
        "total_value": 1000000,
        "positions": {
            "AAPL": 300000,
            "GOOGL": 300000,
            "TSLA": 200000,
            "SPY": 200000
        }
    }
    
    try:
        stress_results = portfolio_stress_test(
            ["AAPL", "GOOGL", "TSLA", "SPY"], 
            stress_test_data
        )
        
        print(f"   ✅ Stress testing completed")
        
        # Show stress test results
        stress_data = stress_results.get("stress_results", {})
        if stress_data:
            print(f"   📊 Stress Scenarios:")
            for scenario_name, scenario_data in list(stress_data.items())[:3]:
                loss_pct = scenario_data.get("portfolio_loss_pct", 0)
                recovery_time = scenario_data.get("recovery_time_estimate", "Unknown")
                print(f"      - {scenario_name}: {loss_pct:.1%} loss, "
                      f"Recovery: {recovery_time}")
        
        # Show recommendations
        recommendations = stress_results.get("recommendations", [])
        if recommendations:
            print(f"   💡 Recommendations: {len(recommendations)} provided")
            for rec in recommendations[:2]:
                print(f"      - {rec['priority'].upper()}: {rec['recommendation'][:60]}...")
                
    except Exception as e:
        print(f"   ❌ Stress testing failed: {e}")
    
    # Test model validation
    print(f"\n🔍 Testing Model Validation")
    print("-" * 25)
    
    # Test with valid output
    valid_output = {
        "overall_risk_score": 0.65,
        "risk_level": "moderate",
        "confidence": 0.82,
        "risk_factors": [
            {"name": "AAPL_market_risk", "value": 0.6, "category": "market_risk"}
        ],
        "risk_scenarios": [
            {"scenario_name": "Base Case", "probability": 0.7}
        ],
        "portfolio_risk_metrics": {"var_95": 50000}
    }
    
    is_valid, validation_score = risk_model.validate_output(valid_output)
    print(f"   Valid Output: {'✅ Passed' if is_valid else '❌ Failed'} "
          f"(Score: {validation_score:.2f})")
    
    # Test with invalid output
    invalid_output = {
        "overall_risk_score": 1.5,  # Invalid range
        "risk_level": "extreme",
        "confidence": -0.1,  # Invalid range
        "risk_factors": "invalid_type"  # Wrong type
    }
    
    is_valid, validation_score = risk_model.validate_output(invalid_output)
    print(f"   Invalid Output: {'✅ Passed' if is_valid else '❌ Failed'} "
          f"(Score: {validation_score:.2f})")
    
    # Test market regime detection
    print(f"\n🌍 Testing Market Regime Detection")
    print("-" * 30)
    
    try:
        # Simulate market data collection
        market_data = risk_model._collect_market_data(["SPY", "VIX"])
        regime = risk_model.regime_detector.detect_regime(market_data)
        print(f"   Current Market Regime: {regime}")
        print(f"   Data Sources: {len(market_data)} symbols analyzed")
        
    except Exception as e:
        print(f"   ❌ Regime detection failed: {e}")
    
    print(f"\n🎯 Demo Summary")
    print("-" * 15)
    print(f"   • AI-based risk prediction model successfully tested")
    print(f"   • Supports multiple risk types and time horizons")
    print(f"   • Includes comprehensive stress testing and scenario analysis")
    print(f"   • Provides portfolio-level risk metrics and early warnings")
    print(f"   • Ready for integration with hybrid AI architecture")
    print(f"   • Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def test_risk_scenarios():
    """Test risk scenario generation"""
    
    print("\n📋 Risk Scenario Generation Test")
    print("-" * 30)
    
    risk_model = RiskPredictionModel()
    
    # Test scenario generation
    input_data = {
        "symbols": ["AAPL", "MSFT", "GOOGL"],
        "risk_types": ["market_risk", "systemic_risk", "tail_risk"],
        "time_horizon": "quarterly",
        "portfolio_data": {"total_value": 1000000}
    }
    
    try:
        result = risk_model.predict(input_data)
        scenarios = result.result.get("risk_scenarios", [])
        
        print(f"   Generated {len(scenarios)} risk scenarios:")
        
        for scenario in scenarios:
            print(f"   📊 {scenario['scenario_name']}:")
            print(f"      Probability: {scenario['probability']:.1%}")
            print(f"      Expected Loss: ${scenario['expected_loss']:,.0f}")
            print(f"      Max Loss: ${scenario['max_loss']:,.0f}")
            print(f"      Timeline: {scenario['timeline']}")
            print(f"      Mitigation Cost: ${scenario['mitigation_cost']:,.0f}")
            print()
            
    except Exception as e:
        print(f"   ❌ Scenario generation failed: {e}")


if __name__ == "__main__":
    print("Starting AI-Based Risk Prediction Demo...")
    
    # Run the main demo
    asyncio.run(demo_risk_prediction())
    
    # Run scenario test
    test_risk_scenarios()
    
    print("\n✅ Demo completed successfully!")