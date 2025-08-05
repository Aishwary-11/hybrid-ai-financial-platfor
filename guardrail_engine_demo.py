"""
Comprehensive Guardrail Engine Demo
Test the validation and safety system for AI outputs
"""

import asyncio
import json
from datetime import datetime
from app.core.guardrail_engine import (
    ComprehensiveGuardrailEngine,
    GuardrailType,
    ViolationSeverity,
    validate_model_output,
    check_output_safety
)
from app.core.hybrid_ai_engine import ModelOutput, TaskCategory, ModelType


async def demo_guardrail_engine():
    """Demonstrate comprehensive guardrail engine capabilities"""
    
    print("🛡️  Comprehensive Guardrail Engine Demo")
    print("=" * 50)
    
    # Initialize the guardrail engine
    guardrail_engine = ComprehensiveGuardrailEngine()
    
    # Test cases with different types of outputs and potential issues
    test_cases = [
        {
            "name": "Valid Earnings Analysis",
            "output": ModelOutput(
                result={
                    "investment_signal": "bullish",
                    "signal_strength": 0.75,
                    "key_themes": ["revenue_growth", "margin_expansion"],
                    "risk_factors": ["competition", "market_volatility"],
                    "confidence_drivers": {"positive_sentiment": 0.8}
                },
                confidence=0.82,
                model_type=ModelType.SPECIALIZED,
                task_category=TaskCategory.EARNINGS_PREDICTION,
                timestamp=datetime.now(),
                validation_score=0.85,
                guardrail_passed=True
            ),
            "context": {"symbols": ["AAPL"], "source": "earnings_call"}
        },
        {
            "name": "Invalid Structure - Missing Fields",
            "output": ModelOutput(
                result={
                    "investment_signal": "bullish",
                    # Missing required fields: signal_strength, key_themes, risk_factors
                },
                confidence=0.90,
                model_type=ModelType.SPECIALIZED,
                task_category=TaskCategory.EARNINGS_PREDICTION,
                timestamp=datetime.now(),
                validation_score=0.60,
                guardrail_passed=True
            ),
            "context": {"symbols": ["TSLA"]}
        },
        {
            "name": "Potential Hallucination - Impossible Values",
            "output": ModelOutput(
                result={
                    "overall_sentiment": "bullish",
                    "sentiment_score": 2.5,  # Invalid range (should be -1 to 1)
                    "confidence": 1.2,  # Invalid range (should be 0 to 1)
                    "entity_sentiments": {"NVDA": 15.0},  # Unrealistic value
                    "market_impact": {"NVDA": {"price_change": 500.0}}  # 50,000% change
                },
                confidence=0.95,
                model_type=ModelType.SPECIALIZED,
                task_category=TaskCategory.SENTIMENT_ANALYSIS,
                timestamp=datetime.now(),
                validation_score=0.30,
                guardrail_passed=True
            ),
            "context": {"symbols": ["NVDA"]}
        },
        {
            "name": "Ethical Compliance Issues",
            "output": ModelOutput(
                result={
                    "overall_risk_score": 0.65,
                    "risk_level": "moderate",
                    "confidence": 0.80,
                    "risk_factors": [{"name": "market_risk", "value": 0.6}],
                    "recommendations": [
                        "This investment is guaranteed profit with no risk",
                        "Based on insider information, buy immediately",
                        "Only suitable for wealthy male investors"
                    ]
                },
                confidence=0.80,
                model_type=ModelType.SPECIALIZED,
                task_category=TaskCategory.RISK_ASSESSMENT,
                timestamp=datetime.now(),
                validation_score=0.75,
                guardrail_passed=True
            ),
            "context": {"symbols": ["SPY"]}
        },
        {
            "name": "Fact-Checking Issues",
            "output": ModelOutput(
                result={
                    "top_themes": [
                        {"name": "artificial_intelligence", "strength": 0.85}
                    ],
                    "theme_strength": {"artificial_intelligence": 0.85},
                    "investment_vehicles": {
                        "artificial_intelligence": ["AAPL at $50,000 per share"]
                    },
                    "market_analysis": "Apple stock will increase by 10,000% next week"
                },
                confidence=0.70,
                model_type=ModelType.SPECIALIZED,
                task_category=TaskCategory.THEMATIC_IDENTIFICATION,
                timestamp=datetime.now(),
                validation_score=0.65,
                guardrail_passed=True
            ),
            "context": {"symbols": ["AAPL"]}
        }
    ]
    
    print("\n🔍 Running Guardrail Validation Tests")
    print("-" * 45)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Task Category: {test_case['output'].task_category.value}")
        print(f"   Model Confidence: {test_case['output'].confidence:.2f}")
        
        try:
            # Run comprehensive validation
            result = await guardrail_engine.validate_output(
                test_case["output"], 
                test_case["context"]
            )
            
            print(f"   ✅ Validation completed")
            print(f"   🎯 Overall Score: {result.overall_score:.3f}")
            print(f"   🛡️  Validation Passed: {'✅ Yes' if result.passed else '❌ No'}")
            print(f"   ⚠️  Violations: {len(result.violations)}")
            print(f"   ⏱️  Processing Time: {result.processing_time:.3f}s")
            
            # Show violations by severity
            if result.violations:
                severity_counts = {}
                for violation in result.violations:
                    severity = violation.severity.value
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                print(f"   📊 Violation Breakdown:")
                for severity, count in severity_counts.items():
                    print(f"      - {severity.title()}: {count}")
                
                # Show top violations
                print(f"   🔍 Top Violations:")
                for violation in result.violations[:3]:  # Show top 3
                    print(f"      - {violation.violation_type.value}: {violation.description[:60]}...")
            
            # Show warnings
            if result.warnings:
                print(f"   ⚠️  Warnings: {len(result.warnings)}")
                for warning in result.warnings[:2]:
                    print(f"      - {warning}")
            
            # Show recommendations
            if result.recommendations:
                print(f"   💡 Recommendations: {len(result.recommendations)}")
                for rec in result.recommendations[:2]:
                    print(f"      - {rec}")
            
        except Exception as e:
            print(f"   ❌ Validation failed: {e}")
    
    # Test individual guardrails
    print(f"\n🔧 Testing Individual Guardrails")
    print("-" * 35)
    
    # Test structure validation
    print(f"\n   Structure Validation:")
    structure_guardrail = guardrail_engine.guardrails[GuardrailType.STRUCTURE_VALIDATION]
    
    invalid_output = ModelOutput(
        result={},  # Empty result
        confidence=1.5,  # Invalid confidence
        model_type=ModelType.SPECIALIZED,
        task_category=TaskCategory.SENTIMENT_ANALYSIS,
        timestamp=datetime.now(),
        validation_score=0.0,
        guardrail_passed=True
    )
    
    structure_violations = await structure_guardrail.validate(invalid_output, {})
    print(f"      Violations found: {len(structure_violations)}")
    for violation in structure_violations[:2]:
        print(f"      - {violation.description}")
    
    # Test hallucination detection
    print(f"\n   Hallucination Detection:")
    hallucination_guardrail = guardrail_engine.guardrails[GuardrailType.HALLUCINATION_DETECTION]
    
    suspicious_output = ModelOutput(
        result={
            "analysis": "I think this stock will definitely go up 1000% based on my opinion",
            "price_prediction": 999999999.99,  # Unrealistic value
            "confidence_note": "This is absolutely guaranteed with secret information"
        },
        confidence=0.75,
        model_type=ModelType.SPECIALIZED,
        task_category=TaskCategory.MARKET_ANALYSIS,
        timestamp=datetime.now(),
        validation_score=0.50,
        guardrail_passed=True
    )
    
    hallucination_violations = await hallucination_guardrail.validate(suspicious_output, {})
    print(f"      Violations found: {len(hallucination_violations)}")
    for violation in hallucination_violations[:2]:
        print(f"      - {violation.description}")
    
    # Test ethical compliance
    print(f"\n   Ethical Compliance:")
    ethical_guardrail = guardrail_engine.guardrails[GuardrailType.ETHICAL_COMPLIANCE]
    
    unethical_output = ModelOutput(
        result={
            "recommendation": "This guaranteed profit investment with insider information is only for men",
            "risk_disclosure": "No risks involved, completely safe"
        },
        confidence=0.80,
        model_type=ModelType.SPECIALIZED,
        task_category=TaskCategory.MARKET_ANALYSIS,
        timestamp=datetime.now(),
        validation_score=0.60,
        guardrail_passed=True
    )
    
    ethical_violations = await ethical_guardrail.validate(unethical_output, {})
    print(f"      Violations found: {len(ethical_violations)}")
    for violation in ethical_violations[:2]:
        print(f"      - {violation.description}")
    
    # Test batch validation
    print(f"\n🔄 Testing Batch Validation")
    print("-" * 25)
    
    batch_outputs = [test_case["output"] for test_case in test_cases[:3]]
    batch_results = await guardrail_engine.batch_validate(batch_outputs)
    
    print(f"   Batch size: {len(batch_outputs)}")
    print(f"   Results: {len(batch_results)}")
    
    passed_count = sum(1 for result in batch_results if result.passed)
    failed_count = len(batch_results) - passed_count
    
    print(f"   ✅ Passed: {passed_count}")
    print(f"   ❌ Failed: {failed_count}")
    
    avg_score = sum(result.overall_score for result in batch_results) / len(batch_results)
    print(f"   📊 Average Score: {avg_score:.3f}")
    
    # Test guardrail configuration
    print(f"\n⚙️  Testing Guardrail Configuration")
    print("-" * 30)
    
    # Configure individual guardrails
    print(f"   Configuring guardrails...")
    
    guardrail_engine.configure_guardrail(
        GuardrailType.STRUCTURE_VALIDATION, 
        enabled=True, 
        threshold=0.9
    )
    
    guardrail_engine.configure_guardrail(
        GuardrailType.HALLUCINATION_DETECTION, 
        enabled=True, 
        threshold=0.8
    )
    
    print(f"   ✅ Structure validation: enabled, threshold=0.9")
    print(f"   ✅ Hallucination detection: enabled, threshold=0.8")
    
    # Test performance reporting
    print(f"\n📊 Testing Performance Reporting")
    print("-" * 30)
    
    performance_report = guardrail_engine.get_performance_report()
    
    print(f"   Total validations: {performance_report['overall_stats']['total_validations']}")
    print(f"   Total violations: {performance_report['overall_stats']['total_violations']}")
    print(f"   Avg processing time: {performance_report['overall_stats']['avg_processing_time']:.3f}s")
    
    if performance_report['guardrail_performance']:
        print(f"   Guardrail performance:")
        for guardrail_type, metrics in performance_report['guardrail_performance'].items():
            print(f"      - {guardrail_type}: {metrics['total_runs']} runs, "
                  f"{metrics['avg_violations']:.1f} avg violations")
    
    # Test violation summary
    print(f"\n📈 Testing Violation Summary")
    print("-" * 25)
    
    violation_summary = guardrail_engine.get_violation_summary(days=1)
    
    print(f"   Period: {violation_summary['period_days']} days")
    print(f"   Total violations: {violation_summary['total_violations']}")
    
    if violation_summary['by_severity']:
        print(f"   By severity:")
        for severity, count in violation_summary['by_severity'].items():
            print(f"      - {severity.title()}: {count}")
    
    if violation_summary['top_violations']:
        print(f"   Top violation types:")
        for violation_desc, count in violation_summary['top_violations'][:3]:
            print(f"      - {violation_desc}: {count}")
    
    # Test utility functions
    print(f"\n🛠️  Testing Utility Functions")
    print("-" * 25)
    
    # Test quick validation
    test_output = test_cases[0]["output"]
    quick_result = await validate_model_output(test_output, test_cases[0]["context"])
    print(f"   Quick validation: {'✅ Passed' if quick_result.passed else '❌ Failed'}")
    print(f"   Score: {quick_result.overall_score:.3f}")
    
    # Test safety check
    safety_result = check_output_safety(test_output)
    print(f"   Safety check: {'✅ Safe' if safety_result else '❌ Unsafe'}")
    
    # Test with unsafe output
    unsafe_output = ModelOutput(
        result={
            "recommendation": "This guaranteed profit scheme with insider information is risk-free"
        },
        confidence=0.95,
        model_type=ModelType.SPECIALIZED,
        task_category=TaskCategory.MARKET_ANALYSIS,
        timestamp=datetime.now(),
        validation_score=0.80,
        guardrail_passed=True
    )
    
    unsafe_safety_result = check_output_safety(unsafe_output)
    print(f"   Unsafe output check: {'✅ Safe' if unsafe_safety_result else '❌ Unsafe'}")
    
    print(f"\n🎯 Demo Summary")
    print("-" * 15)
    print(f"   • Comprehensive guardrail engine successfully tested")
    print(f"   • Multiple validation types: structure, hallucination, fact-checking, ethics")
    print(f"   • Batch processing and performance monitoring capabilities")
    print(f"   • Configurable thresholds and individual guardrail control")
    print(f"   • Real-time violation tracking and reporting")
    print(f"   • Ready for integration with hybrid AI architecture")
    print(f"   • Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def test_guardrail_scenarios():
    """Test specific guardrail scenarios"""
    
    print("\n🎭 Guardrail Scenario Testing")
    print("-" * 30)
    
    # Test different violation severities
    scenarios = [
        {
            "name": "Critical Structure Violation",
            "output": ModelOutput(
                result=None,  # Critical: no result
                confidence=0.8,
                model_type=ModelType.SPECIALIZED,
                task_category=TaskCategory.SENTIMENT_ANALYSIS,
                timestamp=datetime.now(),
                validation_score=0.0,
                guardrail_passed=True
            ),
            "expected_severity": "critical"
        },
        {
            "name": "High Ethical Violation",
            "output": ModelOutput(
                result={
                    "recommendation": "Guaranteed profit with insider information",
                    "risk_level": "none"
                },
                confidence=0.9,
                model_type=ModelType.SPECIALIZED,
                task_category=TaskCategory.MARKET_ANALYSIS,
                timestamp=datetime.now(),
                validation_score=0.7,
                guardrail_passed=True
            ),
            "expected_severity": "high"
        },
        {
            "name": "Medium Range Violation",
            "output": ModelOutput(
                result={
                    "overall_sentiment": "bullish",
                    "sentiment_score": 1.5,  # Out of range
                    "confidence": 0.8,
                    "entity_sentiments": {}
                },
                confidence=0.8,
                model_type=ModelType.SPECIALIZED,
                task_category=TaskCategory.SENTIMENT_ANALYSIS,
                timestamp=datetime.now(),
                validation_score=0.6,
                guardrail_passed=True
            ),
            "expected_severity": "medium"
        }
    ]
    
    print(f"   Testing {len(scenarios)} violation scenarios:")
    
    for scenario in scenarios:
        print(f"\n   📋 {scenario['name']}:")
        
        # Quick safety check
        is_safe = check_output_safety(scenario["output"])
        print(f"      Safety check: {'✅ Safe' if is_safe else '❌ Unsafe'}")
        print(f"      Expected severity: {scenario['expected_severity']}")


if __name__ == "__main__":
    print("Starting Comprehensive Guardrail Engine Demo...")
    
    # Run the main demo
    asyncio.run(demo_guardrail_engine())
    
    # Run scenario tests
    test_guardrail_scenarios()
    
    print("\n✅ Demo completed successfully!")