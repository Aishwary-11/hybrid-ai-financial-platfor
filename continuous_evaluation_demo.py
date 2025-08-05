"""
Continuous Evaluation Pipeline Demo
Test the automated testing and evaluation system with LLM judges
"""

import asyncio
import json
from datetime import datetime
from app.core.continuous_evaluation import (
    ContinuousEvaluationPipeline,
    TestCase,
    TestType,
    create_sentiment_test_case,
    create_earnings_test_case,
    create_risk_test_case,
    create_evaluation_pipeline
)
from app.core.hybrid_ai_engine import HybridAIEngine, TaskCategory


async def demo_continuous_evaluation():
    """Demonstrate continuous evaluation pipeline capabilities"""
    
    print("🔄 Continuous Evaluation Pipeline Demo")
    print("=" * 50)
    
    # Initialize AI engine and evaluation pipeline
    ai_engine = HybridAIEngine()
    evaluation_pipeline = create_evaluation_pipeline(ai_engine)
    
    print("\n📝 Creating Test Cases")
    print("-" * 25)
    
    # Create test cases for different categories
    test_cases = []
    
    # Sentiment analysis test cases
    sentiment_tests = [
        create_sentiment_test_case(
            "sent_001",
            "Apple reported strong quarterly earnings with revenue growth of 15% and expanding margins",
            "bullish",
            0.75
        ),
        create_sentiment_test_case(
            "sent_002", 
            "Tesla faces mounting challenges with declining sales and increased competition",
            "bearish",
            -0.65
        ),
        create_sentiment_test_case(
            "sent_003",
            "Microsoft maintains steady performance with mixed signals from different business segments",
            "neutral",
            0.05
        )
    ]
    
    # Earnings analysis test cases
    earnings_tests = [
        create_earnings_test_case(
            "earn_001",
            "We exceeded expectations this quarter with revenue up 20% and strong margin expansion across all segments. Management raised full-year guidance and expressed confidence in continued growth momentum.",
            "bullish",
            0.80
        ),
        create_earnings_test_case(
            "earn_002",
            "This quarter presented significant challenges with revenue declining 10% and margin pressure from increased competition. We're implementing cost reduction measures.",
            "bearish",
            0.25
        )
    ]
    
    # Risk assessment test cases
    risk_tests = [
        create_risk_test_case(
            "risk_001",
            ["AAPL", "MSFT", "GOOGL"],
            "moderate",
            0.45
        ),
        create_risk_test_case(
            "risk_002",
            ["TSLA", "ARKK", "COIN"],
            "high",
            0.75
        )
    ]
    
    test_cases.extend(sentiment_tests)
    test_cases.extend(earnings_tests)
    test_cases.extend(risk_tests)
    
    print(f"   Created {len(test_cases)} test cases:")
    print(f"   - Sentiment Analysis: {len(sentiment_tests)}")
    print(f"   - Earnings Analysis: {len(earnings_tests)}")
    print(f"   - Risk Assessment: {len(risk_tests)}")
    
    # Add test cases to pipeline
    for test_case in test_cases:
        await evaluation_pipeline.add_test_case(test_case)
    
    print(f"\n🧪 Running Comprehensive Evaluation")
    print("-" * 35)
    
    # Run evaluation for all categories
    evaluation_results = await evaluation_pipeline.run_evaluation()
    
    print(f"   ✅ Evaluation completed")
    print(f"   📊 Evaluation ID: {evaluation_results['evaluation_id']}")
    print(f"   🎯 Categories tested: {len(evaluation_results['categories_tested'])}")
    print(f"   ⏱️  Execution time: {evaluation_results['summary']['execution_time']:.2f}s")
    print(f"   📈 Overall pass rate: {evaluation_results['summary']['pass_rate']:.1%}")
    print(f"   ✅ Tests passed: {evaluation_results['summary']['total_passed']}/{evaluation_results['summary']['total_tests']}")
    
    # Show detailed results by category
    print(f"\n📋 Results by Category:")
    for category, results in evaluation_results["results"].items():
        summary = results["summary"]
        print(f"   {category.title()}:")
        print(f"      Tests: {summary['total_tests']}")
        print(f"      Passed: {summary['passed']}")
        print(f"      Failed: {summary['failed']}")
        print(f"      Warnings: {summary['warnings']}")
        print(f"      Avg Score: {summary['avg_score']:.3f}")
    
    # Show alerts if any
    if evaluation_results["alerts"]:
        print(f"\n⚠️  Alerts ({len(evaluation_results['alerts'])}):")
        for alert in evaluation_results["alerts"]:
            print(f"      - {alert['type']}: {alert.get('category', 'general')}")
            if 'regression_percentage' in alert:
                print(f"        Regression: {alert['regression_percentage']:.1f}%")
    else:
        print(f"\n✅ No alerts detected")
    
    # Test individual judges
    print(f"\n🏛️  Testing Individual Judges")
    print("-" * 25)
    
    # Test accuracy judge
    print(f"   Accuracy Judge:")
    accuracy_judge = evaluation_pipeline.judges[TaskCategory.SENTIMENT_ANALYSIS][TestType.ACCURACY_TEST]
    
    # Create a test model output
    from app.core.hybrid_ai_engine import ModelOutput, ModelType
    
    test_output = ModelOutput(
        result={
            "overall_sentiment": "bullish",
            "sentiment_score": 0.72,
            "confidence": 0.85,
            "entity_sentiments": {"AAPL": 0.75}
        },
        confidence=0.85,
        model_type=ModelType.SPECIALIZED,
        task_category=TaskCategory.SENTIMENT_ANALYSIS,
        timestamp=datetime.now(),
        validation_score=0.80,
        guardrail_passed=True
    )
    
    accuracy_result = await accuracy_judge.evaluate(sentiment_tests[0], test_output)
    print(f"      Result: {accuracy_result.result.value}")
    print(f"      Score: {accuracy_result.score:.3f}")
    print(f"      Feedback: {accuracy_result.judge_feedback}")
    
    # Test consistency judge
    print(f"\n   Consistency Judge:")
    consistency_judge = evaluation_pipeline.judges[TaskCategory.SENTIMENT_ANALYSIS][TestType.CONSISTENCY_TEST]
    
    consistency_result = await consistency_judge.evaluate(sentiment_tests[0], test_output)
    print(f"      Result: {consistency_result.result.value}")
    print(f"      Score: {consistency_result.score:.3f}")
    print(f"      Feedback: {consistency_result.judge_feedback}")
    
    # Test specific category evaluation
    print(f"\n🎯 Testing Category-Specific Evaluation")
    print("-" * 35)
    
    # Run evaluation for just sentiment analysis
    sentiment_results = await evaluation_pipeline.run_evaluation(
        task_category=TaskCategory.SENTIMENT_ANALYSIS,
        test_types=[TestType.ACCURACY_TEST, TestType.CONSISTENCY_TEST]
    )
    
    print(f"   Sentiment Analysis Results:")
    if TaskCategory.SENTIMENT_ANALYSIS.value in sentiment_results["results"]:
        sent_summary = sentiment_results["results"][TaskCategory.SENTIMENT_ANALYSIS.value]["summary"]
        print(f"      Total tests: {sent_summary['total_tests']}")
        print(f"      Pass rate: {sent_summary['passed'] / sent_summary['total_tests']:.1%}")
        print(f"      Average score: {sent_summary['avg_score']:.3f}")
    
    # Test performance tracking
    print(f"\n📊 Testing Performance Tracking")
    print("-" * 30)
    
    # Run multiple evaluations to build history
    print(f"   Running multiple evaluations for trend analysis...")
    
    for i in range(3):
        await evaluation_pipeline.run_evaluation()
        print(f"      Evaluation {i+2} completed")
    
    # Get evaluation report
    report = await evaluation_pipeline.get_evaluation_report(days=1)
    
    print(f"   📈 Performance Report:")
    print(f"      Total evaluations: {report['total_evaluations']}")
    print(f"      Total tests: {report['total_tests']}")
    print(f"      Average pass rate: {report['avg_pass_rate']:.1%}")
    print(f"      Total alerts: {report['total_alerts']}")
    
    if 'performance_trends' in report:
        trends = report['performance_trends']
        print(f"      Performance trend: {trends.get('trend', 'unknown')}")
        if 'avg_recent_pass_rate' in trends:
            print(f"      Recent avg pass rate: {trends['avg_recent_pass_rate']:.1%}")
    
    # Test model health assessment
    print(f"\n🏥 Testing Model Health Assessment")
    print("-" * 30)
    
    model_health = await evaluation_pipeline._assess_model_health()
    
    print(f"   Model Health Status:")
    for category, health_info in model_health.items():
        print(f"      {category}: {health_info['health']} (score: {health_info['score']:.3f})")
    
    # Test daily report generation
    print(f"\n📋 Testing Daily Report Generation")
    print("-" * 30)
    
    daily_report = await evaluation_pipeline._generate_daily_report(evaluation_results)
    
    print(f"   Daily Report Generated:")
    print(f"      Report date: {daily_report['report_date']}")
    print(f"      Recommendations: {len(daily_report['recommendations'])}")
    
    if daily_report['recommendations']:
        print(f"      Top recommendations:")
        for rec in daily_report['recommendations'][:3]:
            print(f"         - {rec}")
    
    # Test regression detection
    print(f"\n🔍 Testing Regression Detection")
    print("-" * 25)
    
    # Simulate a performance regression
    print(f"   Simulating performance regression...")
    
    # Manually set a baseline
    evaluation_pipeline.performance_baselines[TaskCategory.SENTIMENT_ANALYSIS.value] = {
        "avg_score": 0.9,  # High baseline
        "timestamp": datetime.now()
    }
    
    # Create a poor performance result
    poor_results = {
        "results": {
            TaskCategory.SENTIMENT_ANALYSIS.value: {
                "summary": {"avg_score": 0.6}  # Significant drop
            }
        }
    }
    
    regression_alerts = await evaluation_pipeline._check_for_regressions(poor_results)
    
    if regression_alerts:
        print(f"   ⚠️  Regression detected:")
        for alert in regression_alerts:
            print(f"      Type: {alert['type']}")
            print(f"      Category: {alert['category']}")
            print(f"      Regression: {alert['regression_percentage']:.1f}%")
            print(f"      Severity: {alert['severity']}")
    else:
        print(f"   ✅ No regressions detected")
    
    # Test evaluation statistics
    print(f"\n📊 Evaluation Statistics")
    print("-" * 20)
    
    stats = evaluation_pipeline.evaluation_stats
    print(f"   Total evaluations run: {stats['total_evaluations']}")
    print(f"   Total test cases: {stats['total_test_cases']}")
    print(f"   Average accuracy: {stats['avg_accuracy']:.3f}")
    print(f"   Average consistency: {stats['avg_consistency']:.3f}")
    if stats['last_evaluation']:
        print(f"   Last evaluation: {stats['last_evaluation'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n🎯 Demo Summary")
    print("-" * 15)
    print(f"   • Continuous evaluation pipeline successfully tested")
    print(f"   • LLM judges for accuracy and consistency evaluation")
    print(f"   • Automated test case execution and scoring")
    print(f"   • Performance regression detection and alerting")
    print(f"   • Comprehensive reporting and trend analysis")
    print(f"   • Ready for daily automated evaluation scheduling")
    print(f"   • Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def test_judge_criteria():
    """Test judge evaluation criteria"""
    
    print("\n🏛️  Judge Evaluation Criteria Test")
    print("-" * 30)
    
    # Test accuracy judge criteria
    from app.core.continuous_evaluation import AccuracyJudge, ConsistencyJudge
    
    accuracy_judge = AccuracyJudge(TaskCategory.SENTIMENT_ANALYSIS)
    consistency_judge = ConsistencyJudge(TaskCategory.SENTIMENT_ANALYSIS)
    
    print(f"   Accuracy Judge Criteria:")
    for criterion, description in accuracy_judge.get_evaluation_criteria().items():
        print(f"      - {criterion}: {description}")
    
    print(f"\n   Consistency Judge Criteria:")
    for criterion, description in consistency_judge.get_evaluation_criteria().items():
        print(f"      - {criterion}: {description}")


def test_test_case_creation():
    """Test test case creation utilities"""
    
    print("\n📝 Test Case Creation Utilities")
    print("-" * 30)
    
    # Test sentiment test case
    sentiment_case = create_sentiment_test_case(
        "test_sent",
        "Strong earnings beat with positive guidance",
        "bullish",
        0.8
    )
    
    print(f"   Sentiment Test Case:")
    print(f"      ID: {sentiment_case.test_id}")
    print(f"      Category: {sentiment_case.task_category.value}")
    print(f"      Expected sentiment: {sentiment_case.expected_output['overall_sentiment']}")
    print(f"      Expected score: {sentiment_case.expected_output['sentiment_score']}")
    
    # Test earnings test case
    earnings_case = create_earnings_test_case(
        "test_earn",
        "Revenue exceeded expectations with strong margin expansion",
        "bullish",
        0.85
    )
    
    print(f"\n   Earnings Test Case:")
    print(f"      ID: {earnings_case.test_id}")
    print(f"      Category: {earnings_case.task_category.value}")
    print(f"      Expected signal: {earnings_case.expected_output['investment_signal']}")
    print(f"      Expected strength: {earnings_case.expected_output['signal_strength']}")
    
    # Test risk test case
    risk_case = create_risk_test_case(
        "test_risk",
        ["AAPL", "MSFT"],
        "moderate",
        0.5
    )
    
    print(f"\n   Risk Test Case:")
    print(f"      ID: {risk_case.test_id}")
    print(f"      Category: {risk_case.task_category.value}")
    print(f"      Symbols: {risk_case.input_data['symbols']}")
    print(f"      Expected risk level: {risk_case.expected_output['risk_level']}")


if __name__ == "__main__":
    print("Starting Continuous Evaluation Pipeline Demo...")
    
    # Run the main demo
    asyncio.run(demo_continuous_evaluation())
    
    # Run additional tests
    test_judge_criteria()
    test_test_case_creation()
    
    print("\n✅ Demo completed successfully!")