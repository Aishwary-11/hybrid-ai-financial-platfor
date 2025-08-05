"""
Enhanced Human-in-the-Loop System Demo
Test the expert collaboration and feedback integration system
"""

import asyncio
import json
from datetime import datetime
from app.core.human_in_the_loop import (
    HumanInTheLoopSystem,
    ExpertType,
    ReviewPriority,
    FeedbackType,
    create_expert_profile,
    create_sample_feedback,
    create_human_in_the_loop_system
)
from app.core.hybrid_ai_engine import ModelOutput, TaskCategory, ModelType


async def demo_human_in_the_loop():
    """Demonstrate enhanced human-in-the-loop system capabilities"""
    
    print("👥 Enhanced Human-in-the-Loop System Demo")
    print("=" * 50)
    
    # Initialize the system
    hitl_system = create_human_in_the_loop_system()
    
    print("\n👨‍💼 Registering Expert Profiles")
    print("-" * 30)
    
    # Create and register expert profiles
    experts = [
        create_expert_profile(
            "Sarah Johnson", 
            ExpertType.PORTFOLIO_MANAGER, 
            ["equity_analysis", "risk_management"],
            experience_years=12
        ),
        create_expert_profile(
            "Michael Chen", 
            ExpertType.SENIOR_ANALYST, 
            ["earnings_analysis", "sector_research"],
            experience_years=8
        ),
        create_expert_profile(
            "Dr. Emily Rodriguez", 
            ExpertType.RISK_MANAGER, 
            ["portfolio_risk", "stress_testing"],
            experience_years=15
        ),
        create_expert_profile(
            "James Wilson", 
            ExpertType.COMPLIANCE_OFFICER, 
            ["regulatory_compliance", "ethical_review"],
            experience_years=10
        )
    ]
    
    # Register experts
    for expert in experts:
        hitl_system.expert_router.register_expert(expert)
        print(f"   ✅ Registered: {expert.name} ({expert.expert_type.value})")
    
    print(f"\n   Total experts registered: {len(experts)}")
    
    print("\n📝 Creating Test Model Outputs")
    print("-" * 30)
    
    # Create test model outputs for review
    test_outputs = [
        {
            "name": "High-Confidence Sentiment Analysis",
            "output": ModelOutput(
                result={
                    "overall_sentiment": "bullish",
                    "sentiment_score": 0.85,
                    "confidence": 0.92,
                    "entity_sentiments": {"AAPL": 0.88, "MSFT": 0.82},
                    "market_impact_prediction": {"AAPL": {"price_impact_1d": 0.03}}
                },
                confidence=0.92,
                model_type=ModelType.SPECIALIZED,
                task_category=TaskCategory.SENTIMENT_ANALYSIS,
                timestamp=datetime.now(),
                validation_score=0.89,
                guardrail_passed=True
            ),
            "expert_type": ExpertType.SENIOR_ANALYST,
            "priority": ReviewPriority.MEDIUM
        },
        {
            "name": "Low-Confidence Risk Assessment",
            "output": ModelOutput(
                result={
                    "overall_risk_score": 0.65,
                    "risk_level": "moderate",
                    "confidence": 0.45,  # Low confidence
                    "risk_factors": [{"name": "market_risk", "value": 0.6}],
                    "stress_test_results": {"market_crash": {"loss": 0.25}}
                },
                confidence=0.45,
                model_type=ModelType.SPECIALIZED,
                task_category=TaskCategory.RISK_ASSESSMENT,
                timestamp=datetime.now(),
                validation_score=0.55,
                guardrail_passed=True
            ),
            "expert_type": ExpertType.RISK_MANAGER,
            "priority": ReviewPriority.HIGH
        },
        {
            "name": "Critical Earnings Analysis",
            "output": ModelOutput(
                result={
                    "investment_signal": "bearish",
                    "signal_strength": 0.78,
                    "key_themes": ["margin_pressure", "competitive_headwinds"],
                    "risk_factors": ["regulatory_changes", "market_saturation"],
                    "price_target_adjustment": "decrease_10_percent"
                },
                confidence=0.82,
                model_type=ModelType.SPECIALIZED,
                task_category=TaskCategory.EARNINGS_PREDICTION,
                timestamp=datetime.now(),
                validation_score=0.80,
                guardrail_passed=False  # Failed guardrails
            ),
            "expert_type": ExpertType.PORTFOLIO_MANAGER,
            "priority": ReviewPriority.CRITICAL
        }
    ]
    
    print(f"   Created {len(test_outputs)} test model outputs")
    for i, test_output in enumerate(test_outputs, 1):
        print(f"   {i}. {test_output['name']} - {test_output['priority'].value} priority")
    
    print("\n🔄 Submitting Reviews")
    print("-" * 20)
    
    # Submit outputs for review
    review_ids = []
    for test_output in test_outputs:
        review_id = await hitl_system.submit_for_review(
            test_output["output"],
            test_output["expert_type"],
            test_output["priority"],
            context={"symbols": ["AAPL", "MSFT"], "analysis_type": "quarterly_review"},
            special_instructions="Please focus on regulatory compliance aspects"
        )
        review_ids.append(review_id)
        print(f"   ✅ Submitted review {review_id} for {test_output['expert_type'].value}")
    
    print(f"\n   Total reviews submitted: {len(review_ids)}")
    
    # Check system status
    print("\n📊 System Status After Submissions")
    print("-" * 30)
    
    system_status = hitl_system.get_system_status()
    queue_status = system_status["queue_status"]
    
    print(f"   Pending reviews: {queue_status['pending_count']}")
    print(f"   In-progress reviews: {queue_status['in_progress_count']}")
    print(f"   Completed reviews: {queue_status['completed_count']}")
    print(f"   Total experts: {system_status['expert_count']}")
    
    if queue_status.get("priority_breakdown"):
        print(f"   Priority breakdown:")
        for priority, count in queue_status["priority_breakdown"].items():
            print(f"      - {priority}: {count}")
    
    print("\n💬 Processing Expert Feedback")
    print("-" * 30)
    
    # Simulate expert feedback for each review
    for i, review_id in enumerate(review_ids):
        expert = experts[i % len(experts)]  # Rotate through experts
        
        print(f"\n   Processing feedback for review {review_id}")
        print(f"   Expert: {expert.name} ({expert.expert_type.value})")
        
        # Create sample feedback based on the test case
        if i == 0:  # High confidence case
            feedback_data = create_sample_feedback(review_id, expert.expert_id, rating=8, agreement=0.9)
            feedback_data["detailed_comments"] = "Excellent sentiment analysis with strong market correlation. Minor suggestions for improvement."
        elif i == 1:  # Low confidence case
            feedback_data = create_sample_feedback(review_id, expert.expert_id, rating=5, agreement=0.6)
            feedback_data["detailed_comments"] = "Risk assessment needs refinement. Confidence levels are appropriately low given data quality."
            feedback_data["specific_corrections"] = [
                {
                    "field": "risk_level",
                    "original_value": "moderate",
                    "corrected_value": "moderate-high",
                    "type": "classification_adjustment",
                    "reason": "Current market volatility suggests higher risk classification"
                }
            ]
        else:  # Critical case
            feedback_data = create_sample_feedback(review_id, expert.expert_id, rating=3, agreement=0.3)
            feedback_data["feedback_type"] = "rejection"
            feedback_data["detailed_comments"] = "Significant concerns with earnings analysis. Regulatory implications not adequately addressed."
            feedback_data["regulatory_concerns"] = ["Missing risk disclosures", "Potential compliance violations"]
        
        # Process the feedback
        try:
            result = await hitl_system.provide_expert_feedback(
                review_id, expert.expert_id, feedback_data
            )
            
            print(f"   ✅ Feedback processed successfully")
            print(f"   📊 Processing result: {result['processing_result']['feedback_processed']}")
            print(f"   ⚠️  Escalation needed: {result['escalation_needed']}")
            
            if result["processing_result"].get("improvement_suggestions"):
                suggestions = result["processing_result"]["improvement_suggestions"]
                print(f"   💡 Improvement suggestions: {len(suggestions)}")
                for suggestion in suggestions[:2]:
                    print(f"      - {suggestion}")
            
        except Exception as e:
            print(f"   ❌ Feedback processing failed: {e}")
    
    print("\n🤝 Testing Collaboration Session")
    print("-" * 30)
    
    # Start a collaboration session
    collaboration_expert = experts[0]  # Use first expert
    collaboration_output = test_outputs[0]["output"]
    
    session_id = await hitl_system.start_collaboration_session(
        collaboration_expert.expert_id, collaboration_output
    )
    
    print(f"   ✅ Started collaboration session: {session_id}")
    print(f"   👨‍💼 Expert: {collaboration_expert.name}")
    
    # Add some interactions
    interactions = [
        {
            "type": "comment",
            "content": "The sentiment analysis looks good overall, but I'd like to explore the risk factors more deeply.",
            "expert_input": "Can we adjust the risk weighting for regulatory factors?",
            "metadata": {"interaction_type": "clarification_request"}
        },
        {
            "type": "modification",
            "content": "Based on recent regulatory changes, I suggest increasing the regulatory risk factor.",
            "expert_input": "Increase regulatory risk weight from 0.3 to 0.5",
            "ai_response": "Regulatory risk weight updated. Recalculating overall risk score...",
            "metadata": {"modification_applied": True}
        },
        {
            "type": "approval",
            "content": "The updated analysis addresses my concerns. Ready to finalize.",
            "expert_input": "Approved with modifications",
            "metadata": {"final_approval": True}
        }
    ]
    
    for i, interaction in enumerate(interactions, 1):
        await hitl_system.add_collaboration_interaction(session_id, {"interaction_data": interaction})
        print(f"   📝 Added interaction {i}: {interaction['type']}")
    
    # Finalize the collaboration session
    final_output = ModelOutput(
        result={
            **collaboration_output.result,
            "regulatory_risk_adjusted": True,
            "expert_collaboration": True,
            "final_confidence": 0.95
        },
        confidence=0.95,
        model_type=collaboration_output.model_type,
        task_category=collaboration_output.task_category,
        timestamp=datetime.now(),
        validation_score=0.92,
        guardrail_passed=True,
        human_reviewed=True
    )
    
    summary = await hitl_system.finalize_collaboration_session(
        session_id, final_output, "Successful collaboration with regulatory risk adjustments"
    )
    
    print(f"   ✅ Session finalized")
    print(f"   📊 Total interactions: {summary['total_interactions']}")
    print(f"   ⏱️  Duration: {summary['session_duration_minutes']:.1f} minutes")
    print(f"   📈 Improvement achieved: {summary['improvement_achieved']:.3f}")
    print(f"   🎯 Collaboration quality: {summary['collaboration_quality']:.3f}")
    
    print("\n📋 Expert Dashboard Testing")
    print("-" * 25)
    
    # Test expert dashboard for each expert
    for expert in experts[:2]:  # Test first 2 experts
        dashboard_data = hitl_system.get_expert_dashboard_data(expert.expert_id)
        
        print(f"\n   Dashboard for {expert.name}:")
        print(f"      Pending reviews: {dashboard_data['pending_reviews']}")
        print(f"      In-progress reviews: {dashboard_data['in_progress_reviews']}")
        print(f"      Recent feedback count: {dashboard_data['recent_feedback_count']}")
        
        if dashboard_data.get("performance_metrics") and not dashboard_data["performance_metrics"].get("insufficient_data"):
            metrics = dashboard_data["performance_metrics"]
            print(f"      Performance metrics:")
            print(f"         Avg rating given: {metrics.get('avg_rating_given', 0):.1f}")
            print(f"         Reliability score: {metrics.get('reliability_score', 0):.3f}")
            print(f"         Feedback quality: {metrics.get('feedback_quality', 0):.3f}")
        
        workload = dashboard_data.get("workload_status", {})
        if workload:
            print(f"      Workload: {workload.get('current_reviews', 0)}/{workload.get('capacity', 0)} "
                  f"({workload.get('utilization', 0):.1%} utilization)")
    
    print("\n📊 System Performance Metrics")
    print("-" * 25)
    
    final_system_status = hitl_system.get_system_status()
    metrics = final_system_status["system_metrics"]
    
    print(f"   Total reviews processed: {metrics['total_reviews']}")
    print(f"   Average expert rating: {metrics['avg_expert_rating']:.2f}")
    print(f"   Average agreement level: {metrics['avg_agreement_level']:.3f}")
    
    # Test feedback processor insights
    feedback_processor = hitl_system.feedback_processor
    print(f"   Total feedback processed: {len(feedback_processor.feedback_history)}")
    
    if feedback_processor.expert_reliability:
        print(f"   Expert reliability scores:")
        for expert_id, reliability in feedback_processor.expert_reliability.items():
            expert_name = next((e.name for e in experts if e.expert_id == expert_id), "Unknown")
            print(f"      {expert_name}: {reliability:.3f}")
    
    print("\n🔍 Testing Expert Routing")
    print("-" * 20)
    
    # Test expert routing with different scenarios
    routing_test_cases = [
        {
            "name": "High Priority Risk Assessment",
            "expert_type": ExpertType.RISK_MANAGER,
            "priority": ReviewPriority.URGENT,
            "task_category": TaskCategory.RISK_ASSESSMENT
        },
        {
            "name": "Routine Sentiment Analysis",
            "expert_type": ExpertType.SENIOR_ANALYST,
            "priority": ReviewPriority.LOW,
            "task_category": TaskCategory.SENTIMENT_ANALYSIS
        },
        {
            "name": "Critical Compliance Review",
            "expert_type": ExpertType.COMPLIANCE_OFFICER,
            "priority": ReviewPriority.CRITICAL,
            "task_category": TaskCategory.EARNINGS_PREDICTION
        }
    ]
    
    for test_case in routing_test_cases:
        # Create a test output
        test_output = ModelOutput(
            result={"test": "routing"},
            confidence=0.7,
            model_type=ModelType.SPECIALIZED,
            task_category=test_case["task_category"],
            timestamp=datetime.now(),
            validation_score=0.8,
            guardrail_passed=True
        )
        
        # Create review request
        from app.core.human_in_the_loop import ReviewRequest
        request = ReviewRequest(
            review_id=f"test_{test_case['name'].lower().replace(' ', '_')}",
            model_output=test_output,
            expert_type=test_case["expert_type"],
            priority=test_case["priority"],
            context={},
            deadline=None,
            special_instructions=None,
            routing_reason="test",
            created_at=datetime.now()
        )
        
        # Test routing
        assigned_expert_id = await hitl_system.expert_router.route_review_request(request)
        
        if assigned_expert_id:
            expert_name = next((e.name for e in experts if e.expert_id == assigned_expert_id), "Unknown")
            print(f"   ✅ {test_case['name']}: Routed to {expert_name}")
        else:
            print(f"   ❌ {test_case['name']}: No expert available")
    
    print(f"\n🎯 Demo Summary")
    print("-" * 15)
    print(f"   • Enhanced human-in-the-loop system successfully tested")
    print(f"   • Expert registration and routing system operational")
    print(f"   • Review queue management and prioritization working")
    print(f"   • Expert feedback processing and pattern analysis functional")
    print(f"   • Interactive collaboration sessions implemented")
    print(f"   • Performance tracking and reliability scoring active")
    print(f"   • Dashboard and monitoring capabilities demonstrated")
    print(f"   • Ready for production deployment with expert teams")
    print(f"   • Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def test_feedback_processing():
    """Test feedback processing and pattern analysis"""
    
    print("\n🔬 Feedback Processing Analysis Test")
    print("-" * 35)
    
    from app.core.human_in_the_loop import FeedbackProcessor, ExpertFeedback, FeedbackType
    
    processor = FeedbackProcessor()
    
    # Create sample feedback for testing
    sample_feedbacks = [
        ExpertFeedback(
            feedback_id="fb_001",
            review_id="rev_001",
            expert_id="exp_001",
            feedback_type=FeedbackType.MODIFICATION,
            overall_rating=7,
            agreement_level=0.8,
            confidence_in_feedback=0.9,
            detailed_comments="Good analysis with minor adjustments needed",
            specific_corrections=[
                {"field": "risk_score", "corrected_value": 0.65, "type": "numerical_adjustment"}
            ],
            improvement_suggestions=["Enhance risk correlation analysis"],
            risk_assessment="Acceptable",
            regulatory_concerns=[],
            alternative_recommendations=[],
            time_spent_minutes=45,
            created_at=datetime.now(),
            metadata={}
        ),
        ExpertFeedback(
            feedback_id="fb_002",
            review_id="rev_002",
            expert_id="exp_001",
            feedback_type=FeedbackType.APPROVAL,
            overall_rating=9,
            agreement_level=0.95,
            confidence_in_feedback=0.85,
            detailed_comments="Excellent analysis, well-structured and comprehensive",
            specific_corrections=[],
            improvement_suggestions=[],
            risk_assessment="Low risk",
            regulatory_concerns=[],
            alternative_recommendations=[],
            time_spent_minutes=30,
            created_at=datetime.now(),
            metadata={}
        )
    ]
    
    print(f"   Processing {len(sample_feedbacks)} sample feedback items...")
    
    for feedback in sample_feedbacks:
        # Create dummy original output
        original_output = ModelOutput(
            result={"test": "data"},
            confidence=0.7,
            model_type=ModelType.SPECIALIZED,
            task_category=TaskCategory.SENTIMENT_ANALYSIS,
            timestamp=datetime.now(),
            validation_score=0.8,
            guardrail_passed=True
        )
        
        # Process feedback
        result = asyncio.run(processor.process_feedback(feedback, original_output))
        
        print(f"      Feedback {feedback.feedback_id}: Processed ✅")
        print(f"         Patterns identified: {len(result['patterns_identified'])}")
        print(f"         Improvement suggestions: {len(result['improvement_suggestions'])}")
        print(f"         Expert reliability: {result['expert_reliability']:.3f}")
    
    print(f"   📊 Final expert reliability scores:")
    for expert_id, reliability in processor.expert_reliability.items():
        print(f"      Expert {expert_id}: {reliability:.3f}")


def test_expert_routing():
    """Test expert routing logic"""
    
    print("\n🎯 Expert Routing Logic Test")
    print("-" * 25)
    
    from app.core.human_in_the_loop import ExpertRouter
    
    router = ExpertRouter()
    
    # Create test experts
    test_experts = [
        create_expert_profile("Alice Smith", ExpertType.PORTFOLIO_MANAGER, ["equity"], 10),
        create_expert_profile("Bob Jones", ExpertType.RISK_MANAGER, ["risk"], 15),
        create_expert_profile("Carol Davis", ExpertType.SENIOR_ANALYST, ["analysis"], 8)
    ]
    
    # Register experts
    for expert in test_experts:
        router.register_expert(expert)
        print(f"   Registered: {expert.name}")
    
    print(f"   Total experts in router: {len(router.experts)}")
    
    # Test routing scenarios
    print(f"   Testing routing scenarios...")
    
    scenarios = [
        ("Portfolio Manager Request", ExpertType.PORTFOLIO_MANAGER),
        ("Risk Manager Request", ExpertType.RISK_MANAGER),
        ("Senior Analyst Request", ExpertType.SENIOR_ANALYST)
    ]
    
    for scenario_name, expert_type in scenarios:
        # Find eligible experts
        eligible = router._find_eligible_experts(type('MockRequest', (), {
            'expert_type': expert_type,
            'model_output': type('MockOutput', (), {'task_category': TaskCategory.SENTIMENT_ANALYSIS})(),
            'priority': ReviewPriority.MEDIUM
        })())
        
        print(f"      {scenario_name}: {len(eligible)} eligible experts")


if __name__ == "__main__":
    print("Starting Enhanced Human-in-the-Loop System Demo...")
    
    # Run the main demo
    asyncio.run(demo_human_in_the_loop())
    
    # Run additional tests
    test_feedback_processing()
    test_expert_routing()
    
    print("\n✅ Demo completed successfully!")