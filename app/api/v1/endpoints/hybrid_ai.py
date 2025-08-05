"""
Hybrid AI API Endpoints
BlackRock Aladdin-inspired AI system with specialized models and human-in-the-loop validation
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from app.core.hybrid_ai_engine import (
    HybridAIEngine, TaskCategory, ModelType, ModelOutput,
    EarningsCallAnalysisModel, ThematicInvestmentModel
)
from app.core.sentiment_analysis_model import FinancialSentimentModel, analyze_news_sentiment
from app.core.risk_prediction_model import RiskPredictionModel, quick_risk_assessment, portfolio_stress_test
from app.core.guardrail_engine import validate_model_output, check_output_safety

router = APIRouter()

# Initialize the hybrid AI engine
ai_engine = HybridAIEngine()


@router.post("/analyze/earnings")
async def analyze_earnings_call(
    earnings_data: Dict[str, Any],
    require_human_review: bool = False
):
    """Analyze earnings call transcript using specialized model"""
    try:
        transcript = earnings_data.get("transcript", "")
        if not transcript:
            raise HTTPException(status_code=400, detail="Earnings transcript is required")
        
        # Process with specialized earnings model
        output = await ai_engine.process_request(
            TaskCategory.EARNINGS_PREDICTION,
            transcript,
            require_human_review=require_human_review
        )
        
        return {
            "analysis": output.result,
            "confidence": output.confidence,
            "model_type": output.model_type.value,
            "validation_score": output.validation_score,
            "guardrails_passed": output.guardrail_passed,
            "human_reviewed": output.human_reviewed,
            "timestamp": output.timestamp.isoformat(),
            "metadata": output.metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Earnings analysis failed: {str(e)}")


@router.post("/analyze/thematic")
async def identify_thematic_opportunities(
    market_data: Dict[str, Any],
    require_human_review: bool = False
):
    """Identify thematic investment opportunities using specialized model"""
    try:
        # Process with specialized thematic model
        output = await ai_engine.process_request(
            TaskCategory.THEMATIC_IDENTIFICATION,
            market_data,
            require_human_review=require_human_review
        )
        
        return {
            "themes": output.result,
            "confidence": output.confidence,
            "model_type": output.model_type.value,
            "validation_score": output.validation_score,
            "guardrails_passed": output.guardrail_passed,
            "human_reviewed": output.human_reviewed,
            "timestamp": output.timestamp.isoformat(),
            "metadata": output.metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Thematic analysis failed: {str(e)}")


@router.post("/analyze/sentiment")
async def analyze_financial_sentiment(
    sentiment_data: Dict[str, Any],
    require_human_review: bool = False
):
    """Analyze financial sentiment using specialized model"""
    try:
        text = sentiment_data.get("text", "")
        symbols = sentiment_data.get("symbols", [])
        source = sentiment_data.get("source", "news_articles")
        
        if not text:
            raise HTTPException(status_code=400, detail="Text is required for sentiment analysis")
        
        # Process with specialized sentiment model
        output = await ai_engine.process_request(
            TaskCategory.SENTIMENT_ANALYSIS,
            {
                "text": text,
                "symbols": symbols,
                "source": source,
                "include_market_impact": True
            },
            require_human_review=require_human_review
        )
        
        return {
            "sentiment_analysis": output.result,
            "confidence": output.confidence,
            "model_type": output.model_type.value,
            "validation_score": output.validation_score,
            "guardrails_passed": output.guardrail_passed,
            "human_reviewed": output.human_reviewed,
            "timestamp": output.timestamp.isoformat(),
            "metadata": output.metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")


@router.get("/sentiment/trends/{symbol}")
async def get_sentiment_trends(
    symbol: str,
    days: int = 30
):
    """Get sentiment trends for a specific symbol"""
    try:
        sentiment_model = ai_engine.specialized_models.get("sentiment_analysis")
        if not sentiment_model:
            raise HTTPException(status_code=503, detail="Sentiment model not available")
        
        trends = sentiment_model.get_sentiment_trends(symbol.upper(), days)
        
        return {
            "symbol": symbol.upper(),
            "trends": trends,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sentiment trends: {str(e)}")


@router.post("/sentiment/batch")
async def batch_sentiment_analysis(
    batch_request: Dict[str, Any]
):
    """Process multiple sentiment analysis requests in batch"""
    try:
        text_data = batch_request.get("data", [])
        if not text_data:
            raise HTTPException(status_code=400, detail="Data array is required")
        
        from app.core.sentiment_analysis_model import batch_sentiment_analysis
        results = batch_sentiment_analysis(text_data)
        
        return {
            "batch_results": results,
            "total_processed": len(results),
            "successful": len([r for r in results if "error" not in r]),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch sentiment analysis failed: {str(e)}")


@router.post("/analyze/comprehensive")
async def comprehensive_investment_analysis(
    analysis_request: Dict[str, Any],
    require_human_review: bool = True
):
    """Comprehensive investment analysis using multiple specialized models"""
    try:
        results = {}
        
        # Earnings analysis if transcript provided
        if "earnings_transcript" in analysis_request:
            earnings_output = await ai_engine.process_request(
                TaskCategory.EARNINGS_PREDICTION,
                analysis_request["earnings_transcript"],
                require_human_review=False  # Will review comprehensive result
            )
            results["earnings_analysis"] = {
                "signal": earnings_output.result["investment_signal"],
                "strength": earnings_output.result["signal_strength"],
                "themes": earnings_output.result["key_themes"],
                "confidence": earnings_output.confidence
            }
        
        # Thematic analysis if market data provided
        if "market_data" in analysis_request:
            thematic_output = await ai_engine.process_request(
                TaskCategory.THEMATIC_IDENTIFICATION,
                analysis_request["market_data"],
                require_human_review=False
            )
            results["thematic_analysis"] = {
                "top_themes": thematic_output.result["top_themes"],
                "investment_vehicles": thematic_output.result["investment_vehicles"],
                "confidence": thematic_output.confidence
            }
        
        # Generate comprehensive recommendation
        comprehensive_result = _generate_comprehensive_recommendation(results)
        
        # Submit comprehensive analysis for human review if requested
        if require_human_review:
            review_id = ai_engine.human_loop_system.submit_for_review(
                ModelOutput(
                    result=comprehensive_result,
                    confidence=_calculate_overall_confidence(results),
                    model_type=ModelType.ENSEMBLE,
                    task_category=TaskCategory.MARKET_ANALYSIS,
                    timestamp=datetime.now(),
                    validation_score=0.85,
                    guardrail_passed=True
                ),
                expert_type="senior_analyst"
            )
            comprehensive_result["review_id"] = review_id
        
        return {
            "comprehensive_analysis": comprehensive_result,
            "component_analyses": results,
            "timestamp": datetime.now().isoformat(),
            "requires_human_review": require_human_review
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehensive analysis failed: {str(e)}")


@router.post("/human-review/submit")
async def submit_for_human_review(
    review_request: Dict[str, Any]
):
    """Submit analysis for human expert review"""
    try:
        output_data = review_request.get("output")
        expert_type = review_request.get("expert_type", "portfolio_manager")
        
        if not output_data:
            raise HTTPException(status_code=400, detail="Output data is required")
        
        # Create ModelOutput from request data
        model_output = ModelOutput(
            result=output_data.get("result", {}),
            confidence=output_data.get("confidence", 0.5),
            model_type=ModelType(output_data.get("model_type", "specialized")),
            task_category=TaskCategory(output_data.get("task_category", "market_analysis")),
            timestamp=datetime.now(),
            validation_score=output_data.get("validation_score", 0.7),
            guardrail_passed=output_data.get("guardrail_passed", True)
        )
        
        review_id = ai_engine.human_loop_system.submit_for_review(model_output, expert_type)
        
        return {
            "review_id": review_id,
            "expert_type": expert_type,
            "status": "submitted",
            "estimated_review_time": "2-4 hours",
            "priority": ai_engine.human_loop_system.pending_reviews[review_id]["priority"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Review submission failed: {str(e)}")


@router.post("/human-review/feedback/{review_id}")
async def provide_expert_feedback(
    review_id: str,
    feedback: Dict[str, Any]
):
    """Provide expert feedback on reviewed analysis"""
    try:
        success = ai_engine.human_loop_system.provide_expert_feedback(review_id, feedback)
        
        if not success:
            raise HTTPException(status_code=404, detail="Review ID not found")
        
        return {
            "review_id": review_id,
            "feedback_recorded": True,
            "reviewer": feedback.get("reviewer", "unknown"),
            "rating": feedback.get("rating"),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback recording failed: {str(e)}")


@router.get("/human-review/pending")
async def get_pending_reviews():
    """Get list of pending human reviews"""
    try:
        pending = ai_engine.human_loop_system.pending_reviews
        
        review_list = []
        for review_id, review_data in pending.items():
            if review_data["status"] == "pending":
                review_list.append({
                    "review_id": review_id,
                    "expert_type": review_data["expert_type"],
                    "submitted_at": review_data["submitted_at"].isoformat(),
                    "priority": review_data["priority"],
                    "task_category": review_data["output"].task_category.value,
                    "confidence": review_data["output"].confidence
                })
        
        return {
            "pending_reviews": review_list,
            "total_pending": len(review_list),
            "high_priority": len([r for r in review_list if r["priority"] == "high"]),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pending reviews: {str(e)}")


@router.get("/performance/report")
async def get_performance_report():
    """Get comprehensive AI system performance report"""
    try:
        report = ai_engine.get_performance_report()
        return report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance report failed: {str(e)}")


@router.get("/models/specialized")
async def get_specialized_models():
    """Get information about available specialized models"""
    try:
        models_info = {}
        
        for name, model in ai_engine.specialized_models.items():
            models_info[name] = {
                "name": model.name,
                "task_category": model.task_category.value,
                "training_data": {
                    "name": model.training_data.name if model.training_data else None,
                    "size": model.training_data.size if model.training_data else 0,
                    "quality_score": model.training_data.quality_score if model.training_data else 0,
                    "last_updated": model.training_data.last_updated.isoformat() if model.training_data else None
                },
                "performance_metrics": model.performance_metrics,
                "last_validation": model.last_validation.isoformat() if model.last_validation else None
            }
        
        return {
            "specialized_models": models_info,
            "total_models": len(models_info),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@router.post("/models/validate")
async def validate_model_output(
    validation_request: Dict[str, Any]
):
    """Validate model output using guardrail system"""
    try:
        output_data = validation_request.get("output")
        if not output_data:
            raise HTTPException(status_code=400, detail="Output data is required")
        
        # Create ModelOutput for validation
        model_output = ModelOutput(
            result=output_data.get("result", {}),
            confidence=output_data.get("confidence", 0.5),
            model_type=ModelType(output_data.get("model_type", "specialized")),
            task_category=TaskCategory(output_data.get("task_category", "market_analysis")),
            timestamp=datetime.now(),
            validation_score=output_data.get("validation_score", 0.7),
            guardrail_passed=True  # Will be updated by validation
        )
        
        # Run validation
        is_valid, issues = ai_engine.guardrail_system.validate_output(model_output)
        
        return {
            "validation_passed": is_valid,
            "issues": issues,
            "validation_score": model_output.validation_score,
            "recommendations": _get_validation_recommendations(issues),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.get("/collaboration/history")
async def get_collaboration_history():
    """Get human-AI collaboration history and insights"""
    try:
        history = ai_engine.human_loop_system.collaboration_history
        
        # Aggregate statistics
        total_collaborations = sum(len(task_history) for task_history in history.values())
        
        avg_agreement = 0
        avg_human_rating = 0
        if total_collaborations > 0:
            all_agreements = []
            all_ratings = []
            
            for task_history in history.values():
                for collab in task_history:
                    all_agreements.append(collab["agreement_level"])
                    all_ratings.append(collab["human_rating"])
            
            avg_agreement = sum(all_agreements) / len(all_agreements)
            avg_human_rating = sum(all_ratings) / len(all_ratings)
        
        return {
            "collaboration_summary": {
                "total_collaborations": total_collaborations,
                "avg_human_ai_agreement": avg_agreement,
                "avg_human_rating": avg_human_rating,
                "task_categories": list(history.keys())
            },
            "detailed_history": history,
            "insights": {
                "high_agreement_tasks": [task for task, hist in history.items() 
                                       if hist and sum(h["agreement_level"] for h in hist) / len(hist) > 0.8],
                "improvement_areas": _identify_improvement_areas(history)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collaboration history: {str(e)}")


def _generate_comprehensive_recommendation(results: Dict) -> Dict:
    """Generate comprehensive investment recommendation from multiple analyses"""
    recommendation = {
        "overall_signal": "neutral",
        "confidence": 0.5,
        "key_insights": [],
        "recommended_actions": [],
        "risk_considerations": [],
        "time_horizon": "medium_term"
    }
    
    # Combine earnings and thematic signals
    if "earnings_analysis" in results:
        earnings = results["earnings_analysis"]
        recommendation["key_insights"].append(f"Earnings signal: {earnings['signal']}")
        
        if earnings["signal"] == "bullish":
            recommendation["overall_signal"] = "bullish"
            recommendation["recommended_actions"].append("Consider increasing position size")
        elif earnings["signal"] == "bearish":
            recommendation["overall_signal"] = "bearish"
            recommendation["recommended_actions"].append("Consider reducing exposure")
    
    if "thematic_analysis" in results:
        thematic = results["thematic_analysis"]
        if thematic["top_themes"]:
            top_theme = thematic["top_themes"][0]["name"]
            recommendation["key_insights"].append(f"Strong thematic opportunity: {top_theme}")
            recommendation["recommended_actions"].append(f"Explore {top_theme} investment vehicles")
    
    return recommendation


def _calculate_overall_confidence(results: Dict) -> float:
    """Calculate overall confidence from multiple analysis results"""
    confidences = []
    
    if "earnings_analysis" in results:
        confidences.append(results["earnings_analysis"]["confidence"])
    
    if "thematic_analysis" in results:
        confidences.append(results["thematic_analysis"]["confidence"])
    
    return sum(confidences) / len(confidences) if confidences else 0.5


def _get_validation_recommendations(issues: List[str]) -> List[str]:
    """Get recommendations based on validation issues"""
    recommendations = []
    
    for issue in issues:
        if "structure" in issue.lower():
            recommendations.append("Review output format and ensure all required fields are present")
        elif "confidence" in issue.lower():
            recommendations.append("Recalibrate confidence scoring mechanism")
        elif "hallucination" in issue.lower():
            recommendations.append("Implement additional fact-checking against trusted sources")
        else:
            recommendations.append("Review model training data and validation logic")
    
    return recommendations


def _identify_improvement_areas(history: Dict) -> List[str]:
    """Identify areas for AI system improvement based on collaboration history"""
    improvement_areas = []
    
    for task_type, task_history in history.items():
        if not task_history:
            continue
        
        avg_rating = sum(h["human_rating"] for h in task_history) / len(task_history)
        avg_agreement = sum(h["agreement_level"] for h in task_history) / len(task_history)
        
        if avg_rating < 6:
            improvement_areas.append(f"Low human rating for {task_type} - review model training")
        
        if avg_agreement < 0.6:
            improvement_areas.append(f"Low human-AI agreement for {task_type} - calibrate outputs")
    
    return improvement_areas

@router.post("/analyze/risk")
async def predict_portfolio_risk(
    risk_data: Dict[str, Any],
    require_human_review: bool = True
):
    """Predict portfolio risk using specialized AI model"""
    try:
        symbols = risk_data.get("symbols", [])
        risk_types = risk_data.get("risk_types", ["market_risk"])
        time_horizon = risk_data.get("time_horizon", "monthly")
        portfolio_data = risk_data.get("portfolio_data", {})
        
        if not symbols:
            raise HTTPException(status_code=400, detail="Symbols are required for risk analysis")
        
        # Process with specialized risk prediction model
        output = await ai_engine.process_request(
            TaskCategory.RISK_ASSESSMENT,
            {
                "symbols": symbols,
                "risk_types": risk_types,
                "time_horizon": time_horizon,
                "portfolio_data": portfolio_data,
                "market_conditions": risk_data.get("market_conditions", {})
            },
            require_human_review=require_human_review
        )
        
        return {
            "risk_analysis": output.result,
            "confidence": output.confidence,
            "model_type": output.model_type.value,
            "validation_score": output.validation_score,
            "guardrails_passed": output.guardrail_passed,
            "human_reviewed": output.human_reviewed,
            "timestamp": output.timestamp.isoformat(),
            "metadata": output.metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk analysis failed: {str(e)}")


@router.post("/risk/stress-test")
async def perform_stress_test(
    stress_test_data: Dict[str, Any]
):
    """Perform comprehensive portfolio stress testing"""
    try:
        symbols = stress_test_data.get("symbols", [])
        portfolio_data = stress_test_data.get("portfolio_data", {})
        
        if not symbols:
            raise HTTPException(status_code=400, detail="Symbols are required for stress testing")
        
        # Use utility function for stress testing
        stress_results = portfolio_stress_test(symbols, portfolio_data)
        
        return {
            "stress_test_results": stress_results,
            "symbols_tested": symbols,
            "portfolio_value": portfolio_data.get("total_value", 0),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stress testing failed: {str(e)}")


@router.get("/risk/quick-assessment/{symbols}")
async def quick_risk_check(
    symbols: str,
    portfolio_value: float = 1000000
):
    """Quick risk assessment for given symbols"""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        
        # Use utility function for quick assessment
        risk_assessment = quick_risk_assessment(symbol_list, portfolio_value)
        
        return {
            "quick_assessment": risk_assessment,
            "symbols": symbol_list,
            "portfolio_value": portfolio_value,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick risk assessment failed: {str(e)}")


@router.get("/risk/scenarios/{symbols}")
async def get_risk_scenarios(
    symbols: str,
    time_horizon: str = "monthly"
):
    """Get risk scenarios for specific symbols"""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        
        risk_model = ai_engine.specialized_models.get("risk_prediction")
        if not risk_model:
            raise HTTPException(status_code=503, detail="Risk prediction model not available")
        
        # Generate risk scenarios
        input_data = {
            "symbols": symbol_list,
            "risk_types": ["market_risk", "systemic_risk", "tail_risk"],
            "time_horizon": time_horizon,
            "portfolio_data": {"total_value": 1000000}
        }
        
        result = risk_model.predict(input_data)
        
        return {
            "risk_scenarios": result.result.get("risk_scenarios", []),
            "early_warnings": result.result.get("early_warning_signals", []),
            "symbols": symbol_list,
            "time_horizon": time_horizon,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk scenarios failed: {str(e)}")

@router.post("/guardrails/validate")
async def validate_output_with_guardrails(
    validation_request: Dict[str, Any]
):
    """Validate model output using comprehensive guardrail system"""
    try:
        output_data = validation_request.get("output")
        context = validation_request.get("context", {})
        
        if not output_data:
            raise HTTPException(status_code=400, detail="Output data is required")
        
        # Create ModelOutput for validation
        model_output = ModelOutput(
            result=output_data.get("result", {}),
            confidence=output_data.get("confidence", 0.5),
            model_type=ModelType(output_data.get("model_type", "specialized")),
            task_category=TaskCategory(output_data.get("task_category", "market_analysis")),
            timestamp=datetime.now(),
            validation_score=output_data.get("validation_score", 0.7),
            guardrail_passed=True  # Will be updated by validation
        )
        
        # Run comprehensive guardrail validation
        guardrail_result = await ai_engine.guardrail_system.validate_output(model_output, context)
        
        return {
            "validation_passed": guardrail_result.passed,
            "overall_score": guardrail_result.overall_score,
            "violations": [
                {
                    "type": v.violation_type.value,
                    "severity": v.severity.value,
                    "description": v.description,
                    "detected_value": str(v.detected_value),
                    "expected_value": str(v.expected_value),
                    "confidence": v.confidence,
                    "recommendation": v.recommendation,
                    "timestamp": v.timestamp.isoformat()
                }
                for v in guardrail_result.violations
            ],
            "warnings": guardrail_result.warnings,
            "recommendations": guardrail_result.recommendations,
            "processing_time": guardrail_result.processing_time,
            "metadata": guardrail_result.metadata,
            "timestamp": guardrail_result.timestamp.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Guardrail validation failed: {str(e)}")


@router.get("/guardrails/performance")
async def get_guardrail_performance():
    """Get guardrail system performance report"""
    try:
        performance_report = ai_engine.guardrail_system.get_performance_report()
        
        return {
            "performance_report": performance_report,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance report: {str(e)}")


@router.get("/guardrails/violations")
async def get_violation_summary(
    days: int = 7
):
    """Get violation summary for specified time period"""
    try:
        violation_summary = ai_engine.guardrail_system.get_violation_summary(days)
        
        return {
            "violation_summary": violation_summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get violation summary: {str(e)}")


@router.post("/guardrails/configure")
async def configure_guardrails(
    config_request: Dict[str, Any]
):
    """Configure guardrail settings"""
    try:
        from app.core.guardrail_engine import GuardrailType
        
        configurations = config_request.get("configurations", [])
        results = []
        
        for config in configurations:
            guardrail_type_str = config.get("guardrail_type")
            enabled = config.get("enabled")
            threshold = config.get("threshold")
            
            try:
                guardrail_type = GuardrailType(guardrail_type_str)
                ai_engine.guardrail_system.configure_guardrail(
                    guardrail_type, enabled, threshold
                )
                results.append({
                    "guardrail_type": guardrail_type_str,
                    "status": "configured",
                    "enabled": enabled,
                    "threshold": threshold
                })
            except ValueError:
                results.append({
                    "guardrail_type": guardrail_type_str,
                    "status": "error",
                    "message": f"Invalid guardrail type: {guardrail_type_str}"
                })
        
        return {
            "configuration_results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Guardrail configuration failed: {str(e)}")


@router.post("/guardrails/batch-validate")
async def batch_validate_outputs(
    batch_request: Dict[str, Any]
):
    """Validate multiple outputs in batch"""
    try:
        outputs_data = batch_request.get("outputs", [])
        context = batch_request.get("context", {})
        
        if not outputs_data:
            raise HTTPException(status_code=400, detail="Outputs array is required")
        
        # Convert to ModelOutput objects
        model_outputs = []
        for output_data in outputs_data:
            model_output = ModelOutput(
                result=output_data.get("result", {}),
                confidence=output_data.get("confidence", 0.5),
                model_type=ModelType(output_data.get("model_type", "specialized")),
                task_category=TaskCategory(output_data.get("task_category", "market_analysis")),
                timestamp=datetime.now(),
                validation_score=output_data.get("validation_score", 0.7),
                guardrail_passed=True
            )
            model_outputs.append(model_output)
        
        # Run batch validation
        results = await ai_engine.guardrail_system.batch_validate(model_outputs, context)
        
        # Format results
        formatted_results = []
        for i, result in enumerate(results):
            formatted_results.append({
                "index": i,
                "validation_passed": result.passed,
                "overall_score": result.overall_score,
                "violation_count": len(result.violations),
                "processing_time": result.processing_time,
                "warnings": result.warnings,
                "recommendations": result.recommendations
            })
        
        return {
            "batch_results": formatted_results,
            "total_processed": len(formatted_results),
            "passed_count": len([r for r in formatted_results if r["validation_passed"]]),
            "failed_count": len([r for r in formatted_results if not r["validation_passed"]]),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch validation failed: {str(e)}")


@router.post("/guardrails/safety-check")
async def quick_safety_check(
    safety_request: Dict[str, Any]
):
    """Quick safety check for model output"""
    try:
        output_data = safety_request.get("output")
        
        if not output_data:
            raise HTTPException(status_code=400, detail="Output data is required")
        
        # Create ModelOutput for safety check
        model_output = ModelOutput(
            result=output_data.get("result", {}),
            confidence=output_data.get("confidence", 0.5),
            model_type=ModelType(output_data.get("model_type", "specialized")),
            task_category=TaskCategory(output_data.get("task_category", "market_analysis")),
            timestamp=datetime.now(),
            validation_score=output_data.get("validation_score", 0.7),
            guardrail_passed=True
        )
        
        # Run quick safety check
        is_safe = check_output_safety(model_output)
        
        return {
            "safety_check_passed": is_safe,
            "message": "Output passed basic safety checks" if is_safe else "Output failed safety checks",
            "recommendation": "Output is safe to use" if is_safe else "Review output before use",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Safety check failed: {str(e)}")

# Enhanced Human-in-the-Loop Endpoints

@router.post("/human-review/submit-enhanced")
async def submit_for_enhanced_review(
    review_request: Dict[str, Any]
):
    """Submit analysis for enhanced human expert review"""
    try:
        from app.core.human_in_the_loop import ExpertType, ReviewPriority
        
        output_data = review_request.get("output")
        expert_type = ExpertType(review_request.get("expert_type", "portfolio_manager"))
        priority = ReviewPriority(review_request.get("priority", "medium"))
        context = review_request.get("context", {})
        special_instructions = review_request.get("special_instructions")
        
        if not output_data:
            raise HTTPException(status_code=400, detail="Output data is required")
        
        # Create ModelOutput for review
        model_output = ModelOutput(
            result=output_data.get("result", {}),
            confidence=output_data.get("confidence", 0.5),
            model_type=ModelType(output_data.get("model_type", "specialized")),
            task_category=TaskCategory(output_data.get("task_category", "market_analysis")),
            timestamp=datetime.now(),
            validation_score=output_data.get("validation_score", 0.7),
            guardrail_passed=output_data.get("guardrail_passed", True)
        )
        
        # Submit for enhanced review
        review_id = await ai_engine.human_loop_system.submit_for_review(
            model_output, expert_type, priority, context, special_instructions
        )
        
        return {
            "review_id": review_id,
            "expert_type": expert_type.value,
            "priority": priority.value,
            "status": "submitted",
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced review submission failed: {str(e)}")


@router.post("/human-review/provide-feedback/{review_id}")
async def provide_enhanced_expert_feedback(
    review_id: str,
    feedback_request: Dict[str, Any]
):
    """Provide enhanced expert feedback on reviewed analysis"""
    try:
        expert_id = feedback_request.get("expert_id")
        feedback_data = feedback_request.get("feedback_data", {})
        
        if not expert_id:
            raise HTTPException(status_code=400, detail="Expert ID is required")
        
        # Process enhanced feedback
        result = await ai_engine.human_loop_system.provide_expert_feedback(
            review_id, expert_id, feedback_data
        )
        
        return {
            "feedback_processed": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced feedback processing failed: {str(e)}")


@router.post("/collaboration/start")
async def start_collaboration_session(
    collaboration_request: Dict[str, Any]
):
    """Start an interactive collaboration session between expert and AI"""
    try:
        expert_id = collaboration_request.get("expert_id")
        output_data = collaboration_request.get("output")
        
        if not expert_id or not output_data:
            raise HTTPException(status_code=400, detail="Expert ID and output data are required")
        
        # Create ModelOutput for collaboration
        model_output = ModelOutput(
            result=output_data.get("result", {}),
            confidence=output_data.get("confidence", 0.5),
            model_type=ModelType(output_data.get("model_type", "specialized")),
            task_category=TaskCategory(output_data.get("task_category", "market_analysis")),
            timestamp=datetime.now(),
            validation_score=output_data.get("validation_score", 0.7),
            guardrail_passed=output_data.get("guardrail_passed", True)
        )
        
        # Start collaboration session
        session_id = await ai_engine.human_loop_system.start_collaboration_session(
            expert_id, model_output
        )
        
        return {
            "session_id": session_id,
            "expert_id": expert_id,
            "status": "active",
            "started_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Collaboration session start failed: {str(e)}")


@router.post("/collaboration/{session_id}/interact")
async def add_collaboration_interaction(
    session_id: str,
    interaction_request: Dict[str, Any]
):
    """Add an interaction to a collaboration session"""
    try:
        interaction_data = interaction_request.get("interaction_data", {})
        
        # Add interaction to session
        result = await ai_engine.human_loop_system.add_collaboration_interaction(
            session_id, interaction_data
        )
        
        return {
            "interaction_added": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Collaboration interaction failed: {str(e)}")


@router.post("/collaboration/{session_id}/finalize")
async def finalize_collaboration_session(
    session_id: str,
    finalization_request: Dict[str, Any]
):
    """Finalize a collaboration session"""
    try:
        final_output_data = finalization_request.get("final_output")
        session_notes = finalization_request.get("session_notes")
        
        if not final_output_data:
            raise HTTPException(status_code=400, detail="Final output data is required")
        
        # Create final ModelOutput
        final_output = ModelOutput(
            result=final_output_data.get("result", {}),
            confidence=final_output_data.get("confidence", 0.5),
            model_type=ModelType(final_output_data.get("model_type", "specialized")),
            task_category=TaskCategory(final_output_data.get("task_category", "market_analysis")),
            timestamp=datetime.now(),
            validation_score=final_output_data.get("validation_score", 0.7),
            guardrail_passed=final_output_data.get("guardrail_passed", True),
            human_reviewed=True
        )
        
        # Finalize session
        summary = await ai_engine.human_loop_system.finalize_collaboration_session(
            session_id, final_output, session_notes
        )
        
        return {
            "session_finalized": True,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Collaboration finalization failed: {str(e)}")


@router.get("/experts/dashboard/{expert_id}")
async def get_expert_dashboard(
    expert_id: str
):
    """Get dashboard data for a specific expert"""
    try:
        dashboard_data = ai_engine.human_loop_system.get_expert_dashboard_data(expert_id)
        
        return {
            "dashboard_data": dashboard_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Expert dashboard failed: {str(e)}")


@router.post("/experts/register")
async def register_expert(
    expert_data: Dict[str, Any]
):
    """Register a new expert in the system"""
    try:
        from app.core.human_in_the_loop import create_expert_profile, ExpertType
        
        name = expert_data.get("name")
        expert_type = ExpertType(expert_data.get("expert_type", "portfolio_manager"))
        specializations = expert_data.get("specializations", [])
        experience_years = expert_data.get("experience_years", 5)
        
        if not name:
            raise HTTPException(status_code=400, detail="Expert name is required")
        
        # Create expert profile
        expert_profile = create_expert_profile(name, expert_type, specializations, experience_years)
        
        # Register expert
        ai_engine.human_loop_system.expert_router.register_expert(expert_profile)
        
        return {
            "expert_registered": True,
            "expert_id": expert_profile.expert_id,
            "expert_profile": {
                "name": expert_profile.name,
                "expert_type": expert_profile.expert_type.value,
                "specializations": expert_profile.specializations,
                "experience_years": expert_profile.experience_years
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Expert registration failed: {str(e)}")


@router.get("/human-loop/system-status")
async def get_human_loop_system_status():
    """Get overall human-in-the-loop system status"""
    try:
        system_status = ai_engine.human_loop_system.get_system_status()
        
        return {
            "system_status": system_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System status failed: {str(e)}")


@router.get("/human-loop/queue-status")
async def get_review_queue_status():
    """Get current review queue status"""
    try:
        queue_status = ai_engine.human_loop_system.review_queue.get_queue_status()
        
        return {
            "queue_status": queue_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Queue status failed: {str(e)}")


@router.get("/human-loop/performance-metrics")
async def get_human_loop_performance_metrics():
    """Get human-in-the-loop system performance metrics"""
    try:
        system_metrics = ai_engine.human_loop_system.system_metrics
        
        # Get additional metrics
        feedback_processor = ai_engine.human_loop_system.feedback_processor
        expert_reliability = dict(feedback_processor.expert_reliability)
        
        return {
            "system_metrics": system_metrics,
            "expert_reliability_scores": expert_reliability,
            "total_feedback_processed": len(feedback_processor.feedback_history),
            "improvement_patterns": dict(feedback_processor.improvement_patterns),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance metrics failed: {str(e)}")

# Advanced Orchestration Endpoints

@router.post("/orchestration/process-query")
async def process_query_with_advanced_orchestration(
    orchestration_request: Dict[str, Any]
):
    """Process investment query using advanced orchestration"""
    try:
        from app.core.ai_orchestrator import InvestmentQuery, QueryType
        
        # Extract query data
        query_data = orchestration_request.get("query", {})
        user_preferences = orchestration_request.get("user_preferences", {})
        performance_requirements = orchestration_request.get("performance_requirements", {})
        
        # Create investment query
        query = InvestmentQuery(
            query_text=query_data.get("query_text", ""),
            query_type=QueryType(query_data.get("query_type", "general_analysis")),
            symbols=query_data.get("symbols", []),
            time_horizon=query_data.get("time_horizon", "medium_term"),
            risk_tolerance=query_data.get("risk_tolerance", "moderate"),
            user_context=query_data.get("user_context", {}),
            timestamp=datetime.now(),
            metadata=query_data.get("metadata", {})
        )
        
        # Process with advanced orchestrator
        if ai_engine.advanced_orchestrator:
            result = await ai_engine.advanced_orchestrator.process_query(
                query, user_preferences, performance_requirements
            )
            
            return {
                "orchestration_result": result,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=503, detail="Advanced orchestrator not available")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced orchestration failed: {str(e)}")


@router.get("/orchestration/analytics")
async def get_orchestration_analytics():
    """Get orchestration analytics and performance insights"""
    try:
        if ai_engine.advanced_orchestrator:
            analytics = ai_engine.advanced_orchestrator.get_orchestration_analytics()
            
            return {
                "analytics": analytics,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=503, detail="Advanced orchestrator not available")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics retrieval failed: {str(e)}")


@router.get("/orchestration/model-capabilities")
async def get_model_capabilities():
    """Get registered model capabilities and performance profiles"""
    try:
        if ai_engine.advanced_orchestrator:
            capability_analyzer = ai_engine.advanced_orchestrator.intelligent_router.capability_analyzer
            
            # Get model capabilities
            model_capabilities = {}
            for model_id, capabilities in capability_analyzer.model_capabilities.items():
                model_capabilities[model_id] = [cap.value for cap in capabilities]
            
            # Get performance profiles
            performance_profiles = {}
            for profile_key, profile in capability_analyzer.performance_profiles.items():
                performance_profiles[profile_key] = {
                    "model_id": profile.model_id,
                    "task_category": profile.task_category.value,
                    "avg_accuracy": profile.avg_accuracy,
                    "avg_confidence": profile.avg_confidence,
                    "avg_processing_time": profile.avg_processing_time,
                    "reliability_score": profile.reliability_score,
                    "specialization_strength": profile.specialization_strength,
                    "recent_performance_trend": profile.recent_performance_trend,
                    "sample_size": profile.sample_size,
                    "last_updated": profile.last_updated.isoformat()
                }
            
            return {
                "model_capabilities": model_capabilities,
                "performance_profiles": performance_profiles,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=503, detail="Advanced orchestrator not available")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model capabilities retrieval failed: {str(e)}")


@router.post("/orchestration/test-routing")
async def test_routing_strategies(
    routing_test: Dict[str, Any]
):
    """Test different routing strategies for a given query"""
    try:
        from app.core.ai_orchestrator import InvestmentQuery, QueryType
        from app.core.advanced_orchestration import RoutingContext
        
        # Extract test data
        query_data = routing_test.get("query", {})
        test_strategies = routing_test.get("strategies", ["single_model", "parallel_ensemble"])
        
        # Create investment query
        query = InvestmentQuery(
            query_text=query_data.get("query_text", "Test query"),
            query_type=QueryType(query_data.get("query_type", "general_analysis")),
            symbols=query_data.get("symbols", ["AAPL"]),
            time_horizon=query_data.get("time_horizon", "medium_term"),
            risk_tolerance=query_data.get("risk_tolerance", "moderate"),
            user_context=query_data.get("user_context", {}),
            timestamp=datetime.now(),
            metadata=query_data.get("metadata", {})
        )
        
        if ai_engine.advanced_orchestrator:
            # Create routing context
            context = RoutingContext(
                query=query,
                user_preferences={},
                performance_requirements={},
                available_models=list(ai_engine.advanced_orchestrator.available_models.keys()),
                system_load={},
                deadline_constraints=None,
                quality_requirements={"accuracy": 0.8}
            )
            
            # Test different routing strategies
            routing_results = {}
            
            for strategy in test_strategies:
                try:
                    # Create orchestration plan
                    plan = await ai_engine.advanced_orchestrator.intelligent_router.create_orchestration_plan(context)
                    
                    routing_results[strategy] = {
                        "routing_decision": plan.routing_decision.value,
                        "selected_models": plan.selected_models,
                        "synthesis_strategy": plan.synthesis_strategy,
                        "estimated_processing_time": plan.estimated_processing_time,
                        "expected_quality_score": plan.expected_quality_score,
                        "confidence_weighting": plan.confidence_weighting
                    }
                    
                except Exception as e:
                    routing_results[strategy] = {"error": str(e)}
            
            return {
                "routing_test_results": routing_results,
                "query_analyzed": {
                    "query_text": query.query_text,
                    "query_type": query.query_type.value,
                    "symbols": query.symbols
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=503, detail="Advanced orchestrator not available")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Routing strategy testing failed: {str(e)}")


@router.get("/orchestration/performance-monitor/{metric_name}")
async def get_performance_metric(
    metric_name: str,
    days: int = 7
):
    """Get performance metric summary"""
    try:
        if ai_engine.advanced_orchestrator:
            monitor = ai_engine.advanced_orchestrator.performance_monitor
            metric_summary = monitor.get_metric_summary(metric_name, days)
            
            return {
                "metric_summary": metric_summary,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=503, detail="Advanced orchestrator not available")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance metric retrieval failed: {str(e)}")


@router.post("/orchestration/register-model")
async def register_model_with_orchestrator(
    registration_request: Dict[str, Any]
):
    """Register a new model with the advanced orchestrator"""
    try:
        from app.core.advanced_orchestration import ModelCapability
        
        model_id = registration_request.get("model_id")
        capabilities = registration_request.get("capabilities", [])
        
        if not model_id:
            raise HTTPException(status_code=400, detail="Model ID is required")
        
        # Convert capability strings to enums
        model_capabilities = []
        for cap_str in capabilities:
            try:
                capability = ModelCapability(cap_str)
                model_capabilities.append(capability)
            except ValueError:
                logger.warning(f"Unknown capability: {cap_str}")
        
        if ai_engine.advanced_orchestrator:
            # Get model instance (simplified - in practice you'd have a model registry)
            model_instance = ai_engine.specialized_models.get(model_id)
            
            if model_instance:
                ai_engine.advanced_orchestrator.register_model(
                    model_id, model_instance, model_capabilities
                )
                
                return {
                    "model_registered": True,
                    "model_id": model_id,
                    "capabilities": [cap.value for cap in model_capabilities],
                    "timestamp": datetime.now().isoformat()
                }
            else:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        else:
            raise HTTPException(status_code=503, detail="Advanced orchestrator not available")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model registration failed: {str(e)}")