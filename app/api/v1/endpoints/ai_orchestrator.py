"""
AI Orchestrator API Endpoints
BlackRock Aladdin-inspired hybrid AI system endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from app.core.ai_orchestrator import (
    AIOrchestrator, InvestmentQuery, QueryType, RoutingStrategy,
    FoundationModel, SpecializedModel
)
from app.core.foundation_models import create_foundation_models

router = APIRouter()

# Initialize the AI orchestrator
orchestrator = AIOrchestrator()

# Initialize foundation models (demo configuration)
foundation_config = {
    "gpt4": {"enabled": True, "api_key": "demo_key"},
    "gemini": {"enabled": True, "api_key": "demo_key"},
    "claude": {"enabled": True, "api_key": "demo_key"}
}

foundation_models = create_foundation_models(foundation_config)
for model in foundation_models.values():
    orchestrator.register_foundation_model(model)

# Initialize specialized models (demo versions)
from app.core.ai_orchestrator import SpecializedModel

earnings_model = SpecializedModel(
    model_id="earnings_analyzer_v1",
    specialization=QueryType.EARNINGS_ANALYSIS,
    model_path="/models/earnings_analyzer"
)
earnings_model.training_data_size = 400000

thematic_model = SpecializedModel(
    model_id="thematic_identifier_v1", 
    specialization=QueryType.THEMATIC_IDENTIFICATION,
    model_path="/models/thematic_identifier"
)
thematic_model.training_data_size = 150000

sentiment_model = SpecializedModel(
    model_id="sentiment_analyzer_v1",
    specialization=QueryType.SENTIMENT_ANALYSIS,
    model_path="/models/sentiment_analyzer"
)
sentiment_model.training_data_size = 200000

orchestrator.register_specialized_model(earnings_model)
orchestrator.register_specialized_model(thematic_model)
orchestrator.register_specialized_model(sentiment_model)


@router.post("/analyze")
async def analyze_investment_query(
    query_data: Dict[str, Any],
    routing_strategy: str = "specialization_first"
):
    """Main endpoint for investment analysis using hybrid AI"""
    try:
        # Parse routing strategy
        try:
            strategy = RoutingStrategy(routing_strategy)
        except ValueError:
            strategy = RoutingStrategy.SPECIALIZATION_FIRST
        
        # Create investment query
        query = InvestmentQuery(
            query_text=query_data.get("query", ""),
            query_type=QueryType(query_data.get("query_type", "general_analysis")),
            symbols=query_data.get("symbols", []),
            time_horizon=query_data.get("time_horizon", "medium_term"),
            risk_tolerance=query_data.get("risk_tolerance", "moderate"),
            user_context=query_data.get("user_context", {}),
            timestamp=datetime.now(),
            metadata=query_data.get("metadata", {})
        )
        
        # Process query through orchestrator
        result = await orchestrator.process_request(query, strategy)
        
        return {
            "query_id": result.query_id,
            "analysis": result.synthesized_response,
            "confidence": result.confidence,
            "models_used": [
                {
                    "model_id": r.model_id,
                    "model_type": r.model_type.value,
                    "confidence": r.confidence,
                    "processing_time": r.processing_time
                }
                for r in result.model_responses
            ],
            "routing_strategy": result.routing_strategy.value,
            "human_review_required": result.human_review_required,
            "processing_time": result.processing_time,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze/earnings")
async def analyze_earnings(
    earnings_data: Dict[str, Any]
):
    """Specialized earnings analysis endpoint"""
    try:
        query = InvestmentQuery(
            query_text=earnings_data.get("transcript", ""),
            query_type=QueryType.EARNINGS_ANALYSIS,
            symbols=earnings_data.get("symbols", []),
            time_horizon=earnings_data.get("time_horizon", "short_term"),
            risk_tolerance=earnings_data.get("risk_tolerance", "moderate"),
            user_context=earnings_data.get("context", {}),
            timestamp=datetime.now()
        )
        
        result = await orchestrator.process_request(query, RoutingStrategy.SPECIALIZATION_FIRST)
        
        return {
            "earnings_analysis": result.synthesized_response,
            "confidence": result.confidence,
            "specialized_model_used": any(
                r.model_type.value == "specialized" for r in result.model_responses
            ),
            "human_review_required": result.human_review_required,
            "processing_time": result.processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Earnings analysis failed: {str(e)}")


@router.post("/analyze/thematic")
async def identify_themes(
    market_data: Dict[str, Any]
):
    """Thematic investment opportunity identification"""
    try:
        query = InvestmentQuery(
            query_text="Identify thematic investment opportunities",
            query_type=QueryType.THEMATIC_IDENTIFICATION,
            symbols=market_data.get("symbols", []),
            time_horizon=market_data.get("time_horizon", "long_term"),
            risk_tolerance=market_data.get("risk_tolerance", "moderate"),
            user_context=market_data,
            timestamp=datetime.now()
        )
        
        result = await orchestrator.process_request(query, RoutingStrategy.SPECIALIZATION_FIRST)
        
        return {
            "thematic_analysis": result.synthesized_response,
            "confidence": result.confidence,
            "themes_identified": result.synthesized_response.get("primary_prediction", {}).get("top_themes", []),
            "processing_time": result.processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Thematic analysis failed: {str(e)}")


@router.post("/analyze/sentiment")
async def analyze_sentiment(
    sentiment_data: Dict[str, Any]
):
    """Financial sentiment analysis"""
    try:
        query = InvestmentQuery(
            query_text=sentiment_data.get("text", ""),
            query_type=QueryType.SENTIMENT_ANALYSIS,
            symbols=sentiment_data.get("symbols", []),
            time_horizon="short_term",
            risk_tolerance="moderate",
            user_context=sentiment_data,
            timestamp=datetime.now()
        )
        
        result = await orchestrator.process_request(query, RoutingStrategy.SPECIALIZATION_FIRST)
        
        return {
            "sentiment_analysis": result.synthesized_response,
            "confidence": result.confidence,
            "sentiment_score": result.synthesized_response.get("primary_prediction", {}).get("overall_sentiment", 0.5),
            "processing_time": result.processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")


@router.get("/system/status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        status = await orchestrator.get_system_status()
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.get("/models/foundation")
async def get_foundation_models():
    """Get information about available foundation models"""
    try:
        models_info = {}
        
        for model_id, model in orchestrator.foundation_models.items():
            health = await model.health_check()
            models_info[model_id] = {
                "model_id": model.model_id,
                "api_endpoint": model.api_endpoint,
                "healthy": health,
                "capabilities": [c.value for c in model.get_capabilities()],
                "model_type": "foundation"
            }
        
        return {
            "foundation_models": models_info,
            "total_models": len(models_info),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get foundation models: {str(e)}")


@router.get("/models/specialized")
async def get_specialized_models():
    """Get information about available specialized models"""
    try:
        models_info = {}
        
        for model_id, model in orchestrator.specialized_models.items():
            health = await model.health_check()
            models_info[model_id] = {
                "model_id": model.model_id,
                "specialization": model.specialization.value,
                "model_path": model.model_path,
                "training_data_size": model.training_data_size,
                "healthy": health,
                "accuracy_metrics": model.accuracy_metrics,
                "model_type": "specialized"
            }
        
        return {
            "specialized_models": models_info,
            "total_models": len(models_info),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get specialized models: {str(e)}")


@router.post("/routing/test")
async def test_routing_strategies(
    test_query: Dict[str, Any]
):
    """Test different routing strategies on the same query"""
    try:
        query = InvestmentQuery(
            query_text=test_query.get("query", "Test query"),
            query_type=QueryType(test_query.get("query_type", "general_analysis")),
            symbols=test_query.get("symbols", ["AAPL"]),
            time_horizon="medium_term",
            risk_tolerance="moderate",
            user_context={},
            timestamp=datetime.now()
        )
        
        strategies = [
            RoutingStrategy.SPECIALIZATION_FIRST,
            RoutingStrategy.PARALLEL_ENSEMBLE,
            RoutingStrategy.CONFIDENCE_BASED,
            RoutingStrategy.FOUNDATION_FALLBACK
        ]
        
        results = {}
        
        for strategy in strategies:
            try:
                result = await orchestrator.process_request(query, strategy)
                results[strategy.value] = {
                    "confidence": result.confidence,
                    "models_used": len(result.model_responses),
                    "processing_time": result.processing_time,
                    "human_review_required": result.human_review_required
                }
            except Exception as e:
                results[strategy.value] = {"error": str(e)}
        
        return {
            "routing_comparison": results,
            "query_tested": test_query,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Routing test failed: {str(e)}")


@router.get("/performance/metrics")
async def get_performance_metrics():
    """Get system performance metrics"""
    try:
        return {
            "performance_metrics": orchestrator.performance_metrics,
            "total_queries_processed": len(orchestrator.performance_metrics),
            "system_uptime": "operational",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance metrics failed: {str(e)}")


@router.post("/models/register")
async def register_new_model(
    model_config: Dict[str, Any]
):
    """Register a new model with the orchestrator"""
    try:
        model_type = model_config.get("model_type", "specialized")
        
        if model_type == "specialized":
            model = SpecializedModel(
                model_id=model_config["model_id"],
                specialization=QueryType(model_config["specialization"]),
                model_path=model_config["model_path"]
            )
            model.training_data_size = model_config.get("training_data_size", 0)
            orchestrator.register_specialized_model(model)
            
        else:
            return {"error": "Foundation model registration not supported via API"}
        
        return {
            "message": f"Model {model_config['model_id']} registered successfully",
            "model_type": model_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model registration failed: {str(e)}")