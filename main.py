"""
Personal Investment Management Platform (Personal Aladdin)
Institutional-grade investment management for personal investors
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api.v1.endpoints import analytics, indian_market

app = FastAPI(
    title="Personal Aladdin - Indian Market Edition",
    description="Institutional-grade investment management platform with AI-powered Indian stock recommendations",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])

# Import and include live predictions router
from app.api.v1.endpoints import live_predictions, hybrid_ai, ai_orchestrator
app.include_router(live_predictions.router, prefix="/api/v1/live", tags=["live-predictions"])
app.include_router(hybrid_ai.router, prefix="/api/v1/hybrid-ai", tags=["hybrid-ai"])
app.include_router(ai_orchestrator.router, prefix="/api/v1/ai", tags=["ai-orchestrator"])
app.include_router(indian_market.router, prefix="/api/v1/indian-market", tags=["indian-market"])

@app.get("/")
async def root():
    return {
        "message": "Personal Aladdin - Investment Management Platform",
        "version": "1.0.0",
        "features": [
            "Multi-asset portfolio management",
            "Real-time risk analytics", 
            "Quantitative strategies",
            "Alternative data integration",
            "Institutional-grade execution"
        ],
        "endpoints": {
            "portfolio_management": "/api/v1/analytics/portfolio/",
            "risk_analytics": "/api/v1/analytics/risk/",
            "market_data": "/api/v1/analytics/market-data/",
            "optimization": "/api/v1/analytics/optimize",
            "indian_market": "/api/v1/indian-market/",
            "stock_recommendations": "/api/v1/indian-market/recommendations",
            "live_data": "/api/v1/indian-market/live-data/",
            "technical_analysis": "/api/v1/indian-market/technical-analysis/",
            "live_predictions": "/api/v1/live/predict/",
            "live_quotes": "/api/v1/live/quote/",
            "market_sentiment": "/api/v1/live/sentiment/market",
            "comprehensive_analysis": "/api/v1/live/analysis/comprehensive/",
            "hybrid_ai_earnings": "/api/v1/hybrid-ai/analyze/earnings",
            "hybrid_ai_thematic": "/api/v1/hybrid-ai/analyze/thematic",
            "hybrid_ai_comprehensive": "/api/v1/hybrid-ai/analyze/comprehensive",
            "human_review": "/api/v1/hybrid-ai/human-review/",
            "ai_orchestrator": "/api/v1/ai/analyze",
            "ai_system_status": "/api/v1/ai/system/status",
            "foundation_models": "/api/v1/ai/models/foundation",
            "specialized_models": "/api/v1/ai/models/specialized"
        },
        "new_features": [
            "🇮🇳 Live Indian stock market data",
            "🤖 AI-powered stock recommendations", 
            "📊 Technical analysis with buy/sell signals",
            "🔍 Advanced stock screening",
            "📈 Sector analysis and market outlook",
            "⚡ Real-time market sentiment",
            "🔮 Live ML-based price predictions",
            "📱 Real-time quotes and intraday data",
            "🎯 Comprehensive stock analysis",
            "🚨 Automated trading alerts",
            "🧠 Hybrid AI with specialized models",
            "🏛️ BlackRock Aladdin-inspired architecture",
            "👥 Human-in-the-loop validation",
            "🛡️ Comprehensive guardrail system"
        ]
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)