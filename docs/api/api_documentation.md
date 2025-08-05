# Hybrid AI Architecture - API Documentation

## Overview

The Hybrid AI Architecture provides RESTful APIs for accessing advanced AI-driven investment management capabilities. This documentation covers all available endpoints, request/response formats, authentication, and usage examples.

## Base URL
```
Production: https://api.hybrid-ai.platform.com/v1
Staging: https://staging-api.hybrid-ai.platform.com/v1
Development: http://localhost:8000/v1
```

## Authentication

### API Key Authentication
```http
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

### OAuth 2.0 (Recommended)
```http
Authorization: Bearer YOUR_ACCESS_TOKEN
Content-Type: application/json
```

## Core Endpoints

### 1. Investment Analysis

#### Analyze Investment Opportunity
Comprehensive analysis of investment opportunities using hybrid AI models.

```http
POST /analysis/investment
```

**Request Body:**
```json
{
  "query": "Analyze Tesla's Q4 earnings and investment potential",
  "symbols": ["TSLA"],
  "analysis_type": "comprehensive",
  "time_horizon": "12_months",
  "risk_tolerance": "moderate",
  "user_context": {
    "portfolio_size": 1000000,
    "investment_style": "growth",
    "sector_preferences": ["technology", "automotive"]
  }
}
```

**Response:**
```json
{
  "analysis_id": "ana_1234567890",
  "timestamp": "2024-01-15T10:30:00Z",
  "query": "Analyze Tesla's Q4 earnings and investment potential",
  "results": {
    "overall_recommendation": "BUY",
    "confidence_score": 0.87,
    "target_price": 245.50,
    "risk_rating": "MODERATE_HIGH",
    "time_horizon": "12_months",
    "key_insights": [
      "Strong Q4 delivery numbers exceeded expectations",
      "Cybertruck production ramp showing positive momentum",
      "Energy storage business growing at 40% YoY"
    ],
    "model_contributions": {
      "earnings_analyzer": {
        "confidence": 0.92,
        "sentiment": "positive",
        "key_metrics": {
          "revenue_growth": 0.15,
          "margin_improvement": 0.03
        }
      },
      "sentiment_analyzer": {
        "confidence": 0.84,
        "market_sentiment": 0.72,
        "news_sentiment": 0.68
      },
      "risk_predictor": {
        "confidence": 0.89,
        "volatility_forecast": 0.35,
        "downside_risk": 0.18
      }
    }
  },
  "guardrails": {
    "validation_passed": true,
    "hallucination_score": 0.02,
    "fact_check_score": 0.98
  },
  "human_review": {
    "required": false,
    "expert_validation": null
  }
}
```

#### Get Analysis History
Retrieve historical analysis results for a user.

```http
GET /analysis/history?limit=10&offset=0&symbol=TSLA
```

**Response:**
```json
{
  "analyses": [
    {
      "analysis_id": "ana_1234567890",
      "timestamp": "2024-01-15T10:30:00Z",
      "symbol": "TSLA",
      "recommendation": "BUY",
      "confidence_score": 0.87
    }
  ],
  "total_count": 156,
  "has_more": true
}
```

### 2. Specialized Model Endpoints

#### Earnings Analysis
Analyze earnings calls and financial reports using specialized models.

```http
POST /models/earnings/analyze
```

**Request Body:**
```json
{
  "symbol": "AAPL",
  "quarter": "Q4",
  "year": 2024,
  "transcript_text": "Thank you for joining us today...",
  "analysis_focus": ["sentiment", "guidance", "key_metrics"]
}
```

**Response:**
```json
{
  "analysis_id": "earn_9876543210",
  "symbol": "AAPL",
  "quarter": "Q4_2024",
  "results": {
    "overall_sentiment": 0.78,
    "confidence": 0.94,
    "key_themes": [
      "iPhone sales strength",
      "Services growth acceleration",
      "Supply chain improvements"
    ],
    "financial_metrics": {
      "revenue_surprise": 0.03,
      "eps_surprise": 0.05,
      "guidance_tone": "positive"
    },
    "market_impact_prediction": {
      "price_movement": 0.025,
      "volatility_increase": 0.15
    }
  }
}
```

#### Thematic Investment Identification
Identify thematic investment opportunities using proprietary datasets.

```http
POST /models/thematic/identify
```

**Request Body:**
```json
{
  "themes": ["artificial_intelligence", "renewable_energy"],
  "time_horizon": "24_months",
  "investment_size": 500000,
  "risk_tolerance": "moderate"
}
```

**Response:**
```json
{
  "theme_analysis": {
    "artificial_intelligence": {
      "strength_score": 0.89,
      "growth_potential": 0.92,
      "recommended_vehicles": [
        {
          "symbol": "NVDA",
          "allocation": 0.25,
          "rationale": "Leading AI chip manufacturer"
        },
        {
          "symbol": "MSFT",
          "allocation": 0.20,
          "rationale": "Azure AI services growth"
        }
      ]
    }
  }
}
```

#### Financial Sentiment Analysis
Analyze market sentiment using financial news and social media data.

```http
POST /models/sentiment/analyze
```

**Request Body:**
```json
{
  "symbols": ["TSLA", "AAPL", "GOOGL"],
  "time_range": "7_days",
  "sources": ["news", "social_media", "analyst_reports"]
}
```

**Response:**
```json
{
  "sentiment_analysis": {
    "TSLA": {
      "overall_sentiment": 0.65,
      "news_sentiment": 0.72,
      "social_sentiment": 0.58,
      "analyst_sentiment": 0.71,
      "trend": "improving",
      "key_drivers": [
        "Positive delivery numbers",
        "Cybertruck production updates"
      ]
    }
  }
}
```

#### Risk Prediction
Predict portfolio and individual security risks using advanced models.

```http
POST /models/risk/predict
```

**Request Body:**
```json
{
  "portfolio": {
    "positions": [
      {"symbol": "AAPL", "weight": 0.30},
      {"symbol": "GOOGL", "weight": 0.25},
      {"symbol": "TSLA", "weight": 0.20}
    ]
  },
  "time_horizons": ["1_month", "3_months", "12_months"],
  "risk_metrics": ["var", "expected_shortfall", "maximum_drawdown"]
}
```

**Response:**
```json
{
  "risk_predictions": {
    "1_month": {
      "var_95": -0.08,
      "expected_shortfall": -0.12,
      "maximum_drawdown": -0.15,
      "confidence": 0.91
    },
    "portfolio_risk_factors": [
      {
        "factor": "technology_sector_concentration",
        "contribution": 0.45,
        "mitigation": "Consider diversification into other sectors"
      }
    ]
  }
}
```

### 3. Human-in-the-Loop Endpoints

#### Submit for Expert Review
Submit AI outputs for human expert validation.

```http
POST /review/submit
```

**Request Body:**
```json
{
  "analysis_id": "ana_1234567890",
  "expert_type": "portfolio_manager",
  "priority": "high",
  "review_context": {
    "client_importance": "high_value",
    "decision_impact": "major_allocation"
  }
}
```

**Response:**
```json
{
  "review_id": "rev_5555555555",
  "status": "queued",
  "estimated_completion": "2024-01-15T12:00:00Z",
  "assigned_expert": "expert_pm_001"
}
```

#### Get Review Status
Check the status of expert reviews.

```http
GET /review/{review_id}/status
```

**Response:**
```json
{
  "review_id": "rev_5555555555",
  "status": "completed",
  "expert_feedback": {
    "agreement_level": 0.85,
    "rating": 8,
    "comments": "Analysis is thorough, but consider adding sector rotation risk",
    "modifications": [
      {
        "field": "risk_rating",
        "original": "MODERATE_HIGH",
        "suggested": "HIGH",
        "rationale": "Increased regulatory risk not fully captured"
      }
    ]
  },
  "completed_at": "2024-01-15T11:45:00Z"
}
```

### 4. Monitoring and Analytics

#### System Health
Get current system health and performance metrics.

```http
GET /system/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "ai_orchestrator": "healthy",
    "foundation_models": "healthy",
    "specialized_models": "healthy",
    "guardrail_engine": "healthy",
    "database": "healthy"
  },
  "performance_metrics": {
    "avg_response_time": 245,
    "requests_per_minute": 1250,
    "error_rate": 0.002,
    "model_accuracy": 0.94
  }
}
```

#### Performance Analytics
Get detailed performance analytics and trends.

```http
GET /analytics/performance?timeframe=24h&metrics=response_time,accuracy,throughput
```

**Response:**
```json
{
  "timeframe": "24h",
  "metrics": {
    "response_time": {
      "p50": 180,
      "p95": 420,
      "p99": 850,
      "trend": "stable"
    },
    "accuracy": {
      "average": 0.94,
      "trend": "improving",
      "by_model": {
        "earnings_analyzer": 0.96,
        "sentiment_analyzer": 0.92,
        "risk_predictor": 0.95
      }
    },
    "throughput": {
      "requests_per_minute": 1250,
      "peak_rpm": 2100,
      "trend": "increasing"
    }
  }
}
```

## Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid symbol format",
    "details": {
      "field": "symbols",
      "provided": "INVALID_SYMBOL",
      "expected": "Valid stock symbol (e.g., AAPL)"
    },
    "request_id": "req_1234567890",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_API_KEY` | 401 | API key is missing or invalid |
| `INSUFFICIENT_PERMISSIONS` | 403 | User lacks required permissions |
| `VALIDATION_ERROR` | 400 | Request validation failed |
| `RATE_LIMIT_EXCEEDED` | 429 | API rate limit exceeded |
| `MODEL_UNAVAILABLE` | 503 | Requested model is temporarily unavailable |
| `INTERNAL_ERROR` | 500 | Internal server error |

## Rate Limits

### Standard Limits
- **Free Tier**: 100 requests/hour
- **Professional**: 1,000 requests/hour
- **Enterprise**: 10,000 requests/hour
- **Custom**: Negotiated limits

### Rate Limit Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642248000
```

## SDKs and Libraries

### Python SDK
```bash
pip install hybrid-ai-client
```

```python
from hybrid_ai_client import HybridAIClient

client = HybridAIClient(api_key="your_api_key")

# Analyze investment
result = client.analyze_investment(
    query="Analyze AAPL earnings potential",
    symbols=["AAPL"],
    time_horizon="12_months"
)

print(f"Recommendation: {result.recommendation}")
print(f"Confidence: {result.confidence_score}")
```

### JavaScript SDK
```bash
npm install @hybrid-ai/client
```

```javascript
import { HybridAIClient } from '@hybrid-ai/client';

const client = new HybridAIClient({ apiKey: 'your_api_key' });

// Analyze investment
const result = await client.analyzeInvestment({
  query: 'Analyze AAPL earnings potential',
  symbols: ['AAPL'],
  timeHorizon: '12_months'
});

console.log(`Recommendation: ${result.recommendation}`);
console.log(`Confidence: ${result.confidenceScore}`);
```

## Interactive Examples

### Postman Collection
Download our Postman collection for interactive API testing:
[Download Postman Collection](https://api.hybrid-ai.platform.com/postman/collection.json)

### OpenAPI Specification
Access our complete OpenAPI 3.0 specification:
[OpenAPI Spec](https://api.hybrid-ai.platform.com/openapi.json)

### Swagger UI
Interactive API documentation and testing:
[Swagger UI](https://api.hybrid-ai.platform.com/docs)

## Support and Resources

### Documentation
- [Getting Started Guide](https://docs.hybrid-ai.platform.com/getting-started)
- [API Reference](https://docs.hybrid-ai.platform.com/api)
- [SDK Documentation](https://docs.hybrid-ai.platform.com/sdks)

### Support Channels
- **Email**: support@hybrid-ai.platform.com
- **Slack**: [Join our community](https://hybrid-ai-community.slack.com)
- **GitHub**: [Issues and discussions](https://github.com/hybrid-ai/platform)

### Status Page
Monitor API status and incidents:
[Status Page](https://status.hybrid-ai.platform.com)

This API documentation provides comprehensive coverage of all available endpoints, authentication methods, request/response formats, and practical examples for integrating with the Hybrid AI Architecture platform.