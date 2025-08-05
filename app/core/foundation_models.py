"""
Foundation Model Integrations
GPT-4, Gemini, and Claude integrations for general reasoning and orchestration
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from app.core.ai_orchestrator import FoundationModel, InvestmentQuery, ModelResponse, QueryType, ModelType

logger = logging.getLogger(__name__)


class GPT4Model(FoundationModel):
    """GPT-4 integration for advanced reasoning and analysis"""
    
    def __init__(self, api_key: str, model_version: str = "gpt-4"):
        super().__init__(
            model_id=f"gpt4_{model_version}",
            api_endpoint="https://api.openai.com/v1/chat/completions",
            api_key=api_key
        )
        self.model_version = model_version
        self.max_tokens = 4000
        self.temperature = 0.1  # Low temperature for consistent financial analysis
        
    async def predict(self, query: InvestmentQuery) -> ModelResponse:
        """Generate prediction using GPT-4"""
        start_time = datetime.now()
        
        try:
            # Construct specialized prompt for financial analysis
            system_prompt = self._build_system_prompt(query.query_type)
            user_prompt = self._build_user_prompt(query)
            
            # Make API call to GPT-4
            response_data = await self._call_openai_api(system_prompt, user_prompt)
            
            # Parse and structure the response
            structured_response = self._parse_gpt4_response(response_data, query)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ModelResponse(
                model_id=self.model_id,
                model_type=ModelType.FOUNDATION,
                prediction=structured_response,
                confidence=0.80,  # GPT-4 has high general confidence
                explanation=f"Analysis by GPT-4 using advanced reasoning and broad knowledge",
                processing_time=processing_time,
                metadata={
                    "model_version": self.model_version,
                    "tokens_used": response_data.get("usage", {}).get("total_tokens", 0)
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"GPT-4 prediction failed: {e}")
            # Return fallback response
            return await self._generate_fallback_response(query, start_time)
    
    def _build_system_prompt(self, query_type: QueryType) -> str:
        """Build system prompt based on query type"""
        
        base_prompt = """You are an expert financial analyst with deep knowledge of markets, 
        economics, and investment strategies. Provide accurate, data-driven analysis with 
        clear reasoning and appropriate risk considerations."""
        
        if query_type == QueryType.EARNINGS_ANALYSIS:
            return base_prompt + """ Focus on earnings call analysis, management guidance, 
            financial metrics, and market implications. Provide investment signals with confidence levels."""
            
        elif query_type == QueryType.THEMATIC_IDENTIFICATION:
            return base_prompt + """ Identify emerging investment themes by analyzing market trends, 
            technological developments, regulatory changes, and economic shifts. Suggest specific 
            investment vehicles and time horizons."""
            
        elif query_type == QueryType.RISK_ASSESSMENT:
            return base_prompt + """ Conduct comprehensive risk analysis including market risk, 
            credit risk, operational risk, and systemic risk. Provide quantitative risk metrics 
            where possible."""
            
        elif query_type == QueryType.PORTFOLIO_OPTIMIZATION:
            return base_prompt + """ Analyze portfolio construction, asset allocation, and 
            optimization strategies. Consider risk-return profiles, correlation, and diversification."""
            
        else:
            return base_prompt + """ Provide comprehensive market analysis with actionable insights."""
    
    def _build_user_prompt(self, query: InvestmentQuery) -> str:
        """Build user prompt from investment query"""
        
        prompt = f"""
        Query Type: {query.query_type.value}
        Analysis Request: {query.query_text}
        
        Symbols/Assets: {', '.join(query.symbols) if query.symbols else 'General Market'}
        Time Horizon: {query.time_horizon}
        Risk Tolerance: {query.risk_tolerance}
        
        Please provide:
        1. Key Analysis Points
        2. Investment Implications
        3. Risk Considerations
        4. Actionable Recommendations
        5. Confidence Level (0-100%)
        
        Format your response as structured JSON with clear sections.
        """
        
        return prompt.strip()
    
    async def _call_openai_api(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Make API call to OpenAI GPT-4"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_version,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"}
        }
        
        # For demo purposes, simulate API response
        await asyncio.sleep(0.5)  # Simulate API latency
        
        return {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "analysis": "GPT-4 powered comprehensive market analysis",
                        "key_points": [
                            "Strong fundamental indicators",
                            "Positive market sentiment",
                            "Favorable technical setup"
                        ],
                        "investment_implications": "Bullish outlook with moderate risk",
                        "risk_considerations": ["Market volatility", "Economic uncertainty"],
                        "recommendations": ["Consider position increase", "Monitor key levels"],
                        "confidence_level": 82
                    })
                }
            }],
            "usage": {"total_tokens": 350}
        }
    
    def _parse_gpt4_response(self, response_data: Dict[str, Any], query: InvestmentQuery) -> Dict[str, Any]:
        """Parse and structure GPT-4 response"""
        
        try:
            content = response_data["choices"][0]["message"]["content"]
            parsed_content = json.loads(content)
            
            # Add metadata and structure
            structured_response = {
                "model_source": "gpt4_foundation",
                "query_type": query.query_type.value,
                "analysis": parsed_content,
                "symbols_analyzed": query.symbols,
                "reasoning_type": "foundation_model_general_reasoning",
                "capabilities": ["cross_domain_analysis", "strategic_planning", "comprehensive_reasoning"]
            }
            
            return structured_response
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse GPT-4 response: {e}")
            return {
                "model_source": "gpt4_foundation",
                "analysis": "GPT-4 analysis completed with parsing issues",
                "raw_response": str(response_data),
                "error": "Response parsing failed"
            }


class GeminiModel(FoundationModel):
    """Google Gemini integration for multimodal analysis"""
    
    def __init__(self, api_key: str, model_version: str = "gemini-pro"):
        super().__init__(
            model_id=f"gemini_{model_version}",
            api_endpoint="https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent",
            api_key=api_key
        )
        self.model_version = model_version
        
    async def predict(self, query: InvestmentQuery) -> ModelResponse:
        """Generate prediction using Gemini"""
        start_time = datetime.now()
        
        try:
            # Simulate Gemini API call with multimodal capabilities
            response = await self._generate_gemini_response(query)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ModelResponse(
                model_id=self.model_id,
                model_type=ModelType.FOUNDATION,
                prediction=response,
                confidence=0.78,
                explanation="Gemini multimodal analysis with advanced reasoning",
                processing_time=processing_time,
                metadata={
                    "model_version": self.model_version,
                    "multimodal_capable": True
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Gemini prediction failed: {e}")
            return await self._generate_fallback_response(query, start_time)
    
    async def _generate_gemini_response(self, query: InvestmentQuery) -> Dict[str, Any]:
        """Generate Gemini-specific response"""
        
        # Simulate API call
        await asyncio.sleep(0.4)
        
        return {
            "model_source": "gemini_foundation",
            "query_type": query.query_type.value,
            "analysis": {
                "multimodal_insights": "Gemini's multimodal analysis capabilities",
                "reasoning_chain": [
                    "Data pattern recognition",
                    "Cross-modal correlation analysis", 
                    "Contextual understanding"
                ],
                "market_assessment": "Comprehensive market evaluation using advanced AI",
                "confidence_factors": ["Data quality", "Pattern consistency", "Historical validation"]
            },
            "capabilities": ["multimodal_analysis", "complex_reasoning", "pattern_recognition"],
            "symbols_analyzed": query.symbols
        }


class ClaudeModel(FoundationModel):
    """Anthropic Claude integration for ethical and safety-focused analysis"""
    
    def __init__(self, api_key: str, model_version: str = "claude-3-opus"):
        super().__init__(
            model_id=f"claude_{model_version}",
            api_endpoint="https://api.anthropic.com/v1/messages",
            api_key=api_key
        )
        self.model_version = model_version
        
    async def predict(self, query: InvestmentQuery) -> ModelResponse:
        """Generate prediction using Claude"""
        start_time = datetime.now()
        
        try:
            response = await self._generate_claude_response(query)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ModelResponse(
                model_id=self.model_id,
                model_type=ModelType.FOUNDATION,
                prediction=response,
                confidence=0.75,
                explanation="Claude analysis with emphasis on ethical considerations and safety",
                processing_time=processing_time,
                metadata={
                    "model_version": self.model_version,
                    "safety_focused": True,
                    "ethical_guidelines": True
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Claude prediction failed: {e}")
            return await self._generate_fallback_response(query, start_time)
    
    async def _generate_claude_response(self, query: InvestmentQuery) -> Dict[str, Any]:
        """Generate Claude-specific response with ethical focus"""
        
        # Simulate API call
        await asyncio.sleep(0.3)
        
        return {
            "model_source": "claude_foundation",
            "query_type": query.query_type.value,
            "analysis": {
                "ethical_assessment": "Investment recommendation considers ethical implications",
                "safety_considerations": [
                    "Risk disclosure completeness",
                    "Regulatory compliance",
                    "Client best interest alignment"
                ],
                "detailed_reasoning": "Step-by-step analysis with transparent methodology",
                "uncertainty_quantification": "Clear communication of analysis limitations",
                "responsible_ai_principles": "Analysis follows responsible AI guidelines"
            },
            "capabilities": ["ethical_reasoning", "safety_analysis", "detailed_explanations"],
            "symbols_analyzed": query.symbols,
            "compliance_notes": "Analysis adheres to financial advisory standards"
        }


# Factory function for creating foundation models
def create_foundation_models(config: Dict[str, Any]) -> Dict[str, FoundationModel]:
    """Factory function to create foundation model instances"""
    
    models = {}
    
    # Create GPT-4 model if configured
    if config.get("gpt4", {}).get("enabled", False):
        gpt4_config = config["gpt4"]
        models["gpt4"] = GPT4Model(
            api_key=gpt4_config.get("api_key", "demo_key"),
            model_version=gpt4_config.get("model_version", "gpt-4")
        )
    
    # Create Gemini model if configured
    if config.get("gemini", {}).get("enabled", False):
        gemini_config = config["gemini"]
        models["gemini"] = GeminiModel(
            api_key=gemini_config.get("api_key", "demo_key"),
            model_version=gemini_config.get("model_version", "gemini-pro")
        )
    
    # Create Claude model if configured
    if config.get("claude", {}).get("enabled", False):
        claude_config = config["claude"]
        models["claude"] = ClaudeModel(
            api_key=claude_config.get("api_key", "demo_key"),
            model_version=claude_config.get("model_version", "claude-3-opus")
        )
    
    return models