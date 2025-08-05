"""
Hybrid AI Engine - BlackRock Aladdin-inspired Architecture
Combines foundation models with specialized fine-tuned models for institutional-grade investment management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import json
import asyncio
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of models in the hybrid architecture"""
    FOUNDATION = "foundation"  # GPT-4, Gemini for general reasoning
    SPECIALIZED = "specialized"  # Custom fine-tuned models
    ENSEMBLE = "ensemble"  # Combination of multiple models
    VALIDATION = "validation"  # Models used for output validation


class TaskCategory(Enum):
    """Categories of investment management tasks"""
    MARKET_ANALYSIS = "market_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    EARNINGS_PREDICTION = "earnings_prediction"
    THEMATIC_IDENTIFICATION = "thematic_identification"
    REGULATORY_COMPLIANCE = "regulatory_compliance"


@dataclass
class ModelOutput:
    """Standardized model output with confidence and validation"""
    result: Any
    confidence: float
    model_type: ModelType
    task_category: TaskCategory
    timestamp: datetime
    validation_score: float
    human_reviewed: bool = False
    guardrail_passed: bool = True
    metadata: Dict = None


@dataclass
class ProprietaryDataset:
    """Represents a specialized proprietary dataset"""
    name: str
    description: str
    size: int
    last_updated: datetime
    quality_score: float
    source_types: List[str]
    validation_metrics: Dict


class BaseSpecializedModel(ABC):
    """Abstract base class for specialized models"""
    
    def __init__(self, name: str, task_category: TaskCategory):
        self.name = name
        self.task_category = task_category
        self.training_data: Optional[ProprietaryDataset] = None
        self.performance_metrics = {}
        self.last_validation = None
        
    @abstractmethod
    def predict(self, input_data: Any) -> ModelOutput:
        """Make prediction with the specialized model"""
        pass
    
    @abstractmethod
    def validate_output(self, output: Any) -> Tuple[bool, float]:
        """Validate model output against known patterns"""
        pass


class EarningsCallAnalysisModel(BaseSpecializedModel):
    """Specialized model for earnings call transcript analysis"""
    
    def __init__(self):
        super().__init__("EarningsCallAnalyzer", TaskCategory.EARNINGS_PREDICTION)
        self.training_data = ProprietaryDataset(
            name="Earnings Call Transcripts",
            description="400,000+ earnings call transcripts with market correlation data",
            size=400000,
            last_updated=datetime.now(),
            quality_score=0.95,
            source_types=["earnings_calls", "market_data", "analyst_reports"],
            validation_metrics={"accuracy": 0.87, "precision": 0.84, "recall": 0.89}
        )
        
    def predict(self, earnings_text: str) -> ModelOutput:
        """Analyze earnings call transcript for investment signals"""
        try:
            # Simulate advanced NLP analysis on earnings transcript
            sentiment_indicators = self._extract_sentiment_indicators(earnings_text)
            financial_metrics = self._extract_financial_metrics(earnings_text)
            forward_guidance = self._analyze_forward_guidance(earnings_text)
            
            # Generate investment signal
            signal_strength = self._calculate_signal_strength(
                sentiment_indicators, financial_metrics, forward_guidance
            )
            
            result = {
                "investment_signal": "bullish" if signal_strength > 0.6 else "bearish" if signal_strength < 0.4 else "neutral",
                "signal_strength": signal_strength,
                "key_themes": self._identify_key_themes(earnings_text),
                "risk_factors": self._identify_risk_factors(earnings_text),
                "price_target_adjustment": self._suggest_price_target_adjustment(signal_strength),
                "confidence_drivers": sentiment_indicators
            }
            
            # Validate output
            is_valid, validation_score = self.validate_output(result)
            
            return ModelOutput(
                result=result,
                confidence=signal_strength,
                model_type=ModelType.SPECIALIZED,
                task_category=self.task_category,
                timestamp=datetime.now(),
                validation_score=validation_score,
                guardrail_passed=is_valid,
                metadata={"transcript_length": len(earnings_text), "processing_time": 0.5}
            )
            
        except Exception as e:
            logger.error(f"Earnings analysis failed: {e}")
            return self._create_error_output(str(e))
    
    def _extract_sentiment_indicators(self, text: str) -> Dict:
        """Extract sentiment indicators from earnings text"""
        # Simulate advanced sentiment analysis
        positive_words = ["growth", "strong", "exceeded", "optimistic", "confident"]
        negative_words = ["challenging", "decline", "concerns", "headwinds", "uncertainty"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        return {
            "positive_sentiment": positive_count / (positive_count + negative_count + 1),
            "negative_sentiment": negative_count / (positive_count + negative_count + 1),
            "management_confidence": min(positive_count * 0.1, 1.0),
            "forward_looking_statements": len([s for s in text.split('.') if 'expect' in s.lower() or 'forecast' in s.lower()])
        }
    
    def _extract_financial_metrics(self, text: str) -> Dict:
        """Extract financial metrics mentioned in earnings call"""
        # Simulate financial metric extraction
        import re
        
        revenue_mentions = len(re.findall(r'revenue.*?(\d+\.?\d*)', text.lower()))
        margin_mentions = len(re.findall(r'margin.*?(\d+\.?\d*)', text.lower()))
        guidance_mentions = len(re.findall(r'guidance.*?(\d+\.?\d*)', text.lower()))
        
        return {
            "revenue_focus": min(revenue_mentions * 0.2, 1.0),
            "margin_discussion": min(margin_mentions * 0.3, 1.0),
            "guidance_clarity": min(guidance_mentions * 0.25, 1.0)
        }
    
    def _analyze_forward_guidance(self, text: str) -> Dict:
        """Analyze forward guidance quality and sentiment"""
        guidance_keywords = ["outlook", "expect", "forecast", "guidance", "target"]
        guidance_sentences = [s for s in text.split('.') if any(kw in s.lower() for kw in guidance_keywords)]
        
        return {
            "guidance_provided": len(guidance_sentences) > 0,
            "guidance_specificity": min(len(guidance_sentences) * 0.1, 1.0),
            "guidance_confidence": 0.7 if len(guidance_sentences) > 3 else 0.4
        }
    
    def _identify_key_themes(self, text: str) -> List[str]:
        """Identify key investment themes from earnings call"""
        themes = []
        theme_keywords = {
            "digital_transformation": ["digital", "cloud", "AI", "automation"],
            "sustainability": ["ESG", "sustainable", "green", "carbon"],
            "supply_chain": ["supply", "logistics", "inventory"],
            "market_expansion": ["expansion", "growth", "new markets"],
            "cost_optimization": ["efficiency", "cost", "optimization"]
        }
        
        text_lower = text.lower()
        for theme, keywords in theme_keywords.items():
            if any(kw in text_lower for kw in keywords):
                themes.append(theme)
        
        return themes
    
    def _identify_risk_factors(self, text: str) -> List[str]:
        """Identify risk factors mentioned in earnings call"""
        risk_keywords = {
            "regulatory": ["regulation", "compliance", "regulatory"],
            "competitive": ["competition", "competitive", "market share"],
            "economic": ["inflation", "recession", "economic"],
            "operational": ["supply chain", "labor", "capacity"]
        }
        
        risks = []
        text_lower = text.lower()
        for risk_type, keywords in risk_keywords.items():
            if any(kw in text_lower for kw in keywords):
                risks.append(risk_type)
        
        return risks
    
    def _calculate_signal_strength(self, sentiment: Dict, metrics: Dict, guidance: Dict) -> float:
        """Calculate overall investment signal strength"""
        sentiment_score = sentiment["positive_sentiment"] - sentiment["negative_sentiment"]
        metrics_score = (metrics["revenue_focus"] + metrics["margin_discussion"] + metrics["guidance_clarity"]) / 3
        guidance_score = guidance["guidance_confidence"] if guidance["guidance_provided"] else 0.3
        
        # Weighted combination
        signal_strength = (sentiment_score * 0.4 + metrics_score * 0.35 + guidance_score * 0.25)
        return max(0, min(1, (signal_strength + 1) / 2))  # Normalize to 0-1
    
    def _suggest_price_target_adjustment(self, signal_strength: float) -> str:
        """Suggest price target adjustment based on signal strength"""
        if signal_strength > 0.7:
            return "increase_5_10_percent"
        elif signal_strength > 0.6:
            return "increase_0_5_percent"
        elif signal_strength < 0.3:
            return "decrease_5_10_percent"
        elif signal_strength < 0.4:
            return "decrease_0_5_percent"
        else:
            return "maintain"
    
    def validate_output(self, output: Any) -> Tuple[bool, float]:
        """Validate earnings analysis output"""
        try:
            required_fields = ["investment_signal", "signal_strength", "key_themes", "risk_factors"]
            
            # Check required fields
            if not all(field in output for field in required_fields):
                return False, 0.0
            
            # Validate signal strength range
            if not (0 <= output["signal_strength"] <= 1):
                return False, 0.2
            
            # Validate signal consistency
            signal = output["investment_signal"]
            strength = output["signal_strength"]
            
            if signal == "bullish" and strength < 0.5:
                return False, 0.3
            if signal == "bearish" and strength > 0.5:
                return False, 0.3
            
            # All validations passed
            return True, 0.95
            
        except Exception as e:
            logger.error(f"Output validation failed: {e}")
            return False, 0.0
    
    def _create_error_output(self, error_msg: str) -> ModelOutput:
        """Create error output when analysis fails"""
        return ModelOutput(
            result={"error": error_msg, "investment_signal": "neutral"},
            confidence=0.0,
            model_type=ModelType.SPECIALIZED,
            task_category=self.task_category,
            timestamp=datetime.now(),
            validation_score=0.0,
            guardrail_passed=False
        )


class ThematicInvestmentModel(BaseSpecializedModel):
    """Specialized model for identifying thematic investment opportunities"""
    
    def __init__(self):
        super().__init__("ThematicInvestmentIdentifier", TaskCategory.THEMATIC_IDENTIFICATION)
        self.training_data = ProprietaryDataset(
            name="Thematic Investment Data",
            description="Historical thematic trends with performance correlation",
            size=150000,
            last_updated=datetime.now(),
            quality_score=0.92,
            source_types=["market_data", "news", "patent_filings", "regulatory_data"],
            validation_metrics={"theme_accuracy": 0.81, "timing_accuracy": 0.73}
        )
        
    def predict(self, market_data: Dict) -> ModelOutput:
        """Identify emerging thematic investment opportunities"""
        try:
            # Analyze multiple data sources for thematic signals
            news_themes = self._analyze_news_themes(market_data.get("news", []))
            patent_trends = self._analyze_patent_trends(market_data.get("patents", []))
            regulatory_signals = self._analyze_regulatory_signals(market_data.get("regulations", []))
            market_momentum = self._analyze_market_momentum(market_data.get("price_data", {}))
            
            # Identify top themes
            themes = self._synthesize_themes(news_themes, patent_trends, regulatory_signals, market_momentum)
            
            result = {
                "top_themes": themes[:5],  # Top 5 themes
                "theme_strength": {theme["name"]: theme["strength"] for theme in themes[:5]},
                "investment_vehicles": self._suggest_investment_vehicles(themes[:3]),
                "risk_assessment": self._assess_thematic_risks(themes[:3]),
                "time_horizon": self._estimate_time_horizon(themes[:3])
            }
            
            # Calculate overall confidence
            avg_strength = np.mean([theme["strength"] for theme in themes[:5]]) if themes else 0
            
            # Validate output
            is_valid, validation_score = self.validate_output(result)
            
            return ModelOutput(
                result=result,
                confidence=avg_strength,
                model_type=ModelType.SPECIALIZED,
                task_category=self.task_category,
                timestamp=datetime.now(),
                validation_score=validation_score,
                guardrail_passed=is_valid,
                metadata={"themes_analyzed": len(themes), "data_sources": len(market_data)}
            )
            
        except Exception as e:
            logger.error(f"Thematic analysis failed: {e}")
            return self._create_error_output(str(e))
    
    def _analyze_news_themes(self, news_data: List[Dict]) -> List[Dict]:
        """Analyze news for emerging themes"""
        # Simulate advanced news theme analysis
        theme_keywords = {
            "artificial_intelligence": ["AI", "machine learning", "neural networks", "automation"],
            "clean_energy": ["solar", "wind", "renewable", "clean energy", "battery"],
            "cybersecurity": ["cyber", "security", "hacking", "data breach"],
            "biotechnology": ["biotech", "gene therapy", "CRISPR", "pharmaceutical"],
            "space_technology": ["space", "satellite", "SpaceX", "aerospace"]
        }
        
        theme_scores = {}
        for theme, keywords in theme_keywords.items():
            score = 0
            for news_item in news_data:
                content = news_item.get("content", "").lower()
                score += sum(1 for kw in keywords if kw.lower() in content)
            theme_scores[theme] = min(score / 10, 1.0)  # Normalize
        
        return [{"name": theme, "strength": score, "source": "news"} 
                for theme, score in theme_scores.items() if score > 0.1]
    
    def _analyze_patent_trends(self, patent_data: List[Dict]) -> List[Dict]:
        """Analyze patent filings for technology trends"""
        # Simulate patent trend analysis
        tech_categories = {
            "quantum_computing": 0.3,
            "autonomous_vehicles": 0.7,
            "blockchain": 0.4,
            "augmented_reality": 0.6,
            "5g_technology": 0.8
        }
        
        return [{"name": tech, "strength": strength, "source": "patents"} 
                for tech, strength in tech_categories.items()]
    
    def _analyze_regulatory_signals(self, regulatory_data: List[Dict]) -> List[Dict]:
        """Analyze regulatory changes for investment implications"""
        # Simulate regulatory signal analysis
        regulatory_themes = {
            "data_privacy": 0.6,
            "financial_regulation": 0.4,
            "environmental_compliance": 0.8,
            "healthcare_reform": 0.5
        }
        
        return [{"name": theme, "strength": strength, "source": "regulatory"} 
                for theme, strength in regulatory_themes.items()]
    
    def _analyze_market_momentum(self, price_data: Dict) -> List[Dict]:
        """Analyze market momentum for thematic sectors"""
        # Simulate market momentum analysis
        sector_momentum = {
            "technology": 0.7,
            "healthcare": 0.6,
            "renewable_energy": 0.8,
            "fintech": 0.5,
            "e_commerce": 0.4
        }
        
        return [{"name": sector, "strength": momentum, "source": "market"} 
                for sector, momentum in sector_momentum.items()]
    
    def _synthesize_themes(self, *theme_sources) -> List[Dict]:
        """Synthesize themes from multiple sources"""
        all_themes = {}
        
        for source in theme_sources:
            for theme in source:
                name = theme["name"]
                if name in all_themes:
                    all_themes[name]["strength"] = max(all_themes[name]["strength"], theme["strength"])
                    all_themes[name]["sources"].append(theme["source"])
                else:
                    all_themes[name] = {
                        "name": name,
                        "strength": theme["strength"],
                        "sources": [theme["source"]]
                    }
        
        # Sort by strength and multi-source confirmation
        themes = list(all_themes.values())
        for theme in themes:
            # Boost themes confirmed by multiple sources
            theme["strength"] *= (1 + 0.2 * (len(theme["sources"]) - 1))
            theme["strength"] = min(theme["strength"], 1.0)
        
        return sorted(themes, key=lambda x: x["strength"], reverse=True)
    
    def _suggest_investment_vehicles(self, themes: List[Dict]) -> Dict:
        """Suggest investment vehicles for top themes"""
        vehicle_mapping = {
            "artificial_intelligence": ["ARKQ", "BOTZ", "ROBO", "Individual AI stocks"],
            "clean_energy": ["ICLN", "PBW", "QCLN", "Solar/Wind companies"],
            "cybersecurity": ["HACK", "CIBR", "BUG", "Security software companies"],
            "biotechnology": ["IBB", "XBI", "ARKG", "Biotech individual stocks"],
            "space_technology": ["UFO", "ARKX", "Space ETFs", "Aerospace companies"]
        }
        
        suggestions = {}
        for theme in themes:
            theme_name = theme["name"]
            if theme_name in vehicle_mapping:
                suggestions[theme_name] = vehicle_mapping[theme_name]
        
        return suggestions
    
    def _assess_thematic_risks(self, themes: List[Dict]) -> Dict:
        """Assess risks associated with thematic investments"""
        risk_profiles = {
            "artificial_intelligence": ["Regulatory uncertainty", "Competition", "Technical challenges"],
            "clean_energy": ["Policy changes", "Technology disruption", "Commodity prices"],
            "cybersecurity": ["Market saturation", "Rapid technology change", "Economic sensitivity"],
            "biotechnology": ["Regulatory approval", "Clinical trial failures", "Patent expiration"],
            "space_technology": ["High capital requirements", "Technical risks", "Regulatory approval"]
        }
        
        risks = {}
        for theme in themes:
            theme_name = theme["name"]
            if theme_name in risk_profiles:
                risks[theme_name] = risk_profiles[theme_name]
        
        return risks
    
    def _estimate_time_horizon(self, themes: List[Dict]) -> Dict:
        """Estimate investment time horizon for themes"""
        time_horizons = {
            "artificial_intelligence": "2-5 years",
            "clean_energy": "3-7 years",
            "cybersecurity": "1-3 years",
            "biotechnology": "5-10 years",
            "space_technology": "5-15 years"
        }
        
        horizons = {}
        for theme in themes:
            theme_name = theme["name"]
            if theme_name in time_horizons:
                horizons[theme_name] = time_horizons[theme_name]
        
        return horizons
    
    def validate_output(self, output: Any) -> Tuple[bool, float]:
        """Validate thematic analysis output"""
        try:
            required_fields = ["top_themes", "theme_strength", "investment_vehicles"]
            
            if not all(field in output for field in required_fields):
                return False, 0.0
            
            # Validate theme strengths
            for strength in output["theme_strength"].values():
                if not (0 <= strength <= 1):
                    return False, 0.3
            
            return True, 0.9
            
        except Exception as e:
            logger.error(f"Thematic validation failed: {e}")
            return False, 0.0
    
    def _create_error_output(self, error_msg: str) -> ModelOutput:
        """Create error output when analysis fails"""
        return ModelOutput(
            result={"error": error_msg, "top_themes": []},
            confidence=0.0,
            model_type=ModelType.SPECIALIZED,
            task_category=self.task_category,
            timestamp=datetime.now(),
            validation_score=0.0,
            guardrail_passed=False
        )


class OutputGuardrailSystem:
    """Comprehensive guardrail system for model outputs"""
    
    def __init__(self):
        self.validation_rules = {}
        self.trusted_sources = {}
        self.hallucination_detectors = {}
        
    def add_validation_rule(self, task_category: TaskCategory, rule_func):
        """Add validation rule for specific task category"""
        if task_category not in self.validation_rules:
            self.validation_rules[task_category] = []
        self.validation_rules[task_category].append(rule_func)
    
    def validate_output(self, output: ModelOutput) -> Tuple[bool, List[str]]:
        """Comprehensive output validation"""
        issues = []
        
        # Basic structure validation
        if not self._validate_structure(output):
            issues.append("Invalid output structure")
        
        # Task-specific validation
        if output.task_category in self.validation_rules:
            for rule in self.validation_rules[output.task_category]:
                if not rule(output.result):
                    issues.append(f"Failed task-specific validation: {rule.__name__}")
        
        # Hallucination detection
        if self._detect_hallucination(output):
            issues.append("Potential hallucination detected")
        
        # Confidence validation
        if not self._validate_confidence(output):
            issues.append("Confidence score inconsistent with output quality")
        
        return len(issues) == 0, issues
    
    def _validate_structure(self, output: ModelOutput) -> bool:
        """Validate basic output structure"""
        try:
            return (
                hasattr(output, 'result') and
                hasattr(output, 'confidence') and
                hasattr(output, 'timestamp') and
                0 <= output.confidence <= 1
            )
        except:
            return False
    
    def _detect_hallucination(self, output: ModelOutput) -> bool:
        """Detect potential hallucinations in output"""
        # Implement hallucination detection logic
        # This is a simplified version - in production, use more sophisticated methods
        
        if isinstance(output.result, dict):
            # Check for impossible values
            for key, value in output.result.items():
                if isinstance(value, (int, float)):
                    if abs(value) > 1e10:  # Unreasonably large numbers
                        return True
                elif isinstance(value, str):
                    if len(value) > 10000:  # Unreasonably long strings
                        return True
        
        return False
    
    def _validate_confidence(self, output: ModelOutput) -> bool:
        """Validate confidence score consistency"""
        # High confidence should correlate with high validation score
        if output.confidence > 0.8 and output.validation_score < 0.5:
            return False
        
        # Low confidence should not have perfect validation
        if output.confidence < 0.3 and output.validation_score > 0.9:
            return False
        
        return True


class HumanInTheLoopSystem:
    """Human-in-the-loop validation and collaboration system"""
    
    def __init__(self):
        self.pending_reviews = {}
        self.expert_feedback = {}
        self.collaboration_history = {}
        
    def submit_for_review(self, output: ModelOutput, expert_type: str = "portfolio_manager") -> str:
        """Submit model output for human expert review"""
        review_id = hashlib.md5(f"{output.timestamp}_{expert_type}".encode()).hexdigest()[:8]
        
        self.pending_reviews[review_id] = {
            "output": output,
            "expert_type": expert_type,
            "submitted_at": datetime.now(),
            "status": "pending",
            "priority": self._calculate_priority(output)
        }
        
        logger.info(f"Submitted output for {expert_type} review: {review_id}")
        return review_id
    
    def provide_expert_feedback(self, review_id: str, feedback: Dict) -> bool:
        """Record expert feedback on model output"""
        if review_id not in self.pending_reviews:
            return False
        
        self.expert_feedback[review_id] = {
            "feedback": feedback,
            "reviewed_at": datetime.now(),
            "reviewer": feedback.get("reviewer", "unknown")
        }
        
        self.pending_reviews[review_id]["status"] = "reviewed"
        
        # Update model output with human validation
        output = self.pending_reviews[review_id]["output"]
        output.human_reviewed = True
        
        # Store collaboration history
        self._update_collaboration_history(output, feedback)
        
        return True
    
    def get_collaborative_recommendation(self, task_data: Dict, expert_type: str) -> Dict:
        """Generate collaborative recommendation combining AI and human expertise"""
        # This would integrate with actual expert systems in production
        
        collaboration_result = {
            "ai_recommendation": "Generated by specialized model",
            "expert_input": f"Validated by {expert_type}",
            "combined_confidence": 0.85,
            "collaboration_quality": "high",
            "next_steps": ["Monitor implementation", "Schedule follow-up review"]
        }
        
        return collaboration_result
    
    def _calculate_priority(self, output: ModelOutput) -> str:
        """Calculate review priority based on output characteristics"""
        if output.confidence < 0.5 or not output.guardrail_passed:
            return "high"
        elif output.validation_score < 0.7:
            return "medium"
        else:
            return "low"
    
    def _update_collaboration_history(self, output: ModelOutput, feedback: Dict):
        """Update collaboration history for learning"""
        task_key = f"{output.task_category.value}_{output.model_type.value}"
        
        if task_key not in self.collaboration_history:
            self.collaboration_history[task_key] = []
        
        self.collaboration_history[task_key].append({
            "timestamp": datetime.now(),
            "ai_confidence": output.confidence,
            "human_rating": feedback.get("rating", 0),
            "agreement_level": feedback.get("agreement", 0),
            "improvement_suggestions": feedback.get("suggestions", [])
        })


class HybridAIEngine:
    """Main hybrid AI engine orchestrating foundation and specialized models"""
    
    def __init__(self):
        self.foundation_models = {}  # Would integrate with GPT-4, Gemini, etc.
        self.specialized_models = {
            "earnings_analysis": EarningsCallAnalysisModel(),
            "thematic_identification": ThematicInvestmentModel(),
            "sentiment_analysis": self._create_sentiment_model(),
            "risk_prediction": self._create_risk_prediction_model()
        }
        self.guardrail_system = self._create_guardrail_system()
        self.human_loop_system = self._create_human_loop_system()
        self.performance_tracker = {}
        
        # Initialize guardrail rules
        self._setup_guardrail_rules()
        
        # Initialize advanced orchestrator
        self.advanced_orchestrator = self._create_advanced_orchestrator()
    
    def _create_sentiment_model(self):
        """Create financial sentiment analysis model"""
        try:
            from app.core.sentiment_analysis_model import FinancialSentimentModel
            return FinancialSentimentModel()
        except ImportError:
            logger.warning("Sentiment analysis model not available")
            return None
    
    def _create_risk_prediction_model(self):
        """Create risk prediction model"""
        try:
            from app.core.risk_prediction_model import RiskPredictionModel
            return RiskPredictionModel()
        except ImportError:
            logger.warning("Risk prediction model not available")
            return None
    
    def _create_guardrail_system(self):
        """Create comprehensive guardrail system"""
        try:
            from app.core.guardrail_engine import ComprehensiveGuardrailEngine
            return ComprehensiveGuardrailEngine()
        except ImportError:
            logger.warning("Guardrail engine not available, using basic system")
            return OutputGuardrailSystem()
    
    def _create_human_loop_system(self):
        """Create enhanced human-in-the-loop system"""
        try:
            from app.core.human_in_the_loop import HumanInTheLoopSystem
            return HumanInTheLoopSystem()
        except ImportError:
            logger.warning("Enhanced human-in-the-loop system not available, using basic system")
            # Fallback to basic system if available
            return HumanInTheLoopSystem() if 'HumanInTheLoopSystem' in globals() else None
    
    def _create_advanced_orchestrator(self):
        """Create advanced orchestration system"""
        try:
            from app.core.advanced_orchestration import create_advanced_orchestrator, ModelCapability
            orchestrator = create_advanced_orchestrator()
            
            # Register specialized models with their capabilities
            if "earnings_analysis" in self.specialized_models:
                orchestrator.register_model(
                    "earnings_analysis", 
                    self.specialized_models["earnings_analysis"],
                    [ModelCapability.DOMAIN_EXPERTISE, ModelCapability.PATTERN_RECOGNITION, ModelCapability.FORECASTING]
                )
            
            if "sentiment_analysis" in self.specialized_models:
                orchestrator.register_model(
                    "sentiment_analysis",
                    self.specialized_models["sentiment_analysis"],
                    [ModelCapability.SENTIMENT_ANALYSIS, ModelCapability.PATTERN_RECOGNITION, ModelCapability.DOMAIN_EXPERTISE]
                )
            
            if "risk_prediction" in self.specialized_models:
                orchestrator.register_model(
                    "risk_prediction",
                    self.specialized_models["risk_prediction"],
                    [ModelCapability.RISK_ASSESSMENT, ModelCapability.NUMERICAL_ANALYSIS, ModelCapability.FORECASTING]
                )
            
            if "thematic_identification" in self.specialized_models:
                orchestrator.register_model(
                    "thematic_identification",
                    self.specialized_models["thematic_identification"],
                    [ModelCapability.PATTERN_RECOGNITION, ModelCapability.GENERAL_REASONING, ModelCapability.DOMAIN_EXPERTISE]
                )
            
            return orchestrator
            
        except ImportError:
            logger.warning("Advanced orchestrator not available")
            return None
        
    def _setup_guardrail_rules(self):
        """Setup validation rules for different task categories"""
        
        # Check if we have the new comprehensive guardrail system
        if hasattr(self.guardrail_system, 'guardrails'):
            # New comprehensive system - already configured
            logger.info("Using comprehensive guardrail system")
            return
        
        # Fallback to old system if available
        if hasattr(self.guardrail_system, 'add_validation_rule'):
            def validate_earnings_output(result):
                return (
                    "investment_signal" in result and
                    result["investment_signal"] in ["bullish", "bearish", "neutral"] and
                    "signal_strength" in result and
                    0 <= result["signal_strength"] <= 1
                )
            
            def validate_thematic_output(result):
                return (
                    "top_themes" in result and
                    isinstance(result["top_themes"], list) and
                    len(result["top_themes"]) <= 10
                )
            
            self.guardrail_system.add_validation_rule(
                TaskCategory.EARNINGS_PREDICTION, validate_earnings_output
            )
            self.guardrail_system.add_validation_rule(
                TaskCategory.THEMATIC_IDENTIFICATION, validate_thematic_output
            )
    
    async def process_request(self, task_category: TaskCategory, input_data: Any, 
                            require_human_review: bool = False) -> ModelOutput:
        """Process request using appropriate model combination"""
        
        # Route to specialized model if available
        if task_category == TaskCategory.EARNINGS_PREDICTION and "earnings_analysis" in self.specialized_models:
            output = self.specialized_models["earnings_analysis"].predict(input_data)
        elif task_category == TaskCategory.THEMATIC_IDENTIFICATION and "thematic_identification" in self.specialized_models:
            output = self.specialized_models["thematic_identification"].predict(input_data)
        else:
            # Fall back to foundation model (simulated)
            output = await self._use_foundation_model(task_category, input_data)
        
        # Apply guardrails
        is_valid, issues = self.guardrail_system.validate_output(output)
        if not is_valid:
            logger.warning(f"Guardrail validation failed: {issues}")
            output.guardrail_passed = False
        
        # Submit for human review if required or if confidence is low
        if require_human_review or output.confidence < 0.6 or not output.guardrail_passed:
            review_id = self.human_loop_system.submit_for_review(output)
            logger.info(f"Submitted for human review: {review_id}")
        
        # Track performance
        self._track_performance(output)
        
        return output
    
    async def _use_foundation_model(self, task_category: TaskCategory, input_data: Any) -> ModelOutput:
        """Use foundation model for general reasoning (simulated)"""
        # This would integrate with actual foundation models like GPT-4
        
        # Simulate foundation model response
        await asyncio.sleep(0.1)  # Simulate API call
        
        result = {
            "analysis": f"Foundation model analysis for {task_category.value}",
            "recommendation": "Based on general knowledge and reasoning",
            "confidence_note": "Foundation model provides broad analysis"
        }
        
        return ModelOutput(
            result=result,
            confidence=0.7,
            model_type=ModelType.FOUNDATION,
            task_category=task_category,
            timestamp=datetime.now(),
            validation_score=0.8,
            guardrail_passed=True,
            metadata={"model": "foundation", "processing_time": 0.1}
        )
    
    def _track_performance(self, output: ModelOutput):
        """Track model performance for continuous improvement"""
        key = f"{output.task_category.value}_{output.model_type.value}"
        
        if key not in self.performance_tracker:
            self.performance_tracker[key] = {
                "total_requests": 0,
                "avg_confidence": 0,
                "avg_validation_score": 0,
                "guardrail_pass_rate": 0,
                "human_review_rate": 0
            }
        
        tracker = self.performance_tracker[key]
        tracker["total_requests"] += 1
        
        # Update running averages
        n = tracker["total_requests"]
        tracker["avg_confidence"] = ((n-1) * tracker["avg_confidence"] + output.confidence) / n
        tracker["avg_validation_score"] = ((n-1) * tracker["avg_validation_score"] + output.validation_score) / n
        tracker["guardrail_pass_rate"] = ((n-1) * tracker["guardrail_pass_rate"] + (1 if output.guardrail_passed else 0)) / n
        tracker["human_review_rate"] = ((n-1) * tracker["human_review_rate"] + (1 if output.human_reviewed else 0)) / n
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report"""
        return {
            "model_performance": self.performance_tracker,
            "specialized_models": {
                name: {
                    "training_data": model.training_data.__dict__ if model.training_data else None,
                    "performance_metrics": model.performance_metrics
                }
                for name, model in self.specialized_models.items()
            },
            "guardrail_effectiveness": {
                "total_validations": len(self.guardrail_system.validation_rules),
                "validation_rules": list(self.guardrail_system.validation_rules.keys())
            },
            "human_collaboration": {
                "pending_reviews": len(self.human_loop_system.pending_reviews),
                "completed_reviews": len(self.human_loop_system.expert_feedback),
                "collaboration_history_size": sum(len(history) for history in self.human_loop_system.collaboration_history.values())
            },
            "generated_at": datetime.now().isoformat()
        }
    
    def add_specialized_model(self, name: str, model: BaseSpecializedModel):
        """Add new specialized model to the system"""
        self.specialized_models[name] = model
        logger.info(f"Added specialized model: {name}")
    
    def update_model_performance(self, model_name: str, metrics: Dict):
        """Update performance metrics for a specific model"""
        if model_name in self.specialized_models:
            self.specialized_models[model_name].performance_metrics.update(metrics)
            logger.info(f"Updated performance metrics for {model_name}")


# Example usage and testing
async def demonstrate_hybrid_ai_system():
    """Demonstrate the hybrid AI system capabilities"""
    
    print("🤖 HYBRID AI ENGINE - BLACKROCK ALADDIN ARCHITECTURE")
    print("=" * 60)
    
    # Initialize the hybrid AI engine
    ai_engine = HybridAIEngine()
    
    # Test earnings call analysis
    print("\n📊 EARNINGS CALL ANALYSIS")
    print("-" * 30)
    
    sample_earnings_text = """
    We're pleased to report strong quarterly results with revenue growth of 15% year-over-year.
    Our digital transformation initiatives are showing excellent progress, and we're optimistic
    about the outlook for the next quarter. We expect continued growth in our cloud services
    division and are confident in our ability to maintain market leadership. However, we do
    see some headwinds from supply chain challenges and increased competition in certain segments.
    """
    
    earnings_output = await ai_engine.process_request(
        TaskCategory.EARNINGS_PREDICTION, 
        sample_earnings_text
    )
    
    print(f"Investment Signal: {earnings_output.result['investment_signal']}")
    print(f"Signal Strength: {earnings_output.result['signal_strength']:.2f}")
    print(f"Key Themes: {earnings_output.result['key_themes']}")
    print(f"Confidence: {earnings_output.confidence:.2f}")
    print(f"Validation Score: {earnings_output.validation_score:.2f}")
    print(f"Guardrails Passed: {earnings_output.guardrail_passed}")
    
    # Test thematic investment identification
    print("\n🎯 THEMATIC INVESTMENT ANALYSIS")
    print("-" * 35)
    
    sample_market_data = {
        "news": [
            {"content": "AI breakthrough in healthcare diagnostics shows promising results"},
            {"content": "Renewable energy adoption accelerates with new government incentives"},
            {"content": "Cybersecurity threats increase as remote work becomes permanent"}
        ],
        "patents": [],
        "regulations": [],
        "price_data": {}
    }
    
    thematic_output = await ai_engine.process_request(
        TaskCategory.THEMATIC_IDENTIFICATION,
        sample_market_data
    )
    
    print(f"Top Themes: {[theme['name'] for theme in thematic_output.result['top_themes']]}")
    print(f"Theme Strengths: {thematic_output.result['theme_strength']}")
    print(f"Investment Vehicles: {list(thematic_output.result['investment_vehicles'].keys())}")
    print(f"Confidence: {thematic_output.confidence:.2f}")
    print(f"Validation Score: {thematic_output.validation_score:.2f}")
    
    # Demonstrate human-in-the-loop
    print("\n👥 HUMAN-IN-THE-LOOP COLLABORATION")
    print("-" * 40)
    
    # Submit for expert review
    review_id = ai_engine.human_loop_system.submit_for_review(
        earnings_output, "portfolio_manager"
    )
    print(f"Submitted for expert review: {review_id}")
    
    # Simulate expert feedback
    expert_feedback = {
        "reviewer": "Senior Portfolio Manager",
        "rating": 8,
        "agreement": 0.85,
        "suggestions": ["Consider sector rotation impact", "Monitor regulatory changes"]
    }
    
    ai_engine.human_loop_system.provide_expert_feedback(review_id, expert_feedback)
    print(f"Expert feedback recorded: Rating {expert_feedback['rating']}/10")
    
    # Get performance report
    print("\n📈 SYSTEM PERFORMANCE REPORT")
    print("-" * 35)
    
    performance_report = ai_engine.get_performance_report()
    print(f"Specialized Models: {len(performance_report['specialized_models'])}")
    print(f"Pending Reviews: {performance_report['human_collaboration']['pending_reviews']}")
    print(f"Completed Reviews: {performance_report['human_collaboration']['completed_reviews']}")
    
    print("\n✅ HYBRID AI SYSTEM DEMONSTRATION COMPLETE")
    print("🏛️ Institutional-grade AI with human expertise integration")


if __name__ == "__main__":
    asyncio.run(demonstrate_hybrid_ai_system())