"""
Comprehensive Guardrail Engine
BlackRock Aladdin-inspired validation and safety system for AI outputs
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import json
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
import re
import hashlib
from abc import ABC, abstractmethod
import yfinance as yf
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from app.core.hybrid_ai_engine import ModelOutput, TaskCategory, ModelType

logger = logging.getLogger(__name__)


class GuardrailType(Enum):
    """Types of guardrail checks"""
    STRUCTURE_VALIDATION = "structure_validation"
    HALLUCINATION_DETECTION = "hallucination_detection"
    FACT_CHECKING = "fact_checking"
    ETHICAL_COMPLIANCE = "ethical_compliance"
    FINANCIAL_ACCURACY = "financial_accuracy"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    CONFIDENCE_VALIDATION = "confidence_validation"
    CONSISTENCY_CHECK = "consistency_check"


class ViolationSeverity(Enum):
    """Severity levels for guardrail violations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class GuardrailViolation:
    """Represents a guardrail violation"""
    violation_type: GuardrailType
    severity: ViolationSeverity
    description: str
    detected_value: Any
    expected_value: Any
    confidence: float
    recommendation: str
    timestamp: datetime


@dataclass
class GuardrailResult:
    """Result of guardrail validation"""
    passed: bool
    overall_score: float
    violations: List[GuardrailViolation]
    warnings: List[str]
    recommendations: List[str]
    processing_time: float
    timestamp: datetime
    metadata: Dict[str, Any]


class BaseGuardrail(ABC):
    """Abstract base class for guardrail implementations"""
    
    def __init__(self, name: str, guardrail_type: GuardrailType):
        self.name = name
        self.guardrail_type = guardrail_type
        self.enabled = True
        self.threshold = 0.8
        self.performance_metrics = {}
        
    @abstractmethod
    async def validate(self, output: ModelOutput, context: Dict[str, Any]) -> List[GuardrailViolation]:
        """Validate model output and return violations"""
        pass
    
    def set_threshold(self, threshold: float):
        """Set validation threshold"""
        self.threshold = max(0.0, min(1.0, threshold))
    
    def enable(self):
        """Enable this guardrail"""
        self.enabled = True
    
    def disable(self):
        """Disable this guardrail"""
        self.enabled = False


class StructureValidationGuardrail(BaseGuardrail):
    """Validates output structure and required fields"""
    
    def __init__(self):
        super().__init__("StructureValidator", GuardrailType.STRUCTURE_VALIDATION)
        
        # Define required fields for each task category
        self.required_fields = {
            TaskCategory.EARNINGS_PREDICTION: [
                "investment_signal", "signal_strength", "key_themes", "risk_factors"
            ],
            TaskCategory.THEMATIC_IDENTIFICATION: [
                "top_themes", "theme_strength", "investment_vehicles"
            ],
            TaskCategory.SENTIMENT_ANALYSIS: [
                "overall_sentiment", "sentiment_score", "confidence", "entity_sentiments"
            ],
            TaskCategory.RISK_ASSESSMENT: [
                "overall_risk_score", "risk_level", "confidence", "risk_factors"
            ]
        }
    
    async def validate(self, output: ModelOutput, context: Dict[str, Any]) -> List[GuardrailViolation]:
        """Validate output structure"""
        violations = []
        
        if not self.enabled:
            return violations
        
        try:
            # Check if result exists
            if not hasattr(output, 'result') or output.result is None:
                violations.append(GuardrailViolation(
                    violation_type=self.guardrail_type,
                    severity=ViolationSeverity.CRITICAL,
                    description="Output result is missing or None",
                    detected_value=None,
                    expected_value="Valid result object",
                    confidence=1.0,
                    recommendation="Ensure model generates valid output",
                    timestamp=datetime.now()
                ))
                return violations
            
            # Check required fields for task category
            required_fields = self.required_fields.get(output.task_category, [])
            
            for field in required_fields:
                if field not in output.result:
                    violations.append(GuardrailViolation(
                        violation_type=self.guardrail_type,
                        severity=ViolationSeverity.HIGH,
                        description=f"Required field '{field}' is missing",
                        detected_value="missing",
                        expected_value=field,
                        confidence=1.0,
                        recommendation=f"Ensure model output includes '{field}' field",
                        timestamp=datetime.now()
                    ))
            
            # Validate confidence range
            if hasattr(output, 'confidence'):
                if not (0 <= output.confidence <= 1):
                    violations.append(GuardrailViolation(
                        violation_type=self.guardrail_type,
                        severity=ViolationSeverity.MEDIUM,
                        description="Confidence score out of valid range",
                        detected_value=output.confidence,
                        expected_value="0.0 to 1.0",
                        confidence=1.0,
                        recommendation="Normalize confidence scores to 0-1 range",
                        timestamp=datetime.now()
                    ))
            
            # Validate timestamp
            if hasattr(output, 'timestamp'):
                if output.timestamp > datetime.now() + timedelta(minutes=5):
                    violations.append(GuardrailViolation(
                        violation_type=self.guardrail_type,
                        severity=ViolationSeverity.LOW,
                        description="Timestamp is in the future",
                        detected_value=output.timestamp,
                        expected_value="Current or past timestamp",
                        confidence=0.9,
                        recommendation="Use current timestamp for output generation",
                        timestamp=datetime.now()
                    ))
            
        except Exception as e:
            violations.append(GuardrailViolation(
                violation_type=self.guardrail_type,
                severity=ViolationSeverity.HIGH,
                description=f"Structure validation failed: {str(e)}",
                detected_value="validation_error",
                expected_value="successful_validation",
                confidence=0.8,
                recommendation="Review output structure and validation logic",
                timestamp=datetime.now()
            ))
        
        return violations


class HallucinationDetectionGuardrail(BaseGuardrail):
    """Detects potential hallucinations in model outputs"""
    
    def __init__(self):
        super().__init__("HallucinationDetector", GuardrailType.HALLUCINATION_DETECTION)
        self.knowledge_base = self._build_knowledge_base()
        self.suspicious_patterns = self._build_suspicious_patterns()
        
    async def validate(self, output: ModelOutput, context: Dict[str, Any]) -> List[GuardrailViolation]:
        """Detect potential hallucinations"""
        violations = []
        
        if not self.enabled:
            return violations
        
        try:
            # Check for impossible numerical values
            violations.extend(await self._check_impossible_values(output))
            
            # Check for suspicious patterns
            violations.extend(await self._check_suspicious_patterns(output))
            
            # Check against knowledge base
            violations.extend(await self._check_knowledge_base(output, context))
            
            # Check for internal consistency
            violations.extend(await self._check_internal_consistency(output))
            
        except Exception as e:
            logger.error(f"Hallucination detection failed: {e}")
            violations.append(GuardrailViolation(
                violation_type=self.guardrail_type,
                severity=ViolationSeverity.MEDIUM,
                description=f"Hallucination detection error: {str(e)}",
                detected_value="detection_error",
                expected_value="successful_detection",
                confidence=0.7,
                recommendation="Review hallucination detection logic",
                timestamp=datetime.now()
            ))
        
        return violations
    
    async def _check_impossible_values(self, output: ModelOutput) -> List[GuardrailViolation]:
        """Check for impossible numerical values"""
        violations = []
        
        def check_value(value, field_name, min_val=None, max_val=None):
            if isinstance(value, (int, float)):
                if abs(value) > 1e10:  # Unreasonably large numbers
                    violations.append(GuardrailViolation(
                        violation_type=self.guardrail_type,
                        severity=ViolationSeverity.HIGH,
                        description=f"Unreasonably large value in {field_name}",
                        detected_value=value,
                        expected_value=f"Reasonable numerical value",
                        confidence=0.95,
                        recommendation=f"Review calculation logic for {field_name}",
                        timestamp=datetime.now()
                    ))
                
                if min_val is not None and value < min_val:
                    violations.append(GuardrailViolation(
                        violation_type=self.guardrail_type,
                        severity=ViolationSeverity.MEDIUM,
                        description=f"Value below minimum threshold in {field_name}",
                        detected_value=value,
                        expected_value=f">= {min_val}",
                        confidence=0.9,
                        recommendation=f"Ensure {field_name} meets minimum requirements",
                        timestamp=datetime.now()
                    ))
                
                if max_val is not None and value > max_val:
                    violations.append(GuardrailViolation(
                        violation_type=self.guardrail_type,
                        severity=ViolationSeverity.MEDIUM,
                        description=f"Value above maximum threshold in {field_name}",
                        detected_value=value,
                        expected_value=f"<= {max_val}",
                        confidence=0.9,
                        recommendation=f"Ensure {field_name} stays within bounds",
                        timestamp=datetime.now()
                    ))
        
        # Check specific fields based on task category
        if output.task_category == TaskCategory.SENTIMENT_ANALYSIS:
            if "sentiment_score" in output.result:
                check_value(output.result["sentiment_score"], "sentiment_score", -1.0, 1.0)
        
        elif output.task_category == TaskCategory.RISK_ASSESSMENT:
            if "overall_risk_score" in output.result:
                check_value(output.result["overall_risk_score"], "overall_risk_score", 0.0, 1.0)
        
        elif output.task_category == TaskCategory.EARNINGS_PREDICTION:
            if "signal_strength" in output.result:
                check_value(output.result["signal_strength"], "signal_strength", 0.0, 1.0)
        
        return violations
    
    async def _check_suspicious_patterns(self, output: ModelOutput) -> List[GuardrailViolation]:
        """Check for suspicious patterns that might indicate hallucination"""
        violations = []
        
        # Convert output to string for pattern matching
        output_str = json.dumps(output.result, default=str).lower()
        
        for pattern, description in self.suspicious_patterns.items():
            if re.search(pattern, output_str, re.IGNORECASE):
                violations.append(GuardrailViolation(
                    violation_type=self.guardrail_type,
                    severity=ViolationSeverity.MEDIUM,
                    description=f"Suspicious pattern detected: {description}",
                    detected_value=pattern,
                    expected_value="Factual content",
                    confidence=0.7,
                    recommendation="Verify factual accuracy of the content",
                    timestamp=datetime.now()
                ))
        
        return violations
    
    async def _check_knowledge_base(self, output: ModelOutput, context: Dict[str, Any]) -> List[GuardrailViolation]:
        """Check output against known facts in knowledge base"""
        violations = []
        
        # Check for known impossible combinations
        if output.task_category == TaskCategory.SENTIMENT_ANALYSIS:
            sentiment = output.result.get("overall_sentiment", "")
            score = output.result.get("sentiment_score", 0)
            
            if sentiment == "bullish" and score < -0.1:
                violations.append(GuardrailViolation(
                    violation_type=self.guardrail_type,
                    severity=ViolationSeverity.HIGH,
                    description="Inconsistent sentiment signal and score",
                    detected_value=f"{sentiment} with score {score}",
                    expected_value="Consistent sentiment and score",
                    confidence=0.9,
                    recommendation="Ensure sentiment classification matches numerical score",
                    timestamp=datetime.now()
                ))
        
        return violations
    
    async def _check_internal_consistency(self, output: ModelOutput) -> List[GuardrailViolation]:
        """Check for internal consistency within the output"""
        violations = []
        
        # Check confidence consistency
        if hasattr(output, 'confidence') and hasattr(output, 'validation_score'):
            if output.confidence > 0.9 and output.validation_score < 0.5:
                violations.append(GuardrailViolation(
                    violation_type=self.guardrail_type,
                    severity=ViolationSeverity.MEDIUM,
                    description="High confidence with low validation score",
                    detected_value=f"confidence: {output.confidence}, validation: {output.validation_score}",
                    expected_value="Consistent confidence and validation scores",
                    confidence=0.8,
                    recommendation="Review confidence calibration",
                    timestamp=datetime.now()
                ))
        
        return violations
    
    def _build_knowledge_base(self) -> Dict[str, Any]:
        """Build knowledge base of known facts"""
        return {
            "market_facts": {
                "trading_hours": "9:30 AM - 4:00 PM EST",
                "max_daily_change": 0.2,  # 20% circuit breaker
                "min_price": 0.0001
            },
            "financial_ranges": {
                "sentiment_score": (-1.0, 1.0),
                "risk_score": (0.0, 1.0),
                "confidence": (0.0, 1.0),
                "correlation": (-1.0, 1.0)
            }
        }
    
    def _build_suspicious_patterns(self) -> Dict[str, str]:
        """Build patterns that might indicate hallucination"""
        return {
            r"i think|i believe|in my opinion": "Subjective language in factual analysis",
            r"definitely|absolutely|certainly.*100%": "Overconfident language",
            r"impossible|never|always.*will": "Absolute statements about uncertain events",
            r"secret|confidential.*information": "Claims of access to non-public information",
            r"guaranteed.*profit|risk-free": "Impossible financial promises",
            r"\$\d+\.\d{5,}": "Unrealistic precision in financial figures"
        }


class FactCheckingGuardrail(BaseGuardrail):
    """Fact-checks outputs against trusted financial data sources"""
    
    def __init__(self):
        super().__init__("FactChecker", GuardrailType.FACT_CHECKING)
        self.trusted_sources = self._initialize_trusted_sources()
        self.fact_cache = {}
        
    async def validate(self, output: ModelOutput, context: Dict[str, Any]) -> List[GuardrailViolation]:
        """Fact-check model output"""
        violations = []
        
        if not self.enabled:
            return violations
        
        try:
            # Extract symbols from context or output
            symbols = context.get("symbols", [])
            if not symbols and "symbols" in output.result:
                symbols = output.result["symbols"]
            
            # Fact-check financial data
            if symbols:
                violations.extend(await self._check_financial_facts(output, symbols))
            
            # Check market data consistency
            violations.extend(await self._check_market_consistency(output))
            
            # Verify numerical claims
            violations.extend(await self._verify_numerical_claims(output))
            
        except Exception as e:
            logger.error(f"Fact checking failed: {e}")
            violations.append(GuardrailViolation(
                violation_type=self.guardrail_type,
                severity=ViolationSeverity.MEDIUM,
                description=f"Fact checking error: {str(e)}",
                detected_value="fact_check_error",
                expected_value="successful_fact_check",
                confidence=0.7,
                recommendation="Review fact checking logic and data sources",
                timestamp=datetime.now()
            ))
        
        return violations
    
    async def _check_financial_facts(self, output: ModelOutput, symbols: List[str]) -> List[GuardrailViolation]:
        """Check financial facts against market data"""
        violations = []
        
        for symbol in symbols[:3]:  # Limit to 3 symbols to avoid rate limits
            try:
                # Get current market data
                market_data = await self._get_market_data(symbol)
                
                if market_data:
                    # Check price-related claims
                    violations.extend(self._verify_price_claims(output, symbol, market_data))
                    
                    # Check volume claims
                    violations.extend(self._verify_volume_claims(output, symbol, market_data))
                    
            except Exception as e:
                logger.warning(f"Failed to fact-check {symbol}: {e}")
        
        return violations
    
    async def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current market data for symbol"""
        
        cache_key = f"{symbol}_{datetime.now().strftime('%Y-%m-%d')}"
        
        if cache_key in self.fact_cache:
            return self.fact_cache[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="5d")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                volume = hist['Volume'].iloc[-1]
                
                market_data = {
                    "current_price": current_price,
                    "volume": volume,
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "sector": info.get("sector")
                }
                
                self.fact_cache[cache_key] = market_data
                return market_data
                
        except Exception as e:
            logger.warning(f"Failed to get market data for {symbol}: {e}")
        
        return None
    
    def _verify_price_claims(self, output: ModelOutput, symbol: str, market_data: Dict[str, Any]) -> List[GuardrailViolation]:
        """Verify price-related claims"""
        violations = []
        
        # This is a simplified example - in production, you'd have more sophisticated fact-checking
        current_price = market_data.get("current_price", 0)
        
        # Check for unrealistic price predictions
        output_str = json.dumps(output.result, default=str)
        
        # Look for specific price mentions
        price_pattern = r'\$(\d+(?:\.\d+)?)'
        price_matches = re.findall(price_pattern, output_str)
        
        for price_str in price_matches:
            try:
                mentioned_price = float(price_str)
                
                # Check if mentioned price is wildly different from current price
                if current_price > 0:
                    price_ratio = mentioned_price / current_price
                    
                    if price_ratio > 10 or price_ratio < 0.1:  # 10x difference
                        violations.append(GuardrailViolation(
                            violation_type=self.guardrail_type,
                            severity=ViolationSeverity.MEDIUM,
                            description=f"Mentioned price ${mentioned_price} significantly differs from current price ${current_price:.2f} for {symbol}",
                            detected_value=mentioned_price,
                            expected_value=f"Price within reasonable range of ${current_price:.2f}",
                            confidence=0.8,
                            recommendation="Verify price predictions against current market data",
                            timestamp=datetime.now()
                        ))
                        
            except ValueError:
                continue
        
        return violations
    
    def _verify_volume_claims(self, output: ModelOutput, symbol: str, market_data: Dict[str, Any]) -> List[GuardrailViolation]:
        """Verify volume-related claims"""
        violations = []
        
        # Simplified volume verification
        current_volume = market_data.get("volume", 0)
        
        # Check for volume-related claims in output
        output_str = json.dumps(output.result, default=str).lower()
        
        if "high volume" in output_str and current_volume < 100000:
            violations.append(GuardrailViolation(
                violation_type=self.guardrail_type,
                severity=ViolationSeverity.LOW,
                description=f"Claimed high volume but current volume is {current_volume:,} for {symbol}",
                detected_value="high volume claim",
                expected_value="Accurate volume description",
                confidence=0.6,
                recommendation="Verify volume claims against actual trading data",
                timestamp=datetime.now()
            ))
        
        return violations
    
    async def _check_market_consistency(self, output: ModelOutput) -> List[GuardrailViolation]:
        """Check for market consistency"""
        violations = []
        
        # Check for impossible market conditions
        output_str = json.dumps(output.result, default=str).lower()
        
        # Look for contradictory statements
        contradictions = [
            ("bull market", "bear market"),
            ("high volatility", "low volatility"),
            ("strong growth", "declining"),
            ("positive outlook", "negative outlook")
        ]
        
        for term1, term2 in contradictions:
            if term1 in output_str and term2 in output_str:
                violations.append(GuardrailViolation(
                    violation_type=self.guardrail_type,
                    severity=ViolationSeverity.MEDIUM,
                    description=f"Contradictory terms found: '{term1}' and '{term2}'",
                    detected_value=f"{term1}, {term2}",
                    expected_value="Consistent market description",
                    confidence=0.7,
                    recommendation="Ensure consistent market analysis throughout output",
                    timestamp=datetime.now()
                ))
        
        return violations
    
    async def _verify_numerical_claims(self, output: ModelOutput) -> List[GuardrailViolation]:
        """Verify numerical claims for reasonableness"""
        violations = []
        
        # Check percentage claims
        output_str = json.dumps(output.result, default=str)
        percentage_pattern = r'(\d+(?:\.\d+)?)%'
        percentages = re.findall(percentage_pattern, output_str)
        
        for pct_str in percentages:
            try:
                percentage = float(pct_str)
                
                # Check for unrealistic percentages
                if percentage > 1000:  # 1000%+ seems unrealistic for most financial contexts
                    violations.append(GuardrailViolation(
                        violation_type=self.guardrail_type,
                        severity=ViolationSeverity.MEDIUM,
                        description=f"Unrealistic percentage: {percentage}%",
                        detected_value=f"{percentage}%",
                        expected_value="Realistic percentage value",
                        confidence=0.8,
                        recommendation="Verify percentage calculations and context",
                        timestamp=datetime.now()
                    ))
                    
            except ValueError:
                continue
        
        return violations
    
    def _initialize_trusted_sources(self) -> Dict[str, str]:
        """Initialize trusted data sources"""
        return {
            "market_data": "Yahoo Finance, Bloomberg API",
            "economic_data": "Federal Reserve, Bureau of Labor Statistics",
            "company_data": "SEC EDGAR, Company filings",
            "news_sources": "Reuters, Bloomberg, Wall Street Journal"
        }


class EthicalComplianceGuardrail(BaseGuardrail):
    """Ensures outputs comply with ethical guidelines and regulations"""
    
    def __init__(self):
        super().__init__("EthicalCompliance", GuardrailType.ETHICAL_COMPLIANCE)
        self.ethical_guidelines = self._load_ethical_guidelines()
        self.prohibited_content = self._load_prohibited_content()
        
    async def validate(self, output: ModelOutput, context: Dict[str, Any]) -> List[GuardrailViolation]:
        """Check ethical compliance"""
        violations = []
        
        if not self.enabled:
            return violations
        
        try:
            # Check for prohibited content
            violations.extend(await self._check_prohibited_content(output))
            
            # Check fairness and bias
            violations.extend(await self._check_fairness_bias(output))
            
            # Check regulatory compliance
            violations.extend(await self._check_regulatory_compliance(output))
            
            # Check client best interest
            violations.extend(await self._check_client_interest(output))
            
        except Exception as e:
            logger.error(f"Ethical compliance check failed: {e}")
            violations.append(GuardrailViolation(
                violation_type=self.guardrail_type,
                severity=ViolationSeverity.HIGH,
                description=f"Ethical compliance check error: {str(e)}",
                detected_value="compliance_check_error",
                expected_value="successful_compliance_check",
                confidence=0.8,
                recommendation="Review ethical compliance checking logic",
                timestamp=datetime.now()
            ))
        
        return violations
    
    async def _check_prohibited_content(self, output: ModelOutput) -> List[GuardrailViolation]:
        """Check for prohibited content"""
        violations = []
        
        output_str = json.dumps(output.result, default=str).lower()
        
        for prohibited_term, description in self.prohibited_content.items():
            if prohibited_term in output_str:
                violations.append(GuardrailViolation(
                    violation_type=self.guardrail_type,
                    severity=ViolationSeverity.HIGH,
                    description=f"Prohibited content detected: {description}",
                    detected_value=prohibited_term,
                    expected_value="Compliant content",
                    confidence=0.9,
                    recommendation=f"Remove or rephrase content related to {prohibited_term}",
                    timestamp=datetime.now()
                ))
        
        return violations
    
    async def _check_fairness_bias(self, output: ModelOutput) -> List[GuardrailViolation]:
        """Check for fairness and bias issues"""
        violations = []
        
        # Check for discriminatory language
        output_str = json.dumps(output.result, default=str).lower()
        
        bias_indicators = [
            "only suitable for", "not appropriate for", "exclusively for",
            "men only", "women only", "young investors only", "old investors"
        ]
        
        for indicator in bias_indicators:
            if indicator in output_str:
                violations.append(GuardrailViolation(
                    violation_type=self.guardrail_type,
                    severity=ViolationSeverity.MEDIUM,
                    description=f"Potential bias detected: {indicator}",
                    detected_value=indicator,
                    expected_value="Inclusive, non-discriminatory language",
                    confidence=0.7,
                    recommendation="Use inclusive language that doesn't discriminate",
                    timestamp=datetime.now()
                ))
        
        return violations
    
    async def _check_regulatory_compliance(self, output: ModelOutput) -> List[GuardrailViolation]:
        """Check regulatory compliance"""
        violations = []
        
        output_str = json.dumps(output.result, default=str).lower()
        
        # Check for required disclaimers
        if "investment" in output_str or "recommendation" in output_str:
            disclaimer_terms = ["risk", "past performance", "not guarantee"]
            
            missing_disclaimers = []
            for term in disclaimer_terms:
                if term not in output_str:
                    missing_disclaimers.append(term)
            
            if missing_disclaimers:
                violations.append(GuardrailViolation(
                    violation_type=self.guardrail_type,
                    severity=ViolationSeverity.MEDIUM,
                    description=f"Missing regulatory disclaimers: {', '.join(missing_disclaimers)}",
                    detected_value="missing_disclaimers",
                    expected_value="Complete regulatory disclaimers",
                    confidence=0.8,
                    recommendation="Include appropriate risk disclaimers in investment advice",
                    timestamp=datetime.now()
                ))
        
        return violations
    
    async def _check_client_interest(self, output: ModelOutput) -> List[GuardrailViolation]:
        """Check if output serves client's best interest"""
        violations = []
        
        output_str = json.dumps(output.result, default=str).lower()
        
        # Check for potential conflicts of interest
        conflict_indicators = [
            "we recommend our own", "our firm's product", "exclusive to us",
            "higher fees", "commission-based"
        ]
        
        for indicator in conflict_indicators:
            if indicator in output_str:
                violations.append(GuardrailViolation(
                    violation_type=self.guardrail_type,
                    severity=ViolationSeverity.HIGH,
                    description=f"Potential conflict of interest: {indicator}",
                    detected_value=indicator,
                    expected_value="Client-focused recommendations",
                    confidence=0.8,
                    recommendation="Ensure recommendations prioritize client's best interest",
                    timestamp=datetime.now()
                ))
        
        return violations
    
    def _load_ethical_guidelines(self) -> Dict[str, str]:
        """Load ethical guidelines"""
        return {
            "fairness": "Provide equal treatment and opportunities regardless of demographics",
            "transparency": "Be clear about limitations, assumptions, and potential conflicts",
            "accountability": "Take responsibility for recommendations and their consequences",
            "privacy": "Protect client data and respect privacy rights",
            "beneficence": "Act in the client's best interest",
            "non_maleficence": "Do no harm through recommendations or advice"
        }
    
    def _load_prohibited_content(self) -> Dict[str, str]:
        """Load prohibited content patterns"""
        return {
            "guaranteed profit": "Unrealistic profit guarantees",
            "risk-free investment": "False claims about risk-free investments",
            "insider information": "Claims of access to non-public information",
            "market manipulation": "Content that could manipulate markets",
            "discriminatory": "Discriminatory language or recommendations"
        }


class ComprehensiveGuardrailEngine:
    """Main guardrail engine that orchestrates all validation checks"""
    
    def __init__(self):
        self.guardrails = {
            GuardrailType.STRUCTURE_VALIDATION: StructureValidationGuardrail(),
            GuardrailType.HALLUCINATION_DETECTION: HallucinationDetectionGuardrail(),
            GuardrailType.FACT_CHECKING: FactCheckingGuardrail(),
            GuardrailType.ETHICAL_COMPLIANCE: EthicalComplianceGuardrail()
        }
        
        self.performance_metrics = {}
        self.violation_history = defaultdict(list)
        self.processing_stats = {
            "total_validations": 0,
            "total_violations": 0,
            "avg_processing_time": 0.0
        }
        
        logger.info("Comprehensive Guardrail Engine initialized")
    
    async def validate_output(self, output: ModelOutput, 
                            context: Dict[str, Any] = None) -> GuardrailResult:
        """Comprehensive validation of model output"""
        
        start_time = datetime.now()
        context = context or {}
        
        all_violations = []
        warnings = []
        recommendations = []
        
        try:
            # Run all enabled guardrails
            for guardrail_type, guardrail in self.guardrails.items():
                if guardrail.enabled:
                    try:
                        violations = await guardrail.validate(output, context)
                        all_violations.extend(violations)
                        
                        # Update guardrail performance metrics
                        self._update_guardrail_metrics(guardrail_type, len(violations))
                        
                    except Exception as e:
                        logger.error(f"Guardrail {guardrail_type.value} failed: {e}")
                        warnings.append(f"Guardrail {guardrail_type.value} encountered an error")
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(all_violations)
            
            # Determine if validation passed
            passed = len([v for v in all_violations if v.severity in [ViolationSeverity.HIGH, ViolationSeverity.CRITICAL]]) == 0
            
            # Generate recommendations
            recommendations = self._generate_recommendations(all_violations)
            
            # Store violation history
            self._store_violation_history(output, all_violations)
            
            # Update processing stats
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_processing_stats(processing_time, len(all_violations))
            
            result = GuardrailResult(
                passed=passed,
                overall_score=overall_score,
                violations=all_violations,
                warnings=warnings,
                recommendations=recommendations,
                processing_time=processing_time,
                timestamp=datetime.now(),
                metadata={
                    "guardrails_run": len([g for g in self.guardrails.values() if g.enabled]),
                    "total_checks": sum(1 for g in self.guardrails.values() if g.enabled),
                    "violation_breakdown": self._get_violation_breakdown(all_violations)
                }
            )
            
            logger.info(f"Guardrail validation completed: {len(all_violations)} violations, "
                       f"score: {overall_score:.2f}, passed: {passed}")
            
            return result
            
        except Exception as e:
            logger.error(f"Guardrail engine validation failed: {e}")
            
            # Return error result
            return GuardrailResult(
                passed=False,
                overall_score=0.0,
                violations=[GuardrailViolation(
                    violation_type=GuardrailType.STRUCTURE_VALIDATION,
                    severity=ViolationSeverity.CRITICAL,
                    description=f"Guardrail engine error: {str(e)}",
                    detected_value="engine_error",
                    expected_value="successful_validation",
                    confidence=1.0,
                    recommendation="Review guardrail engine configuration",
                    timestamp=datetime.now()
                )],
                warnings=["Guardrail engine encountered a critical error"],
                recommendations=["Review and fix guardrail engine"],
                processing_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )
    
    def _calculate_overall_score(self, violations: List[GuardrailViolation]) -> float:
        """Calculate overall validation score"""
        
        if not violations:
            return 1.0
        
        # Weight violations by severity
        severity_weights = {
            ViolationSeverity.LOW: 0.1,
            ViolationSeverity.MEDIUM: 0.3,
            ViolationSeverity.HIGH: 0.7,
            ViolationSeverity.CRITICAL: 1.0
        }
        
        total_penalty = sum(severity_weights.get(v.severity, 0.5) for v in violations)
        max_possible_penalty = len(violations) * 1.0  # All critical violations
        
        # Calculate score (1.0 = perfect, 0.0 = all critical violations)
        if max_possible_penalty > 0:
            score = 1.0 - (total_penalty / max_possible_penalty)
            return max(0.0, score)
        
        return 1.0
    
    def _generate_recommendations(self, violations: List[GuardrailViolation]) -> List[str]:
        """Generate recommendations based on violations"""
        
        recommendations = []
        
        # Group violations by type
        violation_groups = defaultdict(list)
        for violation in violations:
            violation_groups[violation.violation_type].append(violation)
        
        # Generate type-specific recommendations
        for violation_type, type_violations in violation_groups.items():
            if violation_type == GuardrailType.STRUCTURE_VALIDATION:
                recommendations.append(f"Fix {len(type_violations)} structure validation issues")
            elif violation_type == GuardrailType.HALLUCINATION_DETECTION:
                recommendations.append(f"Address {len(type_violations)} potential hallucinations")
            elif violation_type == GuardrailType.FACT_CHECKING:
                recommendations.append(f"Verify {len(type_violations)} factual claims")
            elif violation_type == GuardrailType.ETHICAL_COMPLIANCE:
                recommendations.append(f"Resolve {len(type_violations)} ethical compliance issues")
        
        # Add severity-based recommendations
        critical_violations = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
        if critical_violations:
            recommendations.insert(0, f"URGENT: Address {len(critical_violations)} critical violations immediately")
        
        high_violations = [v for v in violations if v.severity == ViolationSeverity.HIGH]
        if high_violations:
            recommendations.append(f"High priority: Fix {len(high_violations)} high-severity issues")
        
        return recommendations
    
    def _get_violation_breakdown(self, violations: List[GuardrailViolation]) -> Dict[str, int]:
        """Get breakdown of violations by type and severity"""
        
        breakdown = {
            "by_type": defaultdict(int),
            "by_severity": defaultdict(int)
        }
        
        for violation in violations:
            breakdown["by_type"][violation.violation_type.value] += 1
            breakdown["by_severity"][violation.severity.value] += 1
        
        return dict(breakdown)
    
    def _update_guardrail_metrics(self, guardrail_type: GuardrailType, violation_count: int):
        """Update performance metrics for individual guardrails"""
        
        if guardrail_type not in self.performance_metrics:
            self.performance_metrics[guardrail_type] = {
                "total_runs": 0,
                "total_violations": 0,
                "avg_violations": 0.0,
                "last_run": None
            }
        
        metrics = self.performance_metrics[guardrail_type]
        metrics["total_runs"] += 1
        metrics["total_violations"] += violation_count
        metrics["avg_violations"] = metrics["total_violations"] / metrics["total_runs"]
        metrics["last_run"] = datetime.now()
    
    def _store_violation_history(self, output: ModelOutput, violations: List[GuardrailViolation]):
        """Store violation history for analysis"""
        
        history_entry = {
            "timestamp": datetime.now(),
            "task_category": output.task_category.value,
            "model_type": output.model_type.value,
            "violation_count": len(violations),
            "violations": [
                {
                    "type": v.violation_type.value,
                    "severity": v.severity.value,
                    "description": v.description
                }
                for v in violations
            ]
        }
        
        self.violation_history[output.task_category].append(history_entry)
        
        # Keep only last 100 entries per task category
        if len(self.violation_history[output.task_category]) > 100:
            self.violation_history[output.task_category] = self.violation_history[output.task_category][-100:]
    
    def _update_processing_stats(self, processing_time: float, violation_count: int):
        """Update overall processing statistics"""
        
        self.processing_stats["total_validations"] += 1
        self.processing_stats["total_violations"] += violation_count
        
        # Update average processing time
        total_validations = self.processing_stats["total_validations"]
        current_avg = self.processing_stats["avg_processing_time"]
        self.processing_stats["avg_processing_time"] = (
            (current_avg * (total_validations - 1) + processing_time) / total_validations
        )
    
    def configure_guardrail(self, guardrail_type: GuardrailType, 
                          enabled: bool = None, threshold: float = None):
        """Configure individual guardrail settings"""
        
        if guardrail_type in self.guardrails:
            guardrail = self.guardrails[guardrail_type]
            
            if enabled is not None:
                if enabled:
                    guardrail.enable()
                else:
                    guardrail.disable()
            
            if threshold is not None:
                guardrail.set_threshold(threshold)
            
            logger.info(f"Configured {guardrail_type.value}: enabled={guardrail.enabled}, "
                       f"threshold={guardrail.threshold}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        
        report = {
            "overall_stats": self.processing_stats.copy(),
            "guardrail_performance": {},
            "violation_trends": {},
            "recommendations": []
        }
        
        # Guardrail performance
        for guardrail_type, metrics in self.performance_metrics.items():
            report["guardrail_performance"][guardrail_type.value] = metrics.copy()
            if metrics["last_run"]:
                report["guardrail_performance"][guardrail_type.value]["last_run"] = metrics["last_run"].isoformat()
        
        # Violation trends
        for task_category, history in self.violation_history.items():
            if history:
                recent_violations = [entry["violation_count"] for entry in history[-10:]]  # Last 10
                report["violation_trends"][task_category.value] = {
                    "recent_avg": sum(recent_violations) / len(recent_violations),
                    "total_entries": len(history),
                    "trend": "improving" if len(recent_violations) > 1 and recent_violations[-1] < recent_violations[0] else "stable"
                }
        
        # Generate recommendations
        if self.processing_stats["total_validations"] > 0:
            avg_violations = self.processing_stats["total_violations"] / self.processing_stats["total_validations"]
            
            if avg_violations > 2:
                report["recommendations"].append("High violation rate detected - review model training")
            
            if self.processing_stats["avg_processing_time"] > 1.0:
                report["recommendations"].append("Slow guardrail processing - optimize validation logic")
        
        return report
    
    def get_violation_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get violation summary for specified time period"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        summary = {
            "period_days": days,
            "total_violations": 0,
            "by_type": defaultdict(int),
            "by_severity": defaultdict(int),
            "by_task_category": defaultdict(int),
            "top_violations": []
        }
        
        all_violations = []
        
        for task_category, history in self.violation_history.items():
            for entry in history:
                if entry["timestamp"] >= cutoff_date:
                    summary["total_violations"] += entry["violation_count"]
                    summary["by_task_category"][task_category.value] += entry["violation_count"]
                    
                    for violation in entry["violations"]:
                        summary["by_type"][violation["type"]] += 1
                        summary["by_severity"][violation["severity"]] += 1
                        all_violations.append(violation)
        
        # Get top violation types
        violation_counts = defaultdict(int)
        for violation in all_violations:
            key = f"{violation['type']} - {violation['description'][:50]}..."
            violation_counts[key] += 1
        
        summary["top_violations"] = sorted(
            violation_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return dict(summary)
    
    async def batch_validate(self, outputs: List[ModelOutput], 
                           context: Dict[str, Any] = None) -> List[GuardrailResult]:
        """Validate multiple outputs in batch"""
        
        results = []
        
        for output in outputs:
            try:
                result = await self.validate_output(output, context)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch validation failed for output: {e}")
                # Add error result
                results.append(GuardrailResult(
                    passed=False,
                    overall_score=0.0,
                    violations=[],
                    warnings=[f"Validation error: {str(e)}"],
                    recommendations=["Review output and retry validation"],
                    processing_time=0.0,
                    timestamp=datetime.now(),
                    metadata={"error": str(e)}
                ))
        
        return results
    
    def reset_metrics(self):
        """Reset all performance metrics and history"""
        
        self.performance_metrics.clear()
        self.violation_history.clear()
        self.processing_stats = {
            "total_validations": 0,
            "total_violations": 0,
            "avg_processing_time": 0.0
        }
        
        logger.info("Guardrail engine metrics reset")


# Factory function for creating guardrail engine
def create_guardrail_engine() -> ComprehensiveGuardrailEngine:
    """Factory function to create guardrail engine instance"""
    return ComprehensiveGuardrailEngine()


# Utility functions for guardrail validation
async def validate_model_output(output: ModelOutput, context: Dict[str, Any] = None) -> GuardrailResult:
    """Utility function for quick output validation"""
    
    engine = create_guardrail_engine()
    return await engine.validate_output(output, context)


def check_output_safety(output: ModelOutput) -> bool:
    """Quick safety check for model output"""
    
    try:
        # Basic safety checks
        if not hasattr(output, 'result') or output.result is None:
            return False
        
        if not hasattr(output, 'confidence') or not (0 <= output.confidence <= 1):
            return False
        
        # Check for obvious issues
        output_str = json.dumps(output.result, default=str).lower()
        
        dangerous_patterns = [
            "guaranteed profit", "risk-free", "insider information",
            "market manipulation", "illegal"
        ]
        
        for pattern in dangerous_patterns:
            if pattern in output_str:
                return False
        
        return True
        
    except Exception:
        return False