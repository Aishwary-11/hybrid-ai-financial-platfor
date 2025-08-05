"""
Continuous Evaluation and Testing Pipeline
BlackRock Aladdin-inspired automated testing with LLM judges and performance monitoring
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
import hashlib
from abc import ABC, abstractmethod
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from app.core.hybrid_ai_engine import ModelOutput, TaskCategory, ModelType, HybridAIEngine

logger = logging.getLogger(__name__)


class TestType(Enum):
    """Types of evaluation tests"""
    ACCURACY_TEST = "accuracy_test"
    CONSISTENCY_TEST = "consistency_test"
    REGRESSION_TEST = "regression_test"
    PERFORMANCE_TEST = "performance_test"
    BIAS_TEST = "bias_test"
    ROBUSTNESS_TEST = "robustness_test"
    HALLUCINATION_TEST = "hallucination_test"
    ETHICAL_TEST = "ethical_test"


class TestResult(Enum):
    """Test result outcomes"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class TestCase:
    """Individual test case"""
    test_id: str
    test_type: TestType
    task_category: TaskCategory
    input_data: Dict[str, Any]
    expected_output: Optional[Dict[str, Any]]
    ground_truth: Optional[Any]
    test_description: str
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class EvaluationResult:
    """Result of a single evaluation"""
    test_case_id: str
    test_type: TestType
    result: TestResult
    score: float
    confidence: float
    details: Dict[str, Any]
    execution_time: float
    timestamp: datetime
    judge_feedback: Optional[str]
    recommendations: List[str]


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a model"""
    model_id: str
    task_category: TaskCategory
    accuracy_score: float
    consistency_score: float
    bias_score: float
    robustness_score: float
    overall_score: float
    test_count: int
    pass_rate: float
    avg_execution_time: float
    last_evaluation: datetime
    trend: str  # "improving", "stable", "declining"


class LLMJudge(ABC):
    """Abstract base class for LLM judges"""
    
    def __init__(self, judge_id: str, specialization: TaskCategory):
        self.judge_id = judge_id
        self.specialization = specialization
        self.performance_history = []
        
    @abstractmethod
    async def evaluate(self, test_case: TestCase, model_output: ModelOutput) -> EvaluationResult:
        """Evaluate model output against test case"""
        pass
    
    @abstractmethod
    def get_evaluation_criteria(self) -> Dict[str, str]:
        """Get evaluation criteria for this judge"""
        pass


class AccuracyJudge(LLMJudge):
    """Judge for evaluating accuracy of model outputs"""
    
    def __init__(self, specialization: TaskCategory):
        super().__init__(f"accuracy_judge_{specialization.value}", specialization)
        self.evaluation_criteria = self._build_accuracy_criteria()
        
    async def evaluate(self, test_case: TestCase, model_output: ModelOutput) -> EvaluationResult:
        """Evaluate accuracy of model output"""
        
        start_time = datetime.now()
        
        try:
            # Extract key components for evaluation
            predicted = model_output.result
            expected = test_case.expected_output
            ground_truth = test_case.ground_truth
            
            # Calculate accuracy score based on task category
            accuracy_score = await self._calculate_accuracy_score(
                predicted, expected, ground_truth, test_case.task_category
            )
            
            # Generate judge feedback
            judge_feedback = await self._generate_feedback(
                predicted, expected, accuracy_score, test_case
            )
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                accuracy_score, predicted, expected
            )
            
            # Determine result
            result = TestResult.PASS if accuracy_score >= 0.7 else TestResult.FAIL
            if 0.5 <= accuracy_score < 0.7:
                result = TestResult.WARNING
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return EvaluationResult(
                test_case_id=test_case.test_id,
                test_type=TestType.ACCURACY_TEST,
                result=result,
                score=accuracy_score,
                confidence=model_output.confidence,
                details={
                    "predicted_vs_expected": self._compare_outputs(predicted, expected),
                    "accuracy_breakdown": self._get_accuracy_breakdown(predicted, expected, test_case.task_category),
                    "ground_truth_comparison": self._compare_with_ground_truth(predicted, ground_truth) if ground_truth else None
                },
                execution_time=execution_time,
                timestamp=datetime.now(),
                judge_feedback=judge_feedback,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Accuracy evaluation failed: {e}")
            return EvaluationResult(
                test_case_id=test_case.test_id,
                test_type=TestType.ACCURACY_TEST,
                result=TestResult.ERROR,
                score=0.0,
                confidence=0.0,
                details={"error": str(e)},
                execution_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now(),
                judge_feedback=f"Evaluation failed: {str(e)}",
                recommendations=["Review test case and model output format"]
            )
    
    async def _calculate_accuracy_score(self, predicted: Dict[str, Any], 
                                      expected: Optional[Dict[str, Any]], 
                                      ground_truth: Optional[Any],
                                      task_category: TaskCategory) -> float:
        """Calculate accuracy score based on task category"""
        
        if task_category == TaskCategory.SENTIMENT_ANALYSIS:
            return await self._calculate_sentiment_accuracy(predicted, expected, ground_truth)
        elif task_category == TaskCategory.EARNINGS_PREDICTION:
            return await self._calculate_earnings_accuracy(predicted, expected, ground_truth)
        elif task_category == TaskCategory.RISK_ASSESSMENT:
            return await self._calculate_risk_accuracy(predicted, expected, ground_truth)
        elif task_category == TaskCategory.THEMATIC_IDENTIFICATION:
            return await self._calculate_thematic_accuracy(predicted, expected, ground_truth)
        else:
            return await self._calculate_general_accuracy(predicted, expected, ground_truth)
    
    async def _calculate_sentiment_accuracy(self, predicted: Dict[str, Any], 
                                          expected: Optional[Dict[str, Any]], 
                                          ground_truth: Optional[Any]) -> float:
        """Calculate sentiment analysis accuracy"""
        
        score = 0.0
        components = 0
        
        # Check sentiment signal accuracy
        if expected and "overall_sentiment" in predicted and "overall_sentiment" in expected:
            if predicted["overall_sentiment"] == expected["overall_sentiment"]:
                score += 0.4
            components += 0.4
        
        # Check sentiment score accuracy
        if expected and "sentiment_score" in predicted and "sentiment_score" in expected:
            pred_score = predicted["sentiment_score"]
            exp_score = expected["sentiment_score"]
            
            # Calculate normalized difference
            diff = abs(pred_score - exp_score)
            accuracy = max(0, 1 - diff / 2)  # Max difference is 2 (-1 to 1)
            score += accuracy * 0.4
            components += 0.4
        
        # Check confidence calibration
        if "confidence" in predicted:
            confidence = predicted["confidence"]
            # Well-calibrated confidence should be reasonable
            if 0.3 <= confidence <= 0.9:
                score += 0.2
            components += 0.2
        
        return score / components if components > 0 else 0.0
    
    async def _calculate_earnings_accuracy(self, predicted: Dict[str, Any], 
                                         expected: Optional[Dict[str, Any]], 
                                         ground_truth: Optional[Any]) -> float:
        """Calculate earnings analysis accuracy"""
        
        score = 0.0
        components = 0
        
        # Check investment signal accuracy
        if expected and "investment_signal" in predicted and "investment_signal" in expected:
            if predicted["investment_signal"] == expected["investment_signal"]:
                score += 0.5
            components += 0.5
        
        # Check signal strength accuracy
        if expected and "signal_strength" in predicted and "signal_strength" in expected:
            pred_strength = predicted["signal_strength"]
            exp_strength = expected["signal_strength"]
            
            diff = abs(pred_strength - exp_strength)
            accuracy = max(0, 1 - diff)
            score += accuracy * 0.3
            components += 0.3
        
        # Check key themes presence
        if expected and "key_themes" in predicted and "key_themes" in expected:
            pred_themes = set(predicted["key_themes"])
            exp_themes = set(expected["key_themes"])
            
            if exp_themes:
                overlap = len(pred_themes.intersection(exp_themes))
                theme_accuracy = overlap / len(exp_themes)
                score += theme_accuracy * 0.2
            components += 0.2
        
        return score / components if components > 0 else 0.0
    
    async def _calculate_risk_accuracy(self, predicted: Dict[str, Any], 
                                     expected: Optional[Dict[str, Any]], 
                                     ground_truth: Optional[Any]) -> float:
        """Calculate risk assessment accuracy"""
        
        score = 0.0
        components = 0
        
        # Check overall risk score accuracy
        if expected and "overall_risk_score" in predicted and "overall_risk_score" in expected:
            pred_risk = predicted["overall_risk_score"]
            exp_risk = expected["overall_risk_score"]
            
            diff = abs(pred_risk - exp_risk)
            accuracy = max(0, 1 - diff)
            score += accuracy * 0.4
            components += 0.4
        
        # Check risk level classification
        if expected and "risk_level" in predicted and "risk_level" in expected:
            if predicted["risk_level"] == expected["risk_level"]:
                score += 0.3
            components += 0.3
        
        # Check risk factors identification
        if expected and "risk_factors" in predicted and "risk_factors" in expected:
            pred_factors = len(predicted["risk_factors"])
            exp_factors = len(expected["risk_factors"])
            
            if exp_factors > 0:
                factor_accuracy = min(pred_factors / exp_factors, 1.0)
                score += factor_accuracy * 0.3
            components += 0.3
        
        return score / components if components > 0 else 0.0
    
    async def _calculate_thematic_accuracy(self, predicted: Dict[str, Any], 
                                         expected: Optional[Dict[str, Any]], 
                                         ground_truth: Optional[Any]) -> float:
        """Calculate thematic identification accuracy"""
        
        score = 0.0
        components = 0
        
        # Check top themes accuracy
        if expected and "top_themes" in predicted and "top_themes" in expected:
            pred_themes = [theme["name"] for theme in predicted["top_themes"][:3]]
            exp_themes = [theme["name"] for theme in expected["top_themes"][:3]]
            
            overlap = len(set(pred_themes).intersection(set(exp_themes)))
            if len(exp_themes) > 0:
                theme_accuracy = overlap / len(exp_themes)
                score += theme_accuracy * 0.6
            components += 0.6
        
        # Check investment vehicles suggestions
        if expected and "investment_vehicles" in predicted and "investment_vehicles" in expected:
            pred_vehicles = set(predicted["investment_vehicles"].keys())
            exp_vehicles = set(expected["investment_vehicles"].keys())
            
            if exp_vehicles:
                vehicle_accuracy = len(pred_vehicles.intersection(exp_vehicles)) / len(exp_vehicles)
                score += vehicle_accuracy * 0.4
            components += 0.4
        
        return score / components if components > 0 else 0.0
    
    async def _calculate_general_accuracy(self, predicted: Dict[str, Any], 
                                        expected: Optional[Dict[str, Any]], 
                                        ground_truth: Optional[Any]) -> float:
        """Calculate general accuracy for unknown task categories"""
        
        if not expected:
            return 0.5  # Neutral score when no expected output
        
        # Simple field matching
        matching_fields = 0
        total_fields = 0
        
        for key, exp_value in expected.items():
            total_fields += 1
            if key in predicted:
                pred_value = predicted[key]
                
                if isinstance(exp_value, str) and isinstance(pred_value, str):
                    if exp_value.lower() == pred_value.lower():
                        matching_fields += 1
                elif isinstance(exp_value, (int, float)) and isinstance(pred_value, (int, float)):
                    # Allow 10% tolerance for numerical values
                    if abs(exp_value - pred_value) <= abs(exp_value * 0.1):
                        matching_fields += 1
                elif exp_value == pred_value:
                    matching_fields += 1
        
        return matching_fields / total_fields if total_fields > 0 else 0.0
    
    async def _generate_feedback(self, predicted: Dict[str, Any], 
                               expected: Optional[Dict[str, Any]], 
                               accuracy_score: float,
                               test_case: TestCase) -> str:
        """Generate judge feedback"""
        
        feedback_parts = []
        
        if accuracy_score >= 0.8:
            feedback_parts.append("Excellent accuracy - model output closely matches expectations")
        elif accuracy_score >= 0.6:
            feedback_parts.append("Good accuracy - minor deviations from expected output")
        elif accuracy_score >= 0.4:
            feedback_parts.append("Moderate accuracy - several discrepancies identified")
        else:
            feedback_parts.append("Poor accuracy - significant deviations from expected output")
        
        # Add specific feedback based on task category
        if test_case.task_category == TaskCategory.SENTIMENT_ANALYSIS:
            if expected and "overall_sentiment" in predicted and "overall_sentiment" in expected:
                if predicted["overall_sentiment"] != expected["overall_sentiment"]:
                    feedback_parts.append(f"Sentiment classification mismatch: predicted '{predicted['overall_sentiment']}', expected '{expected['overall_sentiment']}'")
        
        return ". ".join(feedback_parts)
    
    async def _generate_recommendations(self, accuracy_score: float, 
                                      predicted: Dict[str, Any], 
                                      expected: Optional[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for improvement"""
        
        recommendations = []
        
        if accuracy_score < 0.5:
            recommendations.append("Consider retraining the model with additional data")
            recommendations.append("Review model architecture and hyperparameters")
        elif accuracy_score < 0.7:
            recommendations.append("Fine-tune model parameters for better accuracy")
            recommendations.append("Analyze failure cases for pattern identification")
        
        if expected:
            missing_fields = set(expected.keys()) - set(predicted.keys())
            if missing_fields:
                recommendations.append(f"Ensure model outputs include required fields: {', '.join(missing_fields)}")
        
        return recommendations
    
    def _compare_outputs(self, predicted: Dict[str, Any], 
                        expected: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare predicted and expected outputs"""
        
        if not expected:
            return {"comparison": "no_expected_output"}
        
        comparison = {
            "matching_fields": [],
            "mismatched_fields": [],
            "missing_fields": [],
            "extra_fields": []
        }
        
        for key in expected.keys():
            if key in predicted:
                if predicted[key] == expected[key]:
                    comparison["matching_fields"].append(key)
                else:
                    comparison["mismatched_fields"].append({
                        "field": key,
                        "predicted": predicted[key],
                        "expected": expected[key]
                    })
            else:
                comparison["missing_fields"].append(key)
        
        for key in predicted.keys():
            if key not in expected:
                comparison["extra_fields"].append(key)
        
        return comparison
    
    def _get_accuracy_breakdown(self, predicted: Dict[str, Any], 
                              expected: Optional[Dict[str, Any]], 
                              task_category: TaskCategory) -> Dict[str, float]:
        """Get detailed accuracy breakdown"""
        
        breakdown = {}
        
        if task_category == TaskCategory.SENTIMENT_ANALYSIS:
            if expected and "overall_sentiment" in predicted and "overall_sentiment" in expected:
                breakdown["sentiment_classification"] = 1.0 if predicted["overall_sentiment"] == expected["overall_sentiment"] else 0.0
            
            if expected and "sentiment_score" in predicted and "sentiment_score" in expected:
                diff = abs(predicted["sentiment_score"] - expected["sentiment_score"])
                breakdown["sentiment_score"] = max(0, 1 - diff / 2)
        
        elif task_category == TaskCategory.RISK_ASSESSMENT:
            if expected and "risk_level" in predicted and "risk_level" in expected:
                breakdown["risk_classification"] = 1.0 if predicted["risk_level"] == expected["risk_level"] else 0.0
            
            if expected and "overall_risk_score" in predicted and "overall_risk_score" in expected:
                diff = abs(predicted["overall_risk_score"] - expected["overall_risk_score"])
                breakdown["risk_score"] = max(0, 1 - diff)
        
        return breakdown
    
    def _compare_with_ground_truth(self, predicted: Dict[str, Any], 
                                 ground_truth: Any) -> Dict[str, Any]:
        """Compare with ground truth data"""
        
        if ground_truth is None:
            return {"comparison": "no_ground_truth"}
        
        # This would implement actual ground truth comparison
        # For now, return a placeholder
        return {
            "ground_truth_available": True,
            "comparison_method": "placeholder",
            "accuracy": 0.75  # Placeholder accuracy
        }
    
    def _build_accuracy_criteria(self) -> Dict[str, str]:
        """Build accuracy evaluation criteria"""
        
        return {
            "field_matching": "Predicted output fields match expected output fields",
            "value_accuracy": "Predicted values are within acceptable tolerance of expected values",
            "classification_accuracy": "Categorical predictions match expected classifications",
            "numerical_precision": "Numerical predictions are accurate within defined bounds",
            "completeness": "All required output fields are present and populated"
        }
    
    def get_evaluation_criteria(self) -> Dict[str, str]:
        """Get evaluation criteria for this judge"""
        return self.evaluation_criteria


class ConsistencyJudge(LLMJudge):
    """Judge for evaluating consistency of model outputs"""
    
    def __init__(self, specialization: TaskCategory):
        super().__init__(f"consistency_judge_{specialization.value}", specialization)
        self.evaluation_criteria = self._build_consistency_criteria()
        
    async def evaluate(self, test_case: TestCase, model_output: ModelOutput) -> EvaluationResult:
        """Evaluate consistency of model output"""
        
        start_time = datetime.now()
        
        try:
            # Check internal consistency
            consistency_score = await self._calculate_consistency_score(model_output, test_case)
            
            # Generate feedback
            judge_feedback = await self._generate_consistency_feedback(model_output, consistency_score)
            
            # Generate recommendations
            recommendations = await self._generate_consistency_recommendations(consistency_score, model_output)
            
            # Determine result
            result = TestResult.PASS if consistency_score >= 0.7 else TestResult.FAIL
            if 0.5 <= consistency_score < 0.7:
                result = TestResult.WARNING
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return EvaluationResult(
                test_case_id=test_case.test_id,
                test_type=TestType.CONSISTENCY_TEST,
                result=result,
                score=consistency_score,
                confidence=model_output.confidence,
                details={
                    "consistency_checks": self._get_consistency_checks(model_output),
                    "internal_coherence": self._check_internal_coherence(model_output),
                    "confidence_calibration": self._check_confidence_calibration(model_output)
                },
                execution_time=execution_time,
                timestamp=datetime.now(),
                judge_feedback=judge_feedback,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Consistency evaluation failed: {e}")
            return EvaluationResult(
                test_case_id=test_case.test_id,
                test_type=TestType.CONSISTENCY_TEST,
                result=TestResult.ERROR,
                score=0.0,
                confidence=0.0,
                details={"error": str(e)},
                execution_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now(),
                judge_feedback=f"Consistency evaluation failed: {str(e)}",
                recommendations=["Review model output format and consistency logic"]
            )
    
    async def _calculate_consistency_score(self, model_output: ModelOutput, test_case: TestCase) -> float:
        """Calculate consistency score"""
        
        score = 0.0
        checks = 0
        
        # Check confidence-prediction consistency
        confidence_consistency = self._check_confidence_prediction_consistency(model_output)
        score += confidence_consistency
        checks += 1
        
        # Check internal field consistency
        field_consistency = self._check_field_consistency(model_output)
        score += field_consistency
        checks += 1
        
        # Check logical consistency
        logical_consistency = self._check_logical_consistency(model_output)
        score += logical_consistency
        checks += 1
        
        return score / checks if checks > 0 else 0.0
    
    def _check_confidence_prediction_consistency(self, model_output: ModelOutput) -> float:
        """Check if confidence aligns with prediction quality"""
        
        confidence = model_output.confidence
        validation_score = model_output.validation_score
        
        # High confidence should correlate with high validation score
        if confidence > 0.8 and validation_score < 0.5:
            return 0.2  # Poor consistency
        elif confidence < 0.3 and validation_score > 0.8:
            return 0.3  # Poor consistency
        elif abs(confidence - validation_score) < 0.2:
            return 1.0  # Good consistency
        else:
            return 0.7  # Moderate consistency
    
    def _check_field_consistency(self, model_output: ModelOutput) -> float:
        """Check consistency between related fields"""
        
        result = model_output.result
        
        if model_output.task_category == TaskCategory.SENTIMENT_ANALYSIS:
            return self._check_sentiment_field_consistency(result)
        elif model_output.task_category == TaskCategory.RISK_ASSESSMENT:
            return self._check_risk_field_consistency(result)
        elif model_output.task_category == TaskCategory.EARNINGS_PREDICTION:
            return self._check_earnings_field_consistency(result)
        else:
            return 0.8  # Default moderate consistency
    
    def _check_sentiment_field_consistency(self, result: Dict[str, Any]) -> float:
        """Check sentiment analysis field consistency"""
        
        if "overall_sentiment" not in result or "sentiment_score" not in result:
            return 0.5
        
        sentiment = result["overall_sentiment"]
        score = result["sentiment_score"]
        
        # Check if sentiment classification matches score
        if sentiment == "bullish" and score > 0.1:
            return 1.0
        elif sentiment == "bearish" and score < -0.1:
            return 1.0
        elif sentiment == "neutral" and -0.1 <= score <= 0.1:
            return 1.0
        else:
            return 0.3  # Inconsistent
    
    def _check_risk_field_consistency(self, result: Dict[str, Any]) -> float:
        """Check risk assessment field consistency"""
        
        if "risk_level" not in result or "overall_risk_score" not in result:
            return 0.5
        
        risk_level = result["risk_level"]
        risk_score = result["overall_risk_score"]
        
        # Check if risk level matches score
        if risk_level == "low" and risk_score < 0.3:
            return 1.0
        elif risk_level == "moderate" and 0.3 <= risk_score < 0.7:
            return 1.0
        elif risk_level == "high" and risk_score >= 0.7:
            return 1.0
        else:
            return 0.3  # Inconsistent
    
    def _check_earnings_field_consistency(self, result: Dict[str, Any]) -> float:
        """Check earnings analysis field consistency"""
        
        if "investment_signal" not in result or "signal_strength" not in result:
            return 0.5
        
        signal = result["investment_signal"]
        strength = result["signal_strength"]
        
        # Check if signal matches strength
        if signal == "bullish" and strength > 0.5:
            return 1.0
        elif signal == "bearish" and strength < 0.5:
            return 1.0
        elif signal == "neutral" and 0.4 <= strength <= 0.6:
            return 1.0
        else:
            return 0.3  # Inconsistent
    
    def _check_logical_consistency(self, model_output: ModelOutput) -> float:
        """Check logical consistency of the output"""
        
        # Check for logical contradictions
        result_str = json.dumps(model_output.result, default=str).lower()
        
        # Look for contradictory terms
        contradictions = [
            ("positive", "negative"),
            ("bullish", "bearish"),
            ("high risk", "low risk"),
            ("strong", "weak")
        ]
        
        contradiction_count = 0
        for term1, term2 in contradictions:
            if term1 in result_str and term2 in result_str:
                contradiction_count += 1
        
        # Penalize contradictions
        if contradiction_count == 0:
            return 1.0
        elif contradiction_count == 1:
            return 0.7
        else:
            return 0.3
    
    async def _generate_consistency_feedback(self, model_output: ModelOutput, consistency_score: float) -> str:
        """Generate consistency feedback"""
        
        if consistency_score >= 0.8:
            return "Excellent consistency - all output components are well-aligned"
        elif consistency_score >= 0.6:
            return "Good consistency - minor alignment issues detected"
        elif consistency_score >= 0.4:
            return "Moderate consistency - several alignment issues identified"
        else:
            return "Poor consistency - significant internal contradictions found"
    
    async def _generate_consistency_recommendations(self, consistency_score: float, 
                                                  model_output: ModelOutput) -> List[str]:
        """Generate consistency recommendations"""
        
        recommendations = []
        
        if consistency_score < 0.5:
            recommendations.append("Review model logic to eliminate internal contradictions")
            recommendations.append("Implement consistency checks in model output generation")
        elif consistency_score < 0.7:
            recommendations.append("Fine-tune confidence calibration")
            recommendations.append("Improve field relationship validation")
        
        # Check specific consistency issues
        if model_output.confidence > 0.8 and model_output.validation_score < 0.5:
            recommendations.append("Recalibrate confidence scoring mechanism")
        
        return recommendations
    
    def _get_consistency_checks(self, model_output: ModelOutput) -> Dict[str, float]:
        """Get detailed consistency check results"""
        
        return {
            "confidence_prediction_consistency": self._check_confidence_prediction_consistency(model_output),
            "field_consistency": self._check_field_consistency(model_output),
            "logical_consistency": self._check_logical_consistency(model_output)
        }
    
    def _check_internal_coherence(self, model_output: ModelOutput) -> Dict[str, Any]:
        """Check internal coherence of the output"""
        
        return {
            "field_relationships": "consistent" if self._check_field_consistency(model_output) > 0.7 else "inconsistent",
            "confidence_alignment": "aligned" if self._check_confidence_prediction_consistency(model_output) > 0.7 else "misaligned",
            "logical_flow": "coherent" if self._check_logical_consistency(model_output) > 0.7 else "contradictory"
        }
    
    def _check_confidence_calibration(self, model_output: ModelOutput) -> Dict[str, Any]:
        """Check confidence calibration"""
        
        confidence = model_output.confidence
        validation_score = model_output.validation_score
        
        return {
            "confidence_level": confidence,
            "validation_score": validation_score,
            "calibration_quality": "well_calibrated" if abs(confidence - validation_score) < 0.2 else "poorly_calibrated",
            "calibration_error": abs(confidence - validation_score)
        }
    
    def _build_consistency_criteria(self) -> Dict[str, str]:
        """Build consistency evaluation criteria"""
        
        return {
            "internal_coherence": "Output components are logically consistent with each other",
            "confidence_calibration": "Confidence scores align with prediction quality",
            "field_relationships": "Related fields have consistent values",
            "logical_flow": "No contradictory statements or impossible combinations",
            "prediction_alignment": "Predictions align with supporting evidence"
        }
    
    def get_evaluation_criteria(self) -> Dict[str, str]:
        """Get evaluation criteria for this judge"""
        return self.evaluation_criteria

class ContinuousEvaluationPipeline:
    """Main pipeline for continuous evaluation and testing"""
    
    def __init__(self, ai_engine: HybridAIEngine):
        self.ai_engine = ai_engine
        self.judges = self._initialize_judges()
        self.test_cases = {}
        self.evaluation_history = []
        self.performance_baselines = {}
        self.is_running = False
        self.scheduler_thread = None
        
        # Performance tracking
        self.model_metrics = {}
        self.regression_alerts = []
        self.evaluation_stats = {
            "total_evaluations": 0,
            "total_test_cases": 0,
            "avg_accuracy": 0.0,
            "avg_consistency": 0.0,
            "last_evaluation": None
        }
        
        logger.info("Continuous Evaluation Pipeline initialized")
    
    def _initialize_judges(self) -> Dict[TaskCategory, Dict[TestType, LLMJudge]]:
        """Initialize LLM judges for different task categories"""
        
        judges = {}
        
        for task_category in TaskCategory:
            judges[task_category] = {
                TestType.ACCURACY_TEST: AccuracyJudge(task_category),
                TestType.CONSISTENCY_TEST: ConsistencyJudge(task_category)
            }
        
        return judges
    
    async def add_test_case(self, test_case: TestCase):
        """Add a test case to the evaluation suite"""
        
        if test_case.task_category not in self.test_cases:
            self.test_cases[test_case.task_category] = []
        
        self.test_cases[test_case.task_category].append(test_case)
        self.evaluation_stats["total_test_cases"] += 1
        
        logger.info(f"Added test case {test_case.test_id} for {test_case.task_category.value}")
    
    async def run_evaluation(self, task_category: Optional[TaskCategory] = None,
                           test_types: Optional[List[TestType]] = None) -> Dict[str, Any]:
        """Run evaluation for specified task category and test types"""
        
        start_time = datetime.now()
        
        # Default to all task categories and test types
        categories_to_test = [task_category] if task_category else list(TaskCategory)
        types_to_test = test_types if test_types else [TestType.ACCURACY_TEST, TestType.CONSISTENCY_TEST]
        
        evaluation_results = {
            "evaluation_id": hashlib.md5(f"{start_time}".encode()).hexdigest()[:8],
            "start_time": start_time,
            "categories_tested": [cat.value for cat in categories_to_test],
            "test_types": [tt.value for tt in types_to_test],
            "results": {},
            "summary": {},
            "alerts": []
        }
        
        total_tests = 0
        total_passed = 0
        
        for category in categories_to_test:
            if category not in self.test_cases or not self.test_cases[category]:
                logger.warning(f"No test cases available for {category.value}")
                continue
            
            category_results = {
                "test_cases": [],
                "summary": {
                    "total_tests": 0,
                    "passed": 0,
                    "failed": 0,
                    "warnings": 0,
                    "errors": 0,
                    "avg_score": 0.0
                }
            }
            
            for test_case in self.test_cases[category]:
                # Run model on test case
                try:
                    model_output = await self._run_model_on_test_case(test_case)
                    
                    # Evaluate with judges
                    test_case_results = {}
                    
                    for test_type in types_to_test:
                        if test_type in self.judges[category]:
                            judge = self.judges[category][test_type]
                            result = await judge.evaluate(test_case, model_output)
                            test_case_results[test_type.value] = result
                            
                            # Update summary
                            category_results["summary"]["total_tests"] += 1
                            total_tests += 1
                            
                            if result.result == TestResult.PASS:
                                category_results["summary"]["passed"] += 1
                                total_passed += 1
                            elif result.result == TestResult.FAIL:
                                category_results["summary"]["failed"] += 1
                            elif result.result == TestResult.WARNING:
                                category_results["summary"]["warnings"] += 1
                            elif result.result == TestResult.ERROR:
                                category_results["summary"]["errors"] += 1
                    
                    category_results["test_cases"].append({
                        "test_case_id": test_case.test_id,
                        "results": test_case_results
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate test case {test_case.test_id}: {e}")
                    category_results["summary"]["errors"] += 1
            
            # Calculate average score for category
            if category_results["summary"]["total_tests"] > 0:
                total_score = 0
                score_count = 0
                
                for test_case_result in category_results["test_cases"]:
                    for test_type, result in test_case_result["results"].items():
                        total_score += result.score
                        score_count += 1
                
                category_results["summary"]["avg_score"] = total_score / score_count if score_count > 0 else 0.0
            
            evaluation_results["results"][category.value] = category_results
        
        # Generate overall summary
        evaluation_results["summary"] = {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "pass_rate": total_passed / total_tests if total_tests > 0 else 0.0,
            "execution_time": (datetime.now() - start_time).total_seconds(),
            "timestamp": datetime.now()
        }
        
        # Check for regressions
        alerts = await self._check_for_regressions(evaluation_results)
        evaluation_results["alerts"] = alerts
        
        # Store evaluation history
        self.evaluation_history.append(evaluation_results)
        
        # Update statistics
        self._update_evaluation_stats(evaluation_results)
        
        logger.info(f"Evaluation completed: {total_passed}/{total_tests} tests passed "
                   f"({evaluation_results['summary']['pass_rate']:.1%})")
        
        return evaluation_results
    
    async def _run_model_on_test_case(self, test_case: TestCase) -> ModelOutput:
        """Run AI model on test case input"""
        
        # Get the appropriate specialized model
        model_name = self._get_model_name_for_task(test_case.task_category)
        
        if model_name in self.ai_engine.specialized_models:
            model = self.ai_engine.specialized_models[model_name]
            return model.predict(test_case.input_data)
        else:
            # Fallback to general processing
            return await self.ai_engine.process_request(
                test_case.task_category,
                test_case.input_data
            )
    
    def _get_model_name_for_task(self, task_category: TaskCategory) -> str:
        """Get model name for task category"""
        
        mapping = {
            TaskCategory.EARNINGS_PREDICTION: "earnings_analysis",
            TaskCategory.SENTIMENT_ANALYSIS: "sentiment_analysis",
            TaskCategory.RISK_ASSESSMENT: "risk_prediction",
            TaskCategory.THEMATIC_IDENTIFICATION: "thematic_identification"
        }
        
        return mapping.get(task_category, "general")
    
    async def _check_for_regressions(self, evaluation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for performance regressions"""
        
        alerts = []
        
        for category, results in evaluation_results["results"].items():
            current_score = results["summary"]["avg_score"]
            
            # Compare with baseline
            if category in self.performance_baselines:
                baseline_score = self.performance_baselines[category]["avg_score"]
                
                # Check for significant regression (>10% drop)
                if current_score < baseline_score * 0.9:
                    alerts.append({
                        "type": "performance_regression",
                        "category": category,
                        "current_score": current_score,
                        "baseline_score": baseline_score,
                        "regression_percentage": ((baseline_score - current_score) / baseline_score) * 100,
                        "severity": "high" if current_score < baseline_score * 0.8 else "medium",
                        "recommendation": "Investigate model changes and consider rollback"
                    })
            else:
                # Set as baseline if first evaluation
                self.performance_baselines[category] = {
                    "avg_score": current_score,
                    "timestamp": datetime.now()
                }
        
        return alerts
    
    def _update_evaluation_stats(self, evaluation_results: Dict[str, Any]):
        """Update evaluation statistics"""
        
        self.evaluation_stats["total_evaluations"] += 1
        self.evaluation_stats["last_evaluation"] = evaluation_results["summary"]["timestamp"]
        
        # Update running averages
        total_tests = evaluation_results["summary"]["total_tests"]
        if total_tests > 0:
            # Calculate weighted average
            current_weight = total_tests
            total_weight = self.evaluation_stats["total_evaluations"]
            
            # Update accuracy (using pass rate as proxy)
            new_accuracy = evaluation_results["summary"]["pass_rate"]
            self.evaluation_stats["avg_accuracy"] = (
                (self.evaluation_stats["avg_accuracy"] * (total_weight - 1) + new_accuracy) / total_weight
            )
            
            # Update consistency (calculate from detailed results)
            consistency_scores = []
            for category_results in evaluation_results["results"].values():
                for test_case in category_results["test_cases"]:
                    for test_type, result in test_case["results"].items():
                        if test_type == "consistency_test":
                            consistency_scores.append(result.score)
            
            if consistency_scores:
                avg_consistency = sum(consistency_scores) / len(consistency_scores)
                self.evaluation_stats["avg_consistency"] = (
                    (self.evaluation_stats["avg_consistency"] * (total_weight - 1) + avg_consistency) / total_weight
                )
    
    async def run_daily_evaluation(self) -> Dict[str, Any]:
        """Run daily automated evaluation"""
        
        logger.info("Starting daily evaluation")
        
        # Run comprehensive evaluation
        results = await self.run_evaluation()
        
        # Generate daily report
        daily_report = await self._generate_daily_report(results)
        
        # Check for critical issues
        critical_alerts = [alert for alert in results["alerts"] if alert.get("severity") == "high"]
        
        if critical_alerts:
            logger.warning(f"Daily evaluation found {len(critical_alerts)} critical issues")
            await self._handle_critical_alerts(critical_alerts)
        
        return daily_report
    
    async def _generate_daily_report(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate daily evaluation report"""
        
        report = {
            "report_date": datetime.now().strftime("%Y-%m-%d"),
            "evaluation_summary": evaluation_results["summary"],
            "performance_trends": await self._analyze_performance_trends(),
            "model_health": await self._assess_model_health(),
            "recommendations": await self._generate_daily_recommendations(evaluation_results),
            "alerts": evaluation_results["alerts"]
        }
        
        return report
    
    async def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        
        if len(self.evaluation_history) < 2:
            return {"trend": "insufficient_data"}
        
        # Get last 7 evaluations
        recent_evaluations = self.evaluation_history[-7:]
        
        # Calculate trend
        pass_rates = [eval_result["summary"]["pass_rate"] for eval_result in recent_evaluations]
        
        if len(pass_rates) >= 2:
            trend_slope = (pass_rates[-1] - pass_rates[0]) / len(pass_rates)
            
            if trend_slope > 0.01:
                trend = "improving"
            elif trend_slope < -0.01:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "recent_pass_rates": pass_rates,
            "avg_recent_pass_rate": sum(pass_rates) / len(pass_rates),
            "trend_slope": trend_slope if len(pass_rates) >= 2 else 0
        }
    
    async def _assess_model_health(self) -> Dict[str, Any]:
        """Assess overall model health"""
        
        health_status = {}
        
        for category in TaskCategory:
            if category.value in self.performance_baselines:
                baseline = self.performance_baselines[category.value]
                
                # Simple health assessment
                if baseline["avg_score"] >= 0.8:
                    health = "excellent"
                elif baseline["avg_score"] >= 0.6:
                    health = "good"
                elif baseline["avg_score"] >= 0.4:
                    health = "fair"
                else:
                    health = "poor"
                
                health_status[category.value] = {
                    "health": health,
                    "score": baseline["avg_score"],
                    "last_updated": baseline["timestamp"].isoformat()
                }
        
        return health_status
    
    async def _generate_daily_recommendations(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """Generate daily recommendations"""
        
        recommendations = []
        
        # Check overall pass rate
        pass_rate = evaluation_results["summary"]["pass_rate"]
        
        if pass_rate < 0.7:
            recommendations.append("Overall pass rate is below 70% - review model performance")
        
        if pass_rate < 0.5:
            recommendations.append("Critical: Pass rate below 50% - immediate intervention required")
        
        # Check for specific category issues
        for category, results in evaluation_results["results"].items():
            category_pass_rate = results["summary"]["passed"] / results["summary"]["total_tests"] if results["summary"]["total_tests"] > 0 else 0
            
            if category_pass_rate < 0.6:
                recommendations.append(f"Review {category} model - pass rate: {category_pass_rate:.1%}")
        
        # Check for alerts
        if evaluation_results["alerts"]:
            recommendations.append(f"Address {len(evaluation_results['alerts'])} performance alerts")
        
        return recommendations
    
    async def _handle_critical_alerts(self, critical_alerts: List[Dict[str, Any]]):
        """Handle critical performance alerts"""
        
        for alert in critical_alerts:
            logger.critical(f"Critical alert: {alert['type']} in {alert.get('category', 'unknown')}")
            
            if alert["type"] == "performance_regression":
                # Could trigger automatic rollback or notification
                logger.critical(f"Performance regression detected: {alert['regression_percentage']:.1f}% drop")
    
    def start_continuous_evaluation(self, schedule_time: str = "02:00"):
        """Start continuous evaluation with daily scheduling"""
        
        if self.is_running:
            logger.warning("Continuous evaluation is already running")
            return
        
        self.is_running = True
        
        # For demo purposes, we'll simulate scheduling
        # In production, this would use a proper scheduler like APScheduler
        logger.info(f"Continuous evaluation started - would schedule daily at {schedule_time}")
        logger.info("Note: Using simplified scheduling for demo - implement proper scheduler in production")
    
    def _run_scheduled_evaluation(self):
        """Run scheduled evaluation (sync wrapper)"""
        
        try:
            # Run async evaluation in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(self.run_daily_evaluation())
            
            logger.info("Scheduled evaluation completed successfully")
            
        except Exception as e:
            logger.error(f"Scheduled evaluation failed: {e}")
        finally:
            loop.close()
    
    def _run_scheduler(self):
        """Run the scheduler loop"""
        
        # Simplified scheduler for demo
        logger.info("Scheduler would run here - implement proper scheduling in production")
    
    def stop_continuous_evaluation(self):
        """Stop continuous evaluation"""
        
        self.is_running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.shutdown(wait=True)
        
        # Clear scheduler (simplified for demo)
        
        logger.info("Continuous evaluation stopped")
    
    async def get_evaluation_report(self, days: int = 7) -> Dict[str, Any]:
        """Get evaluation report for specified period"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter recent evaluations
        recent_evaluations = [
            eval_result for eval_result in self.evaluation_history
            if eval_result["summary"]["timestamp"] >= cutoff_date
        ]
        
        if not recent_evaluations:
            return {"error": "No evaluations in specified period"}
        
        # Calculate summary statistics
        total_tests = sum(eval_result["summary"]["total_tests"] for eval_result in recent_evaluations)
        total_passed = sum(eval_result["summary"]["total_passed"] for eval_result in recent_evaluations)
        
        avg_pass_rate = total_passed / total_tests if total_tests > 0 else 0.0
        
        # Get all alerts
        all_alerts = []
        for eval_result in recent_evaluations:
            all_alerts.extend(eval_result["alerts"])
        
        return {
            "period_days": days,
            "total_evaluations": len(recent_evaluations),
            "total_tests": total_tests,
            "total_passed": total_passed,
            "avg_pass_rate": avg_pass_rate,
            "total_alerts": len(all_alerts),
            "critical_alerts": len([a for a in all_alerts if a.get("severity") == "high"]),
            "performance_trends": await self._analyze_performance_trends(),
            "model_health": await self._assess_model_health(),
            "recent_evaluations": recent_evaluations[-5:],  # Last 5 evaluations
            "timestamp": datetime.now()
        }


# Factory function for creating evaluation pipeline
def create_evaluation_pipeline(ai_engine: HybridAIEngine) -> ContinuousEvaluationPipeline:
    """Factory function to create evaluation pipeline"""
    return ContinuousEvaluationPipeline(ai_engine)


# Utility functions for creating test cases
def create_sentiment_test_case(test_id: str, input_text: str, expected_sentiment: str, 
                             expected_score: float) -> TestCase:
    """Create a sentiment analysis test case"""
    
    return TestCase(
        test_id=test_id,
        test_type=TestType.ACCURACY_TEST,
        task_category=TaskCategory.SENTIMENT_ANALYSIS,
        input_data={
            "text": input_text,
            "symbols": ["TEST"],
            "source": "news_articles"
        },
        expected_output={
            "overall_sentiment": expected_sentiment,
            "sentiment_score": expected_score,
            "confidence": 0.8
        },
        ground_truth=None,
        test_description=f"Sentiment analysis test for: {input_text[:50]}...",
        created_at=datetime.now(),
        metadata={"category": "sentiment", "difficulty": "medium"}
    )


def create_earnings_test_case(test_id: str, earnings_text: str, expected_signal: str,
                            expected_strength: float) -> TestCase:
    """Create an earnings analysis test case"""
    
    return TestCase(
        test_id=test_id,
        test_type=TestType.ACCURACY_TEST,
        task_category=TaskCategory.EARNINGS_PREDICTION,
        input_data=earnings_text,
        expected_output={
            "investment_signal": expected_signal,
            "signal_strength": expected_strength,
            "key_themes": ["revenue_growth", "margin_expansion"],
            "risk_factors": ["competition"]
        },
        ground_truth=None,
        test_description=f"Earnings analysis test for: {earnings_text[:50]}...",
        created_at=datetime.now(),
        metadata={"category": "earnings", "difficulty": "medium"}
    )


def create_risk_test_case(test_id: str, symbols: List[str], expected_risk_level: str,
                        expected_risk_score: float) -> TestCase:
    """Create a risk assessment test case"""
    
    return TestCase(
        test_id=test_id,
        test_type=TestType.ACCURACY_TEST,
        task_category=TaskCategory.RISK_ASSESSMENT,
        input_data={
            "symbols": symbols,
            "risk_types": ["market_risk", "volatility_risk"],
            "time_horizon": "monthly",
            "portfolio_data": {"total_value": 1000000}
        },
        expected_output={
            "overall_risk_score": expected_risk_score,
            "risk_level": expected_risk_level,
            "confidence": 0.8,
            "risk_factors": [{"name": "market_risk", "value": expected_risk_score}]
        },
        ground_truth=None,
        test_description=f"Risk assessment test for symbols: {', '.join(symbols)}",
        created_at=datetime.now(),
        metadata={"category": "risk", "difficulty": "medium"}
    )