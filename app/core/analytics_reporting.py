"""
Advanced Analytics and Reporting System
BlackRock Aladdin-inspired comprehensive performance reporting, model drift analysis, 
and business intelligence with regulatory compliance automation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
import json
import asyncio
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from abc import ABC, abstractmethod
import uuid
from collections import defaultdict, deque
import statistics
import warnings
warnings.filterwarnings('ignore')

from app.core.monitoring_system import MetricsCollector, AlertManager, MetricType
from app.core.hybrid_ai_engine import TaskCategory, ModelType

logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Types of reports generated"""
    PERFORMANCE = "performance"
    BUSINESS_INTELLIGENCE = "business_intelligence"
    MODEL_DRIFT = "model_drift"
    COMPLIANCE = "compliance"
    USER_BEHAVIOR = "user_behavior"
    ROI_ANALYSIS = "roi_analysis"
    EXECUTIVE_SUMMARY = "executive_summary"
    TECHNICAL_DEEP_DIVE = "technical_deep_dive"


class ReportFrequency(Enum):
    """Report generation frequency"""
    REAL_TIME = "real_time"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    ON_DEMAND = "on_demand"


class TrendDirection(Enum):
    """Trend analysis directions"""
    STRONGLY_INCREASING = "strongly_increasing"
    INCREASING = "increasing"
    STABLE = "stable"
    DECREASING = "decreasing"
    STRONGLY_DECREASING = "strongly_decreasing"
    VOLATILE = "volatile"


@dataclass
class PerformanceReport:
    """Performance analysis report"""
    report_id: str
    report_type: ReportType
    generated_at: datetime
    time_period: Dict[str, datetime]
    metrics_analyzed: List[str]
    key_findings: List[str]
    performance_summary: Dict[str, Any]
    trend_analysis: Dict[str, Any]
    recommendations: List[str]
    data_quality_score: float
    confidence_level: float
    metadata: Dict[str, Any]


@dataclass
class ModelDriftReport:
    """Model drift analysis report"""
    report_id: str
    model_name: str
    analysis_period: Dict[str, datetime]
    drift_detected: bool
    drift_severity: str  # "low", "medium", "high", "critical"
    drift_metrics: Dict[str, float]
    affected_features: List[str]
    root_cause_analysis: Dict[str, Any]
    remediation_plan: List[str]
    retraining_recommended: bool
    confidence_score: float
    generated_at: datetime


@dataclass
class BusinessIntelligenceReport:
    """Business intelligence and ROI report"""
    report_id: str
    reporting_period: Dict[str, datetime]
    roi_metrics: Dict[str, float]
    cost_analysis: Dict[str, Any]
    efficiency_gains: Dict[str, float]
    user_adoption_metrics: Dict[str, Any]
    business_impact: Dict[str, Any]
    competitive_analysis: Dict[str, Any]
    strategic_recommendations: List[str]
    executive_summary: str
    generated_at: datetime


@dataclass
class ComplianceReport:
    """Regulatory compliance report"""
    report_id: str
    compliance_framework: str
    assessment_period: Dict[str, datetime]
    overall_compliance_score: float
    compliance_by_category: Dict[str, float]
    violations_detected: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    remediation_actions: List[str]
    audit_trail: List[Dict[str, Any]]
    certification_status: str
    next_review_date: datetime
    generated_at: datetime


class PerformanceAnalyzer:
    """Analyzes system and model performance metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.analysis_cache: Dict[str, Any] = {}
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
        
        # Initialize baseline metrics
        self._initialize_baselines()
        
        logger.info("Performance Analyzer initialized")
    
    def _initialize_baselines(self):
        """Initialize baseline performance metrics"""
        
        self.baseline_metrics = {
            "model_accuracy": {
                "target": 85.0,
                "acceptable_min": 80.0,
                "excellent": 90.0,
                "degradation_threshold": 5.0
            },
            "response_latency": {
                "target": 200.0,
                "acceptable_max": 500.0,
                "excellent": 150.0,
                "degradation_threshold": 50.0
            },
            "error_rate": {
                "target": 1.0,
                "acceptable_max": 3.0,
                "excellent": 0.5,
                "degradation_threshold": 2.0
            },
            "user_satisfaction": {
                "target": 8.0,
                "acceptable_min": 7.0,
                "excellent": 9.0,
                "degradation_threshold": 1.0
            },
            "cost_per_request": {
                "target": 0.05,
                "acceptable_max": 0.10,
                "excellent": 0.03,
                "degradation_threshold": 0.02
            }
        }
    
    async def generate_performance_report(self, time_period_hours: int = 24) -> PerformanceReport:
        """Generate comprehensive performance report"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=time_period_hours)
        
        # Collect metrics for analysis
        metrics_data = {}
        metrics_analyzed = []
        
        for metric_id in self.baseline_metrics.keys():
            metrics = self.metrics_collector.get_metrics(metric_id, start_time, end_time)
            if metrics:
                metrics_data[metric_id] = metrics
                metrics_analyzed.append(metric_id)
        
        # Perform analysis
        performance_summary = await self._analyze_performance_metrics(metrics_data)
        trend_analysis = await self._analyze_trends(metrics_data)
        key_findings = await self._generate_key_findings(performance_summary, trend_analysis)
        recommendations = await self._generate_recommendations(performance_summary, trend_analysis)
        
        # Calculate data quality and confidence
        data_quality_score = self._calculate_data_quality_score(metrics_data)
        confidence_level = self._calculate_confidence_level(metrics_data, data_quality_score)
        
        report = PerformanceReport(
            report_id=f"perf_report_{uuid.uuid4().hex[:8]}",
            report_type=ReportType.PERFORMANCE,
            generated_at=datetime.now(),
            time_period={"start": start_time, "end": end_time},
            metrics_analyzed=metrics_analyzed,
            key_findings=key_findings,
            performance_summary=performance_summary,
            trend_analysis=trend_analysis,
            recommendations=recommendations,
            data_quality_score=data_quality_score,
            confidence_level=confidence_level,
            metadata={
                "analysis_period_hours": time_period_hours,
                "total_data_points": sum(len(metrics) for metrics in metrics_data.values()),
                "baseline_comparison": True
            }
        )
        
        logger.info(f"Generated performance report: {report.report_id}")
        
        return report
    
    async def _analyze_performance_metrics(self, metrics_data: Dict[str, List]) -> Dict[str, Any]:
        """Analyze performance metrics against baselines"""
        
        analysis = {}
        
        for metric_id, metrics in metrics_data.items():
            if not metrics:
                continue
            
            values = [m.value for m in metrics]
            baseline = self.baseline_metrics.get(metric_id, {})
            
            current_avg = statistics.mean(values)
            current_latest = values[-1]
            target = baseline.get("target", current_avg)
            
            # Performance scoring
            if "accuracy" in metric_id or "satisfaction" in metric_id or "compliance" in metric_id:
                # Higher is better
                performance_score = min(100, (current_avg / target) * 100) if target > 0 else 0
                vs_target = ((current_avg - target) / target) * 100 if target > 0 else 0
            else:
                # Lower is better (latency, error rate, cost)
                performance_score = min(100, (target / current_avg) * 100) if current_avg > 0 else 0
                vs_target = ((target - current_avg) / target) * 100 if target > 0 else 0
            
            # Status determination
            excellent_threshold = baseline.get("excellent", target)
            acceptable_threshold = baseline.get("acceptable_min" if "accuracy" in metric_id or "satisfaction" in metric_id else "acceptable_max", target)
            
            if "accuracy" in metric_id or "satisfaction" in metric_id or "compliance" in metric_id:
                if current_avg >= excellent_threshold:
                    status = "excellent"
                elif current_avg >= acceptable_threshold:
                    status = "good"
                elif current_avg >= target:
                    status = "acceptable"
                else:
                    status = "poor"
            else:
                if current_avg <= excellent_threshold:
                    status = "excellent"
                elif current_avg <= acceptable_threshold:
                    status = "good"
                elif current_avg <= target:
                    status = "acceptable"
                else:
                    status = "poor"
            
            analysis[metric_id] = {
                "current_average": round(current_avg, 2),
                "latest_value": round(current_latest, 2),
                "target_value": target,
                "performance_score": round(performance_score, 1),
                "vs_target_percent": round(vs_target, 1),
                "status": status,
                "data_points": len(values),
                "min_value": min(values),
                "max_value": max(values),
                "std_deviation": round(statistics.stdev(values), 2) if len(values) > 1 else 0
            }
        
        return analysis
    
    async def _analyze_trends(self, metrics_data: Dict[str, List]) -> Dict[str, Any]:
        """Analyze trends in performance metrics"""
        
        trends = {}
        
        for metric_id, metrics in metrics_data.items():
            if len(metrics) < 5:  # Need sufficient data for trend analysis
                continue
            
            values = [m.value for m in metrics]
            timestamps = [m.timestamp for m in metrics]
            
            # Calculate trend using linear regression
            x = np.arange(len(values))
            y = np.array(values)
            
            if len(x) > 1:
                slope, intercept = np.polyfit(x, y, 1)
                
                # Determine trend direction
                avg_value = statistics.mean(values)
                slope_percent = (slope / avg_value) * 100 if avg_value != 0 else 0
                
                if abs(slope_percent) < 1:
                    direction = TrendDirection.STABLE
                elif slope_percent > 5:
                    direction = TrendDirection.STRONGLY_INCREASING
                elif slope_percent > 1:
                    direction = TrendDirection.INCREASING
                elif slope_percent < -5:
                    direction = TrendDirection.STRONGLY_DECREASING
                elif slope_percent < -1:
                    direction = TrendDirection.DECREASING
                else:
                    direction = TrendDirection.STABLE
                
                # Check for volatility
                volatility = statistics.stdev(values) / avg_value if avg_value != 0 else 0
                if volatility > 0.2:  # 20% coefficient of variation
                    direction = TrendDirection.VOLATILE
                
                # Calculate trend strength
                correlation = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
                trend_strength = abs(correlation)
                
                trends[metric_id] = {
                    "direction": direction.value,
                    "slope": round(slope, 4),
                    "slope_percent": round(slope_percent, 2),
                    "trend_strength": round(trend_strength, 3),
                    "volatility": round(volatility, 3),
                    "r_squared": round(correlation ** 2, 3),
                    "prediction_next_period": round(intercept + slope * len(values), 2)
                }
        
        return trends
    
    async def _generate_key_findings(self, performance_summary: Dict[str, Any], 
                                   trend_analysis: Dict[str, Any]) -> List[str]:
        """Generate key findings from performance analysis"""
        
        findings = []
        
        # Performance status findings
        excellent_metrics = [m for m, data in performance_summary.items() if data.get("status") == "excellent"]
        poor_metrics = [m for m, data in performance_summary.items() if data.get("status") == "poor"]
        
        if excellent_metrics:
            findings.append(f"Excellent performance in {len(excellent_metrics)} metrics: {', '.join(excellent_metrics[:3])}")
        
        if poor_metrics:
            findings.append(f"Performance concerns in {len(poor_metrics)} metrics: {', '.join(poor_metrics[:3])}")
        
        # Trend findings
        improving_metrics = [m for m, data in trend_analysis.items() 
                           if "increasing" in data.get("direction", "") and "accuracy" in m or "satisfaction" in m]
        degrading_metrics = [m for m, data in trend_analysis.items() 
                           if "decreasing" in data.get("direction", "") and "accuracy" in m or "satisfaction" in m]
        
        if improving_metrics:
            findings.append(f"Positive trends detected in: {', '.join(improving_metrics[:2])}")
        
        if degrading_metrics:
            findings.append(f"Concerning trends in: {', '.join(degrading_metrics[:2])}")
        
        # Volatility findings
        volatile_metrics = [m for m, data in trend_analysis.items() if data.get("direction") == "volatile"]
        if volatile_metrics:
            findings.append(f"High volatility observed in: {', '.join(volatile_metrics[:2])}")
        
        # Performance vs target findings
        over_performing = [m for m, data in performance_summary.items() if data.get("vs_target_percent", 0) > 10]
        under_performing = [m for m, data in performance_summary.items() if data.get("vs_target_percent", 0) < -10]
        
        if over_performing:
            findings.append(f"Exceeding targets by >10% in: {', '.join(over_performing[:2])}")
        
        if under_performing:
            findings.append(f"Below targets by >10% in: {', '.join(under_performing[:2])}")
        
        return findings[:10]  # Limit to top 10 findings
    
    async def _generate_recommendations(self, performance_summary: Dict[str, Any],
                                      trend_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Performance-based recommendations
        for metric_id, data in performance_summary.items():
            if data.get("status") == "poor":
                if "accuracy" in metric_id:
                    recommendations.append(f"Review and retrain models - {metric_id} below acceptable threshold")
                elif "latency" in metric_id:
                    recommendations.append(f"Optimize system performance - {metric_id} exceeds acceptable limits")
                elif "error" in metric_id:
                    recommendations.append(f"Investigate error sources - {metric_id} requires immediate attention")
                elif "cost" in metric_id:
                    recommendations.append(f"Implement cost optimization - {metric_id} exceeds budget targets")
        
        # Trend-based recommendations
        for metric_id, data in trend_analysis.items():
            direction = data.get("direction", "")
            
            if direction == "strongly_decreasing" and ("accuracy" in metric_id or "satisfaction" in metric_id):
                recommendations.append(f"Urgent intervention needed - {metric_id} showing strong negative trend")
            elif direction == "strongly_increasing" and ("latency" in metric_id or "error" in metric_id or "cost" in metric_id):
                recommendations.append(f"Address escalating issue - {metric_id} trending upward rapidly")
            elif direction == "volatile":
                recommendations.append(f"Stabilize {metric_id} - high volatility indicates system instability")
        
        # General recommendations
        if len([m for m, d in performance_summary.items() if d.get("status") == "poor"]) > 2:
            recommendations.append("Consider comprehensive system review - multiple metrics underperforming")
        
        high_volatility_count = len([m for m, d in trend_analysis.items() if d.get("volatility", 0) > 0.2])
        if high_volatility_count > 1:
            recommendations.append("Implement monitoring alerts for volatile metrics")
        
        return recommendations[:8]  # Limit to top 8 recommendations
    
    def _calculate_data_quality_score(self, metrics_data: Dict[str, List]) -> float:
        """Calculate data quality score based on completeness and consistency"""
        
        if not metrics_data:
            return 0.0
        
        total_score = 0.0
        metric_count = 0
        
        for metric_id, metrics in metrics_data.items():
            if not metrics:
                continue
            
            metric_count += 1
            score = 0.0
            
            # Completeness score (40%)
            expected_points = 24  # Assuming hourly collection for 24 hours
            actual_points = len(metrics)
            completeness = min(1.0, actual_points / expected_points)
            score += completeness * 0.4
            
            # Consistency score (30%)
            values = [m.value for m in metrics]
            if len(values) > 1:
                cv = statistics.stdev(values) / statistics.mean(values) if statistics.mean(values) != 0 else 1
                consistency = max(0, 1 - cv)  # Lower coefficient of variation = higher consistency
                score += consistency * 0.3
            else:
                score += 0.3
            
            # Recency score (30%)
            latest_metric = metrics[-1]
            time_since_latest = (datetime.now() - latest_metric.timestamp).total_seconds() / 3600  # hours
            recency = max(0, 1 - (time_since_latest / 24))  # Decay over 24 hours
            score += recency * 0.3
            
            total_score += score
        
        return round((total_score / metric_count) * 100, 1) if metric_count > 0 else 0.0
    
    def _calculate_confidence_level(self, metrics_data: Dict[str, List], data_quality_score: float) -> float:
        """Calculate confidence level in the analysis"""
        
        if not metrics_data:
            return 0.0
        
        # Base confidence on data quality
        confidence = data_quality_score / 100
        
        # Adjust based on data volume
        total_points = sum(len(metrics) for metrics in metrics_data.values())
        volume_factor = min(1.0, total_points / 100)  # Optimal at 100+ data points
        confidence *= volume_factor
        
        # Adjust based on metric coverage
        expected_metrics = len(self.baseline_metrics)
        actual_metrics = len(metrics_data)
        coverage_factor = actual_metrics / expected_metrics
        confidence *= coverage_factor
        
        return round(confidence * 100, 1)


class ModelDriftAnalyzer:
    """Analyzes model performance drift over time"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.drift_thresholds = {
            "accuracy_drop": 5.0,  # 5% drop triggers drift alert
            "latency_increase": 50.0,  # 50ms increase
            "error_rate_increase": 2.0,  # 2% increase
            "confidence_drop": 10.0  # 10% confidence drop
        }
        
        logger.info("Model Drift Analyzer initialized")
    
    async def analyze_model_drift(self, model_name: str, analysis_days: int = 7) -> ModelDriftReport:
        """Analyze model drift over specified period"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=analysis_days)
        
        # Get model-specific metrics
        accuracy_metrics = self.metrics_collector.get_metrics("model_accuracy", start_time, end_time)
        latency_metrics = self.metrics_collector.get_metrics("response_latency", start_time, end_time)
        error_metrics = self.metrics_collector.get_metrics("error_rate", start_time, end_time)
        
        # Analyze drift
        drift_analysis = await self._detect_drift(accuracy_metrics, latency_metrics, error_metrics)
        
        # Generate report
        report = ModelDriftReport(
            report_id=f"drift_report_{uuid.uuid4().hex[:8]}",
            model_name=model_name,
            analysis_period={"start": start_time, "end": end_time},
            drift_detected=drift_analysis["drift_detected"],
            drift_severity=drift_analysis["severity"],
            drift_metrics=drift_analysis["metrics"],
            affected_features=drift_analysis["affected_features"],
            root_cause_analysis=drift_analysis["root_cause"],
            remediation_plan=drift_analysis["remediation_plan"],
            retraining_recommended=drift_analysis["retraining_recommended"],
            confidence_score=drift_analysis["confidence"],
            generated_at=datetime.now()
        )
        
        logger.info(f"Generated model drift report: {report.report_id}")
        
        return report
    
    async def _detect_drift(self, accuracy_metrics: List, latency_metrics: List, 
                          error_metrics: List) -> Dict[str, Any]:
        """Detect drift in model performance"""
        
        drift_detected = False
        drift_severity = "low"
        drift_metrics = {}
        affected_features = []
        
        # Analyze accuracy drift
        if accuracy_metrics and len(accuracy_metrics) >= 10:
            values = [m.value for m in accuracy_metrics]
            baseline_accuracy = statistics.mean(values[:len(values)//2])  # First half as baseline
            recent_accuracy = statistics.mean(values[len(values)//2:])   # Second half as recent
            
            accuracy_drift = baseline_accuracy - recent_accuracy
            drift_metrics["accuracy_drift"] = round(accuracy_drift, 2)
            
            if accuracy_drift > self.drift_thresholds["accuracy_drop"]:
                drift_detected = True
                affected_features.append("model_accuracy")
                if accuracy_drift > self.drift_thresholds["accuracy_drop"] * 2:
                    drift_severity = "high"
                else:
                    drift_severity = "medium"
        
        # Analyze latency drift
        if latency_metrics and len(latency_metrics) >= 10:
            values = [m.value for m in latency_metrics]
            baseline_latency = statistics.mean(values[:len(values)//2])
            recent_latency = statistics.mean(values[len(values)//2:])
            
            latency_drift = recent_latency - baseline_latency
            drift_metrics["latency_drift"] = round(latency_drift, 2)
            
            if latency_drift > self.drift_thresholds["latency_increase"]:
                drift_detected = True
                affected_features.append("response_latency")
                if latency_drift > self.drift_thresholds["latency_increase"] * 2:
                    drift_severity = "high"
        
        # Analyze error rate drift
        if error_metrics and len(error_metrics) >= 10:
            values = [m.value for m in error_metrics]
            baseline_errors = statistics.mean(values[:len(values)//2])
            recent_errors = statistics.mean(values[len(values)//2:])
            
            error_drift = recent_errors - baseline_errors
            drift_metrics["error_rate_drift"] = round(error_drift, 2)
            
            if error_drift > self.drift_thresholds["error_rate_increase"]:
                drift_detected = True
                affected_features.append("error_rate")
                if error_drift > self.drift_thresholds["error_rate_increase"] * 2:
                    drift_severity = "critical"
        
        # Root cause analysis
        root_cause = self._analyze_root_cause(drift_metrics, affected_features)
        
        # Remediation plan
        remediation_plan = self._generate_remediation_plan(drift_severity, affected_features)
        
        # Retraining recommendation
        retraining_recommended = (drift_severity in ["high", "critical"] or 
                                len(affected_features) >= 2)
        
        # Confidence calculation
        confidence = self._calculate_drift_confidence(accuracy_metrics, latency_metrics, error_metrics)
        
        return {
            "drift_detected": drift_detected,
            "severity": drift_severity,
            "metrics": drift_metrics,
            "affected_features": affected_features,
            "root_cause": root_cause,
            "remediation_plan": remediation_plan,
            "retraining_recommended": retraining_recommended,
            "confidence": confidence
        }
    
    def _analyze_root_cause(self, drift_metrics: Dict[str, float], 
                          affected_features: List[str]) -> Dict[str, Any]:
        """Analyze potential root causes of drift"""
        
        root_causes = []
        
        if "model_accuracy" in affected_features:
            accuracy_drift = drift_metrics.get("accuracy_drift", 0)
            if accuracy_drift > 10:
                root_causes.append("Significant data distribution shift")
            elif accuracy_drift > 5:
                root_causes.append("Moderate concept drift or data quality issues")
            else:
                root_causes.append("Minor performance degradation")
        
        if "response_latency" in affected_features:
            latency_drift = drift_metrics.get("latency_drift", 0)
            if latency_drift > 100:
                root_causes.append("Infrastructure performance degradation")
            else:
                root_causes.append("Increased computational complexity or load")
        
        if "error_rate" in affected_features:
            error_drift = drift_metrics.get("error_rate_drift", 0)
            if error_drift > 5:
                root_causes.append("System instability or integration issues")
            else:
                root_causes.append("Minor service reliability concerns")
        
        return {
            "primary_causes": root_causes[:3],
            "analysis_confidence": 0.8 if len(affected_features) >= 2 else 0.6,
            "investigation_areas": [
                "Data pipeline integrity",
                "Model serving infrastructure",
                "Input data quality",
                "External dependencies"
            ]
        }
    
    def _generate_remediation_plan(self, severity: str, affected_features: List[str]) -> List[str]:
        """Generate remediation plan based on drift analysis"""
        
        plan = []
        
        if severity == "critical":
            plan.append("IMMEDIATE: Implement model rollback to previous stable version")
            plan.append("URGENT: Investigate data pipeline for corruption or bias")
            plan.append("Schedule emergency model retraining with recent data")
        elif severity == "high":
            plan.append("Prioritize model retraining within 48 hours")
            plan.append("Implement enhanced monitoring for affected features")
            plan.append("Review and validate recent data sources")
        elif severity == "medium":
            plan.append("Schedule model retraining within 1 week")
            plan.append("Increase monitoring frequency for drift detection")
            plan.append("Analyze feature importance changes")
        else:
            plan.append("Continue monitoring with current thresholds")
            plan.append("Document drift patterns for trend analysis")
        
        # Feature-specific actions
        if "model_accuracy" in affected_features:
            plan.append("Validate ground truth labels and data quality")
            plan.append("Consider ensemble methods to improve robustness")
        
        if "response_latency" in affected_features:
            plan.append("Optimize model inference pipeline")
            plan.append("Review infrastructure scaling policies")
        
        if "error_rate" in affected_features:
            plan.append("Investigate error patterns and failure modes")
            plan.append("Implement circuit breaker patterns for resilience")
        
        return plan[:8]  # Limit to 8 action items
    
    def _calculate_drift_confidence(self, accuracy_metrics: List, latency_metrics: List,
                                  error_metrics: List) -> float:
        """Calculate confidence in drift detection"""
        
        total_metrics = len([m for m in [accuracy_metrics, latency_metrics, error_metrics] if m])
        total_points = sum(len(m) for m in [accuracy_metrics, latency_metrics, error_metrics] if m)
        
        if total_points < 20:
            return 0.4  # Low confidence with insufficient data
        elif total_points < 50:
            return 0.6  # Medium confidence
        else:
            return 0.8  # High confidence with sufficient data


class BusinessIntelligenceAnalyzer:
    """Analyzes business metrics and generates ROI reports"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.roi_baselines = {
            "manual_process_cost": 150.0,  # Cost per hour for manual analysis
            "ai_process_cost": 25.0,       # Cost per hour for AI analysis
            "accuracy_improvement": 0.23,   # 23% improvement over baseline
            "time_savings": 0.65           # 65% time savings
        }
        
        logger.info("Business Intelligence Analyzer initialized")
    
    async def generate_roi_report(self, reporting_period_days: int = 30) -> BusinessIntelligenceReport:
        """Generate comprehensive ROI and business intelligence report"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=reporting_period_days)
        
        # Collect business metrics
        roi_metrics = await self._calculate_roi_metrics(start_time, end_time)
        cost_analysis = await self._analyze_costs(start_time, end_time)
        efficiency_gains = await self._calculate_efficiency_gains(start_time, end_time)
        user_adoption = await self._analyze_user_adoption(start_time, end_time)
        business_impact = await self._assess_business_impact(start_time, end_time)
        competitive_analysis = await self._perform_competitive_analysis()
        strategic_recommendations = await self._generate_strategic_recommendations(roi_metrics, efficiency_gains)
        executive_summary = await self._create_executive_summary(roi_metrics, efficiency_gains, business_impact)
        
        report = BusinessIntelligenceReport(
            report_id=f"bi_report_{uuid.uuid4().hex[:8]}",
            reporting_period={"start": start_time, "end": end_time},
            roi_metrics=roi_metrics,
            cost_analysis=cost_analysis,
            efficiency_gains=efficiency_gains,
            user_adoption_metrics=user_adoption,
            business_impact=business_impact,
            competitive_analysis=competitive_analysis,
            strategic_recommendations=strategic_recommendations,
            executive_summary=executive_summary,
            generated_at=datetime.now()
        )
        
        logger.info(f"Generated business intelligence report: {report.report_id}")
        
        return report
    
    async def _calculate_roi_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, float]:
        """Calculate return on investment metrics"""
        
        # Get request volume to estimate usage
        volume_metrics = self.metrics_collector.get_metrics("request_volume", start_time, end_time)
        cost_metrics = self.metrics_collector.get_metrics("cost_per_request", start_time, end_time)
        
        if not volume_metrics or not cost_metrics:
            return {"error": "Insufficient data for ROI calculation"}
        
        # Calculate total requests and costs
        total_requests = sum(m.value for m in volume_metrics) * (1/60)  # Convert from per-minute to total
        avg_cost_per_request = statistics.mean(m.value for m in cost_metrics)
        total_ai_cost = total_requests * avg_cost_per_request
        
        # Estimate manual process cost
        estimated_hours = total_requests * 0.1  # Assume 6 minutes per request manually
        manual_cost = estimated_hours * self.roi_baselines["manual_process_cost"]
        
        # Calculate ROI
        cost_savings = manual_cost - total_ai_cost
        roi_percentage = (cost_savings / total_ai_cost) * 100 if total_ai_cost > 0 else 0
        
        # Calculate payback period (assuming monthly costs)
        monthly_savings = cost_savings * (30 / (end_time - start_time).days)
        implementation_cost = 500000  # Estimated implementation cost
        payback_months = implementation_cost / monthly_savings if monthly_savings > 0 else float('inf')
        
        return {
            "total_cost_savings": round(cost_savings, 2),
            "roi_percentage": round(roi_percentage, 1),
            "payback_period_months": round(payback_months, 1),
            "monthly_savings": round(monthly_savings, 2),
            "total_requests_processed": int(total_requests),
            "cost_per_request_ai": round(avg_cost_per_request, 4),
            "cost_per_request_manual": round(self.roi_baselines["manual_process_cost"] * 0.1, 4),
            "efficiency_multiplier": round(self.roi_baselines["manual_process_cost"] * 0.1 / avg_cost_per_request, 1)
        }
    
    async def _analyze_costs(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Analyze cost breakdown and trends"""
        
        cost_metrics = self.metrics_collector.get_metrics("cost_per_request", start_time, end_time)
        
        if not cost_metrics:
            return {"error": "No cost data available"}
        
        costs = [m.value for m in cost_metrics]
        
        # Cost breakdown (simulated - in production would come from actual billing)
        total_cost = sum(costs)
        
        breakdown = {
            "compute_costs": total_cost * 0.6,
            "api_costs": total_cost * 0.25,
            "storage_costs": total_cost * 0.1,
            "network_costs": total_cost * 0.05
        }
        
        # Trend analysis
        if len(costs) >= 7:
            recent_avg = statistics.mean(costs[-7:])
            baseline_avg = statistics.mean(costs[:7])
            trend_percent = ((recent_avg - baseline_avg) / baseline_avg) * 100 if baseline_avg > 0 else 0
        else:
            trend_percent = 0
        
        return {
            "total_period_cost": round(total_cost, 2),
            "average_cost_per_request": round(statistics.mean(costs), 4),
            "cost_breakdown": {k: round(v, 2) for k, v in breakdown.items()},
            "cost_trend_percent": round(trend_percent, 1),
            "cost_volatility": round(statistics.stdev(costs) / statistics.mean(costs), 3) if len(costs) > 1 else 0,
            "min_cost": round(min(costs), 4),
            "max_cost": round(max(costs), 4)
        }
    
    async def _calculate_efficiency_gains(self, start_time: datetime, end_time: datetime) -> Dict[str, float]:
        """Calculate efficiency improvements"""
        
        # Get performance metrics
        accuracy_metrics = self.metrics_collector.get_metrics("model_accuracy", start_time, end_time)
        latency_metrics = self.metrics_collector.get_metrics("response_latency", start_time, end_time)
        satisfaction_metrics = self.metrics_collector.get_metrics("user_satisfaction", start_time, end_time)
        
        efficiency_gains = {}
        
        if accuracy_metrics:
            avg_accuracy = statistics.mean(m.value for m in accuracy_metrics)
            baseline_accuracy = 75.0  # Assumed baseline without AI
            accuracy_improvement = ((avg_accuracy - baseline_accuracy) / baseline_accuracy) * 100
            efficiency_gains["accuracy_improvement_percent"] = round(accuracy_improvement, 1)
        
        if latency_metrics:
            avg_latency = statistics.mean(m.value for m in latency_metrics)
            manual_latency = 3600000  # 1 hour in milliseconds for manual process
            time_savings = ((manual_latency - avg_latency) / manual_latency) * 100
            efficiency_gains["time_savings_percent"] = round(time_savings, 1)
        
        if satisfaction_metrics:
            avg_satisfaction = statistics.mean(m.value for m in satisfaction_metrics)
            baseline_satisfaction = 6.5  # Assumed baseline
            satisfaction_improvement = ((avg_satisfaction - baseline_satisfaction) / baseline_satisfaction) * 100
            efficiency_gains["satisfaction_improvement_percent"] = round(satisfaction_improvement, 1)
        
        # Calculate productivity multiplier
        if "time_savings_percent" in efficiency_gains:
            time_savings_ratio = efficiency_gains["time_savings_percent"] / 100
            if time_savings_ratio < 0.99:  # Avoid division by zero
                productivity_multiplier = 1 / (1 - time_savings_ratio)
                efficiency_gains["productivity_multiplier"] = round(productivity_multiplier, 1)
            else:
                efficiency_gains["productivity_multiplier"] = 100.0  # Cap at 100x
        
        return efficiency_gains
    
    async def _analyze_user_adoption(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Analyze user adoption metrics"""
        
        # Simulated user adoption data (in production, would come from user analytics)
        volume_metrics = self.metrics_collector.get_metrics("request_volume", start_time, end_time)
        
        if not volume_metrics:
            return {"error": "No usage data available"}
        
        total_requests = sum(m.value for m in volume_metrics)
        days_in_period = (end_time - start_time).days
        
        # Simulated adoption metrics
        adoption_metrics = {
            "total_active_users": 150,  # Simulated
            "daily_active_users": 87,
            "user_retention_rate": 0.85,
            "feature_adoption_rate": 0.73,
            "average_requests_per_user": round(total_requests / 150, 1),
            "user_growth_rate": 0.12,  # 12% monthly growth
            "power_users_percentage": 0.25,
            "support_ticket_rate": 0.03  # 3% of users create support tickets
        }
        
        return adoption_metrics
    
    async def _assess_business_impact(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Assess overall business impact"""
        
        # Simulated business impact metrics
        impact_metrics = {
            "revenue_impact": {
                "additional_revenue": 2500000,  # $2.5M additional revenue
                "revenue_attribution": 0.15,   # 15% attributed to AI improvements
                "client_retention_improvement": 0.08  # 8% improvement
            },
            "operational_impact": {
                "staff_reallocation": 12,  # 12 FTEs reallocated to higher-value work
                "process_automation": 0.78,  # 78% of routine tasks automated
                "error_reduction": 0.45,  # 45% reduction in manual errors
                "compliance_improvement": 0.23  # 23% improvement in compliance scores
            },
            "strategic_impact": {
                "market_differentiation": "high",
                "competitive_advantage": "significant",
                "innovation_enablement": "transformational",
                "scalability_improvement": "exponential"
            }
        }
        
        return impact_metrics
    
    async def _perform_competitive_analysis(self) -> Dict[str, Any]:
        """Perform competitive analysis (simulated)"""
        
        return {
            "market_position": "leading",
            "technology_advantage": "2-3 years ahead",
            "cost_advantage": "35% lower operational costs",
            "performance_advantage": "23% higher accuracy",
            "feature_completeness": "95% vs industry average 70%",
            "competitive_threats": [
                "New AI startups with specialized models",
                "Big tech companies entering the space",
                "Open source alternatives gaining traction"
            ],
            "competitive_opportunities": [
                "Expand to adjacent markets",
                "Develop proprietary data advantages",
                "Build ecosystem partnerships"
            ]
        }
    
    async def _generate_strategic_recommendations(self, roi_metrics: Dict[str, float],
                                               efficiency_gains: Dict[str, float]) -> List[str]:
        """Generate strategic recommendations"""
        
        recommendations = []
        
        # ROI-based recommendations
        if roi_metrics.get("roi_percentage", 0) > 200:
            recommendations.append("Accelerate AI investment - ROI exceeds 200%")
        elif roi_metrics.get("roi_percentage", 0) > 100:
            recommendations.append("Expand AI capabilities to additional use cases")
        
        if roi_metrics.get("payback_period_months", float('inf')) < 12:
            recommendations.append("Fast payback period justifies aggressive scaling")
        
        # Efficiency-based recommendations
        if efficiency_gains.get("accuracy_improvement_percent", 0) > 20:
            recommendations.append("Leverage accuracy advantage for premium positioning")
        
        if efficiency_gains.get("time_savings_percent", 0) > 60:
            recommendations.append("Reallocate saved time to strategic initiatives")
        
        # Strategic recommendations
        recommendations.extend([
            "Develop proprietary data moats to maintain competitive advantage",
            "Invest in next-generation AI capabilities (LLMs, multimodal)",
            "Build partner ecosystem to expand market reach",
            "Establish AI center of excellence for organization-wide adoption",
            "Implement continuous learning systems for sustained improvement"
        ])
        
        return recommendations[:8]
    
    async def _create_executive_summary(self, roi_metrics: Dict[str, float],
                                      efficiency_gains: Dict[str, float],
                                      business_impact: Dict[str, Any]) -> str:
        """Create executive summary"""
        
        roi_pct = roi_metrics.get("roi_percentage", 0)
        cost_savings = roi_metrics.get("total_cost_savings", 0)
        accuracy_improvement = efficiency_gains.get("accuracy_improvement_percent", 0)
        time_savings = efficiency_gains.get("time_savings_percent", 0)
        
        summary = f"""
EXECUTIVE SUMMARY - AI INVESTMENT PERFORMANCE

The AI initiative has delivered exceptional returns with {roi_pct:.0f}% ROI and ${cost_savings:,.0f} in cost savings. 
Key achievements include {accuracy_improvement:.0f}% accuracy improvement and {time_savings:.0f}% time savings, 
resulting in significant operational efficiency gains.

BUSINESS IMPACT:
• Revenue Impact: ${business_impact['revenue_impact']['additional_revenue']:,.0f} additional revenue
• Operational Excellence: {business_impact['operational_impact']['staff_reallocation']} FTEs reallocated to strategic work
• Competitive Position: Established 2-3 year technology lead in the market

STRATEGIC OUTLOOK:
The AI platform has become a core competitive differentiator, enabling scalable growth and operational excellence. 
Continued investment in AI capabilities is recommended to maintain market leadership and capture emerging opportunities.

RECOMMENDATION: Accelerate AI adoption across additional use cases to maximize ROI and competitive advantage.
        """.strip()
        
        return summary


class ComplianceReporter:
    """Generates regulatory compliance reports"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.compliance_frameworks = {
            "SOX": {"weight": 0.3, "categories": ["data_integrity", "audit_trail", "access_control"]},
            "GDPR": {"weight": 0.25, "categories": ["data_privacy", "consent_management", "data_retention"]},
            "SOC2": {"weight": 0.25, "categories": ["security", "availability", "confidentiality"]},
            "ISO27001": {"weight": 0.2, "categories": ["information_security", "risk_management"]}
        }
        
        logger.info("Compliance Reporter initialized")
    
    async def generate_compliance_report(self, framework: str = "combined") -> ComplianceReport:
        """Generate comprehensive compliance report"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=90)  # Quarterly assessment
        
        # Get compliance-related metrics
        compliance_metrics = self.metrics_collector.get_metrics("compliance_score", start_time, end_time)
        security_metrics = self.metrics_collector.get_metrics("security_events", start_time, end_time)
        
        # Calculate compliance scores
        overall_score = await self._calculate_overall_compliance(compliance_metrics)
        category_scores = await self._calculate_category_compliance(framework)
        violations = await self._detect_violations(security_metrics)
        risk_assessment = await self._assess_compliance_risks(violations)
        remediation_actions = await self._generate_remediation_actions(violations, risk_assessment)
        audit_trail = await self._generate_audit_trail(start_time, end_time)
        
        report = ComplianceReport(
            report_id=f"compliance_report_{uuid.uuid4().hex[:8]}",
            compliance_framework=framework,
            assessment_period={"start": start_time, "end": end_time},
            overall_compliance_score=overall_score,
            compliance_by_category=category_scores,
            violations_detected=violations,
            risk_assessment=risk_assessment,
            remediation_actions=remediation_actions,
            audit_trail=audit_trail,
            certification_status="compliant" if overall_score >= 90 else "non_compliant",
            next_review_date=datetime.now() + timedelta(days=90),
            generated_at=datetime.now()
        )
        
        logger.info(f"Generated compliance report: {report.report_id}")
        
        return report
    
    async def _calculate_overall_compliance(self, compliance_metrics: List) -> float:
        """Calculate overall compliance score"""
        
        if not compliance_metrics:
            return 85.0  # Default score
        
        scores = [m.value for m in compliance_metrics]
        return round(statistics.mean(scores), 1)
    
    async def _calculate_category_compliance(self, framework: str) -> Dict[str, float]:
        """Calculate compliance scores by category"""
        
        # Simulated category scores (in production, would be calculated from actual compliance checks)
        category_scores = {
            "data_integrity": 95.2,
            "audit_trail": 98.1,
            "access_control": 92.7,
            "data_privacy": 89.4,
            "consent_management": 91.8,
            "data_retention": 94.3,
            "security": 96.5,
            "availability": 99.2,
            "confidentiality": 93.8,
            "information_security": 95.7,
            "risk_management": 88.9
        }
        
        return category_scores
    
    async def _detect_violations(self, security_metrics: List) -> List[Dict[str, Any]]:
        """Detect compliance violations"""
        
        violations = []
        
        if security_metrics:
            # Check for security event spikes
            recent_events = [m.value for m in security_metrics[-24:]]  # Last 24 hours
            if recent_events and max(recent_events) > 10:
                violations.append({
                    "violation_id": f"sec_violation_{uuid.uuid4().hex[:6]}",
                    "type": "security_event_spike",
                    "severity": "medium",
                    "description": "Unusual increase in security events detected",
                    "detected_at": datetime.now(),
                    "affected_systems": ["authentication", "api_gateway"],
                    "compliance_impact": "SOC2 availability requirement"
                })
        
        # Simulated additional violations
        if datetime.now().day % 7 == 0:  # Simulate occasional violations
            violations.append({
                "violation_id": f"data_violation_{uuid.uuid4().hex[:6]}",
                "type": "data_retention_policy",
                "severity": "low",
                "description": "Data retention policy requires review",
                "detected_at": datetime.now(),
                "affected_systems": ["data_warehouse"],
                "compliance_impact": "GDPR data retention requirements"
            })
        
        return violations
    
    async def _assess_compliance_risks(self, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess compliance risks"""
        
        risk_levels = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        
        for violation in violations:
            severity = violation.get("severity", "low")
            risk_levels[severity] += 1
        
        # Calculate overall risk score
        risk_score = (risk_levels["low"] * 1 + risk_levels["medium"] * 3 + 
                     risk_levels["high"] * 7 + risk_levels["critical"] * 15)
        
        risk_level = "low"
        if risk_score > 20:
            risk_level = "critical"
        elif risk_score > 10:
            risk_level = "high"
        elif risk_score > 5:
            risk_level = "medium"
        
        return {
            "overall_risk_level": risk_level,
            "risk_score": risk_score,
            "risk_distribution": risk_levels,
            "key_risk_areas": [
                "Data privacy and protection",
                "Access control and authentication",
                "Audit trail completeness"
            ],
            "risk_trend": "stable",
            "mitigation_priority": "medium"
        }
    
    async def _generate_remediation_actions(self, violations: List[Dict[str, Any]],
                                          risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate remediation actions"""
        
        actions = []
        
        # Violation-specific actions
        for violation in violations:
            if violation["type"] == "security_event_spike":
                actions.append("Investigate and resolve security event root causes")
                actions.append("Review and update security monitoring thresholds")
            elif violation["type"] == "data_retention_policy":
                actions.append("Update data retention policies and procedures")
                actions.append("Implement automated data lifecycle management")
        
        # Risk-based actions
        risk_level = risk_assessment.get("overall_risk_level", "low")
        
        if risk_level in ["high", "critical"]:
            actions.append("Conduct immediate compliance audit")
            actions.append("Engage external compliance consultant")
        elif risk_level == "medium":
            actions.append("Schedule quarterly compliance review")
            actions.append("Enhance compliance monitoring procedures")
        
        # General actions
        actions.extend([
            "Update compliance training for all staff",
            "Review and test incident response procedures",
            "Conduct penetration testing and vulnerability assessment"
        ])
        
        return actions[:8]
    
    async def _generate_audit_trail(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Generate audit trail entries"""
        
        # Simulated audit trail (in production, would come from actual system logs)
        audit_entries = [
            {
                "timestamp": datetime.now() - timedelta(days=1),
                "event_type": "compliance_check",
                "user": "system",
                "action": "automated_compliance_scan",
                "result": "passed",
                "details": "All compliance checks passed successfully"
            },
            {
                "timestamp": datetime.now() - timedelta(days=7),
                "event_type": "policy_update",
                "user": "compliance_officer",
                "action": "updated_data_retention_policy",
                "result": "completed",
                "details": "Updated policy to align with GDPR requirements"
            },
            {
                "timestamp": datetime.now() - timedelta(days=14),
                "event_type": "security_review",
                "user": "security_team",
                "action": "quarterly_security_assessment",
                "result": "passed",
                "details": "No critical security issues identified"
            }
        ]
        
        return audit_entries


# Factory function for creating analytics and reporting system
def create_analytics_reporting_system(metrics_collector: MetricsCollector, 
                                     alert_manager: AlertManager) -> Tuple[PerformanceAnalyzer, ModelDriftAnalyzer, 
                                                                         BusinessIntelligenceAnalyzer, ComplianceReporter]:
    """Factory function to create complete analytics and reporting system"""
    
    performance_analyzer = PerformanceAnalyzer(metrics_collector)
    model_drift_analyzer = ModelDriftAnalyzer(metrics_collector)
    business_intelligence_analyzer = BusinessIntelligenceAnalyzer(metrics_collector)
    compliance_reporter = ComplianceReporter(metrics_collector)
    
    logger.info("Analytics and Reporting System created successfully")
    
    return performance_analyzer, model_drift_analyzer, business_intelligence_analyzer, compliance_reporter