"""
Comprehensive Monitoring and Analytics System
BlackRock Aladdin-inspired real-time monitoring, anomaly detection, and business intelligence
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

from app.core.hybrid_ai_engine import ModelOutput, TaskCategory, ModelType

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics to monitor"""
    PERFORMANCE = "performance"
    BUSINESS = "business"
    TECHNICAL = "technical"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    USER_BEHAVIOR = "user_behavior"


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class AnomalyType(Enum):
    """Types of anomalies detected"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ACCURACY_DROP = "accuracy_drop"
    LATENCY_SPIKE = "latency_spike"
    ERROR_RATE_INCREASE = "error_rate_increase"
    USAGE_ANOMALY = "usage_anomaly"
    COST_SPIKE = "cost_spike"
    SECURITY_THREAT = "security_threat"


@dataclass
class Metric:
    """Individual metric data point"""
    metric_id: str
    metric_name: str
    metric_type: MetricType
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str]
    metadata: Dict[str, Any]


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    metric_id: Optional[str]
    threshold_value: Optional[float]
    actual_value: Optional[float]
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime]
    assigned_to: Optional[str]
    tags: Dict[str, str]
    metadata: Dict[str, Any]


@dataclass
class Anomaly:
    """Detected anomaly"""
    anomaly_id: str
    anomaly_type: AnomalyType
    description: str
    confidence: float
    severity: AlertSeverity
    detected_at: datetime
    metric_ids: List[str]
    affected_components: List[str]
    root_cause_analysis: Dict[str, Any]
    recommended_actions: List[str]
    metadata: Dict[str, Any]


@dataclass
class Dashboard:
    """Dashboard configuration"""
    dashboard_id: str
    name: str
    description: str
    dashboard_type: str  # "executive", "technical", "operational"
    widgets: List[Dict[str, Any]]
    refresh_interval: int
    permissions: List[str]
    created_at: datetime
    updated_at: datetime


class MetricsCollector:
    """Collects and stores system metrics"""
    
    def __init__(self):
        self.metrics_store: Dict[str, List[Metric]] = defaultdict(list)
        self.metric_definitions: Dict[str, Dict[str, Any]] = {}
        self.collection_intervals: Dict[str, int] = {}
        self.active_collectors: Dict[str, bool] = {}
        
        # Initialize core metrics
        self._initialize_core_metrics()
        
        logger.info("Metrics Collector initialized")
    
    def _initialize_core_metrics(self):
        """Initialize core system metrics"""
        
        core_metrics = {
            "model_accuracy": {
                "name": "Model Accuracy",
                "type": MetricType.PERFORMANCE,
                "unit": "percentage",
                "description": "Model prediction accuracy",
                "collection_interval": 300  # 5 minutes
            },
            "response_latency": {
                "name": "Response Latency",
                "type": MetricType.PERFORMANCE,
                "unit": "milliseconds",
                "description": "API response time",
                "collection_interval": 60  # 1 minute
            },
            "request_volume": {
                "name": "Request Volume",
                "type": MetricType.BUSINESS,
                "unit": "requests/minute",
                "description": "Number of requests per minute",
                "collection_interval": 60
            },
            "error_rate": {
                "name": "Error Rate",
                "type": MetricType.TECHNICAL,
                "unit": "percentage",
                "description": "Percentage of failed requests",
                "collection_interval": 60
            },
            "cost_per_request": {
                "name": "Cost Per Request",
                "type": MetricType.BUSINESS,
                "unit": "dollars",
                "description": "Average cost per API request",
                "collection_interval": 300
            },
            "user_satisfaction": {
                "name": "User Satisfaction",
                "type": MetricType.BUSINESS,
                "unit": "score",
                "description": "User satisfaction score (1-10)",
                "collection_interval": 3600  # 1 hour
            },
            "security_events": {
                "name": "Security Events",
                "type": MetricType.SECURITY,
                "unit": "events/hour",
                "description": "Security-related events per hour",
                "collection_interval": 300
            },
            "compliance_score": {
                "name": "Compliance Score",
                "type": MetricType.COMPLIANCE,
                "unit": "percentage",
                "description": "Regulatory compliance score",
                "collection_interval": 3600
            }
        }
        
        for metric_id, config in core_metrics.items():
            self.register_metric(metric_id, config)
    
    def register_metric(self, metric_id: str, config: Dict[str, Any]):
        """Register a new metric for collection"""
        
        self.metric_definitions[metric_id] = config
        self.collection_intervals[metric_id] = config.get("collection_interval", 300)
        self.active_collectors[metric_id] = True
        
        logger.info(f"Registered metric: {config['name']} ({metric_id})")
    
    def collect_metric(self, metric_id: str, value: float, tags: Dict[str, str] = None,
                      metadata: Dict[str, Any] = None) -> Metric:
        """Collect a single metric data point"""
        
        if metric_id not in self.metric_definitions:
            raise ValueError(f"Unknown metric: {metric_id}")
        
        config = self.metric_definitions[metric_id]
        
        metric = Metric(
            metric_id=metric_id,
            metric_name=config["name"],
            metric_type=MetricType(config["type"]),
            value=value,
            unit=config["unit"],
            timestamp=datetime.now(),
            tags=tags or {},
            metadata=metadata or {}
        )
        
        # Store metric
        self.metrics_store[metric_id].append(metric)
        
        # Keep only last 10000 data points per metric
        if len(self.metrics_store[metric_id]) > 10000:
            self.metrics_store[metric_id] = self.metrics_store[metric_id][-10000:]
        
        return metric
    
    def get_metrics(self, metric_id: str, start_time: datetime = None,
                   end_time: datetime = None) -> List[Metric]:
        """Get metrics for a specific metric ID within time range"""
        
        metrics = self.metrics_store.get(metric_id, [])
        
        if start_time or end_time:
            filtered_metrics = []
            for metric in metrics:
                if start_time and metric.timestamp < start_time:
                    continue
                if end_time and metric.timestamp > end_time:
                    continue
                filtered_metrics.append(metric)
            return filtered_metrics
        
        return metrics
    
    def get_latest_metric(self, metric_id: str) -> Optional[Metric]:
        """Get the latest metric value"""
        
        metrics = self.metrics_store.get(metric_id, [])
        return metrics[-1] if metrics else None
    
    def get_metric_statistics(self, metric_id: str, window_minutes: int = 60) -> Dict[str, float]:
        """Get statistical summary of metrics within time window"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=window_minutes)
        
        metrics = self.get_metrics(metric_id, start_time, end_time)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "latest": values[-1],
            "trend": self._calculate_trend(values)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for values"""
        
        if len(values) < 2:
            return "stable"
        
        # Simple trend calculation using first and last quartile
        q1_end = len(values) // 4
        q4_start = 3 * len(values) // 4
        
        if q1_end >= q4_start:
            return "stable"
        
        q1_avg = statistics.mean(values[:q1_end]) if q1_end > 0 else values[0]
        q4_avg = statistics.mean(values[q4_start:]) if q4_start < len(values) else values[-1]
        
        change_percent = ((q4_avg - q1_avg) / q1_avg) * 100 if q1_avg != 0 else 0
        
        if change_percent > 5:
            return "increasing"
        elif change_percent < -5:
            return "decreasing"
        else:
            return "stable"
    
    async def start_collection(self):
        """Start automated metric collection"""
        
        logger.info("Starting automated metric collection")
        
        # Start collection tasks for each metric
        tasks = []
        for metric_id in self.metric_definitions.keys():
            if self.active_collectors.get(metric_id, False):
                task = asyncio.create_task(self._collect_metric_periodically(metric_id))
                tasks.append(task)
        
        # Wait for all collection tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _collect_metric_periodically(self, metric_id: str):
        """Collect a specific metric periodically"""
        
        interval = self.collection_intervals.get(metric_id, 300)
        
        while self.active_collectors.get(metric_id, False):
            try:
                # Simulate metric collection (in production, this would call actual systems)
                value = await self._simulate_metric_collection(metric_id)
                self.collect_metric(metric_id, value)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error collecting metric {metric_id}: {e}")
                await asyncio.sleep(interval)
    
    async def _simulate_metric_collection(self, metric_id: str) -> float:
        """Simulate metric collection for demo purposes"""
        
        # Generate realistic simulated values
        base_values = {
            "model_accuracy": 85.0,
            "response_latency": 250.0,
            "request_volume": 150.0,
            "error_rate": 2.0,
            "cost_per_request": 0.05,
            "user_satisfaction": 8.2,
            "security_events": 5.0,
            "compliance_score": 95.0
        }
        
        base_value = base_values.get(metric_id, 50.0)
        
        # Add some realistic variation
        variation = np.random.normal(0, base_value * 0.1)
        value = max(0, base_value + variation)
        
        # Add occasional spikes for demonstration
        if np.random.random() < 0.05:  # 5% chance of spike
            spike_multiplier = np.random.uniform(1.5, 3.0)
            value *= spike_multiplier
        
        return round(value, 2)
    
    def stop_collection(self, metric_id: str = None):
        """Stop metric collection"""
        
        if metric_id:
            self.active_collectors[metric_id] = False
            logger.info(f"Stopped collection for metric: {metric_id}")
        else:
            for mid in self.active_collectors.keys():
                self.active_collectors[mid] = False
            logger.info("Stopped all metric collection")
    
    def get_collection_status(self) -> Dict[str, Any]:
        """Get status of metric collection"""
        
        return {
            "total_metrics": len(self.metric_definitions),
            "active_collectors": sum(1 for active in self.active_collectors.values() if active),
            "total_data_points": sum(len(metrics) for metrics in self.metrics_store.values()),
            "metrics_status": {
                metric_id: {
                    "active": self.active_collectors.get(metric_id, False),
                    "data_points": len(self.metrics_store.get(metric_id, [])),
                    "latest_collection": self.get_latest_metric(metric_id).timestamp.isoformat() 
                                       if self.get_latest_metric(metric_id) else None
                }
                for metric_id in self.metric_definitions.keys()
            }
        }


class AnomalyDetector:
    """Detects anomalies in system metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.detection_rules: Dict[str, Dict[str, Any]] = {}
        self.anomaly_history: List[Anomaly] = []
        self.detection_enabled = True
        
        # Initialize detection rules
        self._initialize_detection_rules()
        
        logger.info("Anomaly Detector initialized")
    
    def _initialize_detection_rules(self):
        """Initialize anomaly detection rules"""
        
        detection_rules = {
            "model_accuracy_drop": {
                "metric_id": "model_accuracy",
                "rule_type": "threshold",
                "threshold": 80.0,
                "operator": "less_than",
                "severity": AlertSeverity.HIGH,
                "description": "Model accuracy dropped below acceptable threshold"
            },
            "response_latency_spike": {
                "metric_id": "response_latency",
                "rule_type": "statistical",
                "method": "z_score",
                "threshold": 3.0,
                "severity": AlertSeverity.MEDIUM,
                "description": "Response latency significantly higher than normal"
            },
            "error_rate_increase": {
                "metric_id": "error_rate",
                "rule_type": "threshold",
                "threshold": 5.0,
                "operator": "greater_than",
                "severity": AlertSeverity.HIGH,
                "description": "Error rate exceeded acceptable threshold"
            },
            "cost_spike": {
                "metric_id": "cost_per_request",
                "rule_type": "statistical",
                "method": "percentage_change",
                "threshold": 50.0,
                "severity": AlertSeverity.MEDIUM,
                "description": "Cost per request increased significantly"
            },
            "security_events_spike": {
                "metric_id": "security_events",
                "rule_type": "statistical",
                "method": "z_score",
                "threshold": 2.5,
                "severity": AlertSeverity.CRITICAL,
                "description": "Unusual increase in security events"
            }
        }
        
        for rule_id, rule_config in detection_rules.items():
            self.add_detection_rule(rule_id, rule_config)
    
    def add_detection_rule(self, rule_id: str, rule_config: Dict[str, Any]):
        """Add a new anomaly detection rule"""
        
        self.detection_rules[rule_id] = rule_config
        logger.info(f"Added detection rule: {rule_id}")
    
    async def detect_anomalies(self) -> List[Anomaly]:
        """Detect anomalies across all monitored metrics"""
        
        if not self.detection_enabled:
            return []
        
        detected_anomalies = []
        
        for rule_id, rule_config in self.detection_rules.items():
            try:
                anomaly = await self._apply_detection_rule(rule_id, rule_config)
                if anomaly:
                    detected_anomalies.append(anomaly)
                    self.anomaly_history.append(anomaly)
                    
            except Exception as e:
                logger.error(f"Error applying detection rule {rule_id}: {e}")
        
        # Keep only last 1000 anomalies
        if len(self.anomaly_history) > 1000:
            self.anomaly_history = self.anomaly_history[-1000:]
        
        return detected_anomalies
    
    async def _apply_detection_rule(self, rule_id: str, rule_config: Dict[str, Any]) -> Optional[Anomaly]:
        """Apply a specific detection rule"""
        
        metric_id = rule_config["metric_id"]
        rule_type = rule_config["rule_type"]
        
        # Get recent metrics
        metrics = self.metrics_collector.get_metrics(
            metric_id, 
            start_time=datetime.now() - timedelta(hours=1)
        )
        
        if not metrics:
            return None
        
        if rule_type == "threshold":
            return await self._apply_threshold_rule(rule_id, rule_config, metrics)
        elif rule_type == "statistical":
            return await self._apply_statistical_rule(rule_id, rule_config, metrics)
        else:
            logger.warning(f"Unknown rule type: {rule_type}")
            return None
    
    async def _apply_threshold_rule(self, rule_id: str, rule_config: Dict[str, Any], 
                                  metrics: List[Metric]) -> Optional[Anomaly]:
        """Apply threshold-based detection rule"""
        
        latest_metric = metrics[-1]
        threshold = rule_config["threshold"]
        operator = rule_config.get("operator", "greater_than")
        
        anomaly_detected = False
        
        if operator == "greater_than" and latest_metric.value > threshold:
            anomaly_detected = True
        elif operator == "less_than" and latest_metric.value < threshold:
            anomaly_detected = True
        elif operator == "equals" and abs(latest_metric.value - threshold) < 0.01:
            anomaly_detected = True
        
        if anomaly_detected:
            return Anomaly(
                anomaly_id=f"anomaly_{uuid.uuid4().hex[:8]}",
                anomaly_type=self._get_anomaly_type(rule_id),
                description=rule_config["description"],
                confidence=0.9,
                severity=AlertSeverity(rule_config["severity"]),
                detected_at=datetime.now(),
                metric_ids=[rule_config["metric_id"]],
                affected_components=[rule_config["metric_id"]],
                root_cause_analysis={
                    "rule_type": "threshold",
                    "threshold": threshold,
                    "actual_value": latest_metric.value,
                    "operator": operator
                },
                recommended_actions=self._get_recommended_actions(rule_id),
                metadata={"rule_id": rule_id}
            )
        
        return None
    
    async def _apply_statistical_rule(self, rule_id: str, rule_config: Dict[str, Any],
                                    metrics: List[Metric]) -> Optional[Anomaly]:
        """Apply statistical-based detection rule"""
        
        if len(metrics) < 10:  # Need sufficient data for statistical analysis
            return None
        
        values = [m.value for m in metrics]
        method = rule_config.get("method", "z_score")
        threshold = rule_config["threshold"]
        
        if method == "z_score":
            mean_val = statistics.mean(values[:-1])  # Exclude latest value
            std_val = statistics.stdev(values[:-1]) if len(values) > 2 else 0
            
            if std_val == 0:
                return None
            
            z_score = abs((values[-1] - mean_val) / std_val)
            
            if z_score > threshold:
                return Anomaly(
                    anomaly_id=f"anomaly_{uuid.uuid4().hex[:8]}",
                    anomaly_type=self._get_anomaly_type(rule_id),
                    description=rule_config["description"],
                    confidence=min(0.9, z_score / threshold),
                    severity=AlertSeverity(rule_config["severity"]),
                    detected_at=datetime.now(),
                    metric_ids=[rule_config["metric_id"]],
                    affected_components=[rule_config["metric_id"]],
                    root_cause_analysis={
                        "rule_type": "statistical",
                        "method": method,
                        "z_score": z_score,
                        "threshold": threshold,
                        "mean": mean_val,
                        "std_dev": std_val,
                        "actual_value": values[-1]
                    },
                    recommended_actions=self._get_recommended_actions(rule_id),
                    metadata={"rule_id": rule_id}
                )
        
        elif method == "percentage_change":
            if len(values) < 2:
                return None
            
            recent_avg = statistics.mean(values[-5:])  # Last 5 values
            baseline_avg = statistics.mean(values[:-5]) if len(values) > 5 else values[0]
            
            if baseline_avg == 0:
                return None
            
            percentage_change = abs((recent_avg - baseline_avg) / baseline_avg) * 100
            
            if percentage_change > threshold:
                return Anomaly(
                    anomaly_id=f"anomaly_{uuid.uuid4().hex[:8]}",
                    anomaly_type=self._get_anomaly_type(rule_id),
                    description=rule_config["description"],
                    confidence=min(0.9, percentage_change / threshold),
                    severity=AlertSeverity(rule_config["severity"]),
                    detected_at=datetime.now(),
                    metric_ids=[rule_config["metric_id"]],
                    affected_components=[rule_config["metric_id"]],
                    root_cause_analysis={
                        "rule_type": "statistical",
                        "method": method,
                        "percentage_change": percentage_change,
                        "threshold": threshold,
                        "recent_avg": recent_avg,
                        "baseline_avg": baseline_avg
                    },
                    recommended_actions=self._get_recommended_actions(rule_id),
                    metadata={"rule_id": rule_id}
                )
        
        return None
    
    def _get_anomaly_type(self, rule_id: str) -> AnomalyType:
        """Get anomaly type based on rule ID"""
        
        type_mapping = {
            "model_accuracy_drop": AnomalyType.ACCURACY_DROP,
            "response_latency_spike": AnomalyType.LATENCY_SPIKE,
            "error_rate_increase": AnomalyType.ERROR_RATE_INCREASE,
            "cost_spike": AnomalyType.COST_SPIKE,
            "security_events_spike": AnomalyType.SECURITY_THREAT
        }
        
        return type_mapping.get(rule_id, AnomalyType.PERFORMANCE_DEGRADATION)
    
    def _get_recommended_actions(self, rule_id: str) -> List[str]:
        """Get recommended actions for specific rule"""
        
        action_mapping = {
            "model_accuracy_drop": [
                "Review recent model changes",
                "Check data quality",
                "Consider model retraining",
                "Validate input data sources"
            ],
            "response_latency_spike": [
                "Check system resources",
                "Review recent deployments",
                "Scale up infrastructure",
                "Optimize database queries"
            ],
            "error_rate_increase": [
                "Check application logs",
                "Review recent code changes",
                "Validate external dependencies",
                "Monitor system health"
            ],
            "cost_spike": [
                "Review usage patterns",
                "Check for inefficient queries",
                "Optimize resource allocation",
                "Review pricing changes"
            ],
            "security_events_spike": [
                "Investigate security logs",
                "Check for suspicious activity",
                "Review access patterns",
                "Contact security team"
            ]
        }
        
        return action_mapping.get(rule_id, ["Investigate the issue", "Contact system administrator"])
    
    def get_anomaly_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of anomalies in the last N hours"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_anomalies = [
            a for a in self.anomaly_history 
            if a.detected_at > cutoff_time
        ]
        
        # Group by severity
        severity_counts = defaultdict(int)
        for anomaly in recent_anomalies:
            severity_counts[anomaly.severity.value] += 1
        
        # Group by type
        type_counts = defaultdict(int)
        for anomaly in recent_anomalies:
            type_counts[anomaly.anomaly_type.value] += 1
        
        return {
            "total_anomalies": len(recent_anomalies),
            "time_window_hours": hours,
            "severity_breakdown": dict(severity_counts),
            "type_breakdown": dict(type_counts),
            "latest_anomaly": recent_anomalies[-1].detected_at.isoformat() if recent_anomalies else None,
            "detection_enabled": self.detection_enabled
        }


class AlertManager:
    """Manages system alerts and notifications"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.notification_channels: Dict[str, Dict[str, Any]] = {}
        self.escalation_policies: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default notification channels
        self._initialize_notification_channels()
        
        logger.info("Alert Manager initialized")
    
    def _initialize_notification_channels(self):
        """Initialize default notification channels"""
        
        channels = {
            "email": {
                "type": "email",
                "enabled": True,
                "config": {
                    "smtp_server": "smtp.company.com",
                    "recipients": ["ops-team@company.com", "ai-team@company.com"]
                }
            },
            "slack": {
                "type": "slack",
                "enabled": True,
                "config": {
                    "webhook_url": "https://hooks.slack.com/services/...",
                    "channel": "#ai-alerts"
                }
            },
            "pagerduty": {
                "type": "pagerduty",
                "enabled": True,
                "config": {
                    "integration_key": "your-pagerduty-key",
                    "severity_mapping": {
                        "critical": "critical",
                        "high": "error",
                        "medium": "warning",
                        "low": "info"
                    }
                }
            }
        }
        
        for channel_id, config in channels.items():
            self.add_notification_channel(channel_id, config)
    
    def add_notification_channel(self, channel_id: str, config: Dict[str, Any]):
        """Add a notification channel"""
        
        self.notification_channels[channel_id] = config
        logger.info(f"Added notification channel: {channel_id}")
    
    async def create_alert_from_anomaly(self, anomaly: Anomaly) -> Alert:
        """Create an alert from detected anomaly"""
        
        alert = Alert(
            alert_id=f"alert_{uuid.uuid4().hex[:8]}",
            title=f"Anomaly Detected: {anomaly.anomaly_type.value.replace('_', ' ').title()}",
            description=anomaly.description,
            severity=anomaly.severity,
            status=AlertStatus.ACTIVE,
            metric_id=anomaly.metric_ids[0] if anomaly.metric_ids else None,
            threshold_value=None,
            actual_value=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            resolved_at=None,
            assigned_to=None,
            tags={"anomaly_id": anomaly.anomaly_id},
            metadata={
                "anomaly_type": anomaly.anomaly_type.value,
                "confidence": anomaly.confidence,
                "affected_components": anomaly.affected_components,
                "recommended_actions": anomaly.recommended_actions
            }
        )
        
        self.alerts[alert.alert_id] = alert
        
        # Send notifications
        await self._send_alert_notifications(alert)
        
        logger.info(f"Created alert: {alert.title} ({alert.alert_id})")
        
        return alert
    
    async def create_manual_alert(self, title: str, description: str, severity: AlertSeverity,
                                metric_id: str = None, tags: Dict[str, str] = None) -> Alert:
        """Create a manual alert"""
        
        alert = Alert(
            alert_id=f"alert_{uuid.uuid4().hex[:8]}",
            title=title,
            description=description,
            severity=severity,
            status=AlertStatus.ACTIVE,
            metric_id=metric_id,
            threshold_value=None,
            actual_value=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            resolved_at=None,
            assigned_to=None,
            tags=tags or {},
            metadata={"manual_alert": True}
        )
        
        self.alerts[alert.alert_id] = alert
        
        # Send notifications
        await self._send_alert_notifications(alert)
        
        logger.info(f"Created manual alert: {alert.title} ({alert.alert_id})")
        
        return alert
    
    async def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications through configured channels"""
        
        for channel_id, channel_config in self.notification_channels.items():
            if not channel_config.get("enabled", False):
                continue
            
            try:
                await self._send_notification(channel_id, channel_config, alert)
            except Exception as e:
                logger.error(f"Failed to send notification via {channel_id}: {e}")
    
    async def _send_notification(self, channel_id: str, channel_config: Dict[str, Any], alert: Alert):
        """Send notification through specific channel"""
        
        channel_type = channel_config["type"]
        
        if channel_type == "email":
            await self._send_email_notification(channel_config, alert)
        elif channel_type == "slack":
            await self._send_slack_notification(channel_config, alert)
        elif channel_type == "pagerduty":
            await self._send_pagerduty_notification(channel_config, alert)
        else:
            logger.warning(f"Unknown notification channel type: {channel_type}")
    
    async def _send_email_notification(self, config: Dict[str, Any], alert: Alert):
        """Send email notification (simulated)"""
        
        # In production, this would use actual SMTP
        logger.info(f"📧 Email notification sent for alert: {alert.title}")
        logger.info(f"   Recipients: {config['config']['recipients']}")
        logger.info(f"   Severity: {alert.severity.value}")
    
    async def _send_slack_notification(self, config: Dict[str, Any], alert: Alert):
        """Send Slack notification (simulated)"""
        
        # In production, this would use Slack API
        severity_emoji = {
            "critical": "🚨",
            "high": "⚠️",
            "medium": "⚡",
            "low": "ℹ️",
            "info": "📝"
        }
        
        emoji = severity_emoji.get(alert.severity.value, "📝")
        
        logger.info(f"💬 Slack notification sent for alert: {alert.title}")
        logger.info(f"   Channel: {config['config']['channel']}")
        logger.info(f"   Message: {emoji} {alert.severity.value.upper()}: {alert.title}")
    
    async def _send_pagerduty_notification(self, config: Dict[str, Any], alert: Alert):
        """Send PagerDuty notification (simulated)"""
        
        # In production, this would use PagerDuty API
        logger.info(f"📟 PagerDuty notification sent for alert: {alert.title}")
        logger.info(f"   Severity: {alert.severity.value}")
        logger.info(f"   Integration Key: {config['config']['integration_key'][:10]}...")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.assigned_to = acknowledged_by
        alert.updated_at = datetime.now()
        
        logger.info(f"Alert acknowledged: {alert.title} by {acknowledged_by}")
        
        return True
    
    def resolve_alert(self, alert_id: str, resolved_by: str, resolution_note: str = None) -> bool:
        """Resolve an alert"""
        
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        alert.updated_at = datetime.now()
        
        if resolution_note:
            alert.metadata["resolution_note"] = resolution_note
            alert.metadata["resolved_by"] = resolved_by
        
        logger.info(f"Alert resolved: {alert.title} by {resolved_by}")
        
        return True
    
    def get_active_alerts(self, severity: AlertSeverity = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity"""
        
        active_alerts = [
            alert for alert in self.alerts.values()
            if alert.status == AlertStatus.ACTIVE
        ]
        
        if severity:
            active_alerts = [
                alert for alert in active_alerts
                if alert.severity == severity
            ]
        
        # Sort by severity and creation time
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.HIGH: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 3,
            AlertSeverity.INFO: 4
        }
        
        active_alerts.sort(key=lambda a: (severity_order[a.severity], a.created_at))
        
        return active_alerts
    
    def get_alert_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert statistics for the last N hours"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self.alerts.values()
            if alert.created_at > cutoff_time
        ]
        
        # Count by severity
        severity_counts = defaultdict(int)
        for alert in recent_alerts:
            severity_counts[alert.severity.value] += 1
        
        # Count by status
        status_counts = defaultdict(int)
        for alert in recent_alerts:
            status_counts[alert.status.value] += 1
        
        # Calculate resolution time for resolved alerts
        resolved_alerts = [a for a in recent_alerts if a.status == AlertStatus.RESOLVED and a.resolved_at]
        avg_resolution_time = 0
        
        if resolved_alerts:
            resolution_times = [
                (a.resolved_at - a.created_at).total_seconds() / 60  # minutes
                for a in resolved_alerts
            ]
            avg_resolution_time = statistics.mean(resolution_times)
        
        return {
            "total_alerts": len(recent_alerts),
            "active_alerts": len([a for a in recent_alerts if a.status == AlertStatus.ACTIVE]),
            "time_window_hours": hours,
            "severity_breakdown": dict(severity_counts),
            "status_breakdown": dict(status_counts),
            "avg_resolution_time_minutes": round(avg_resolution_time, 2),
            "notification_channels": len(self.notification_channels)
        }


class DashboardManager:
    """Manages monitoring dashboards"""
    
    def __init__(self, metrics_collector: MetricsCollector, alert_manager: AlertManager):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.dashboards: Dict[str, Dashboard] = {}
        
        # Initialize default dashboards
        self._initialize_default_dashboards()
        
        logger.info("Dashboard Manager initialized")
    
    def _initialize_default_dashboards(self):
        """Initialize default monitoring dashboards"""
        
        # Executive Dashboard
        executive_dashboard = Dashboard(
            dashboard_id="executive_dashboard",
            name="Executive Dashboard",
            description="High-level business metrics and KPIs",
            dashboard_type="executive",
            widgets=[
                {
                    "widget_id": "business_kpis",
                    "title": "Business KPIs",
                    "type": "metrics_grid",
                    "metrics": ["request_volume", "user_satisfaction", "cost_per_request"],
                    "size": "large"
                },
                {
                    "widget_id": "system_health",
                    "title": "System Health",
                    "type": "status_indicator",
                    "metrics": ["model_accuracy", "error_rate", "compliance_score"],
                    "size": "medium"
                },
                {
                    "widget_id": "alerts_summary",
                    "title": "Active Alerts",
                    "type": "alert_summary",
                    "size": "medium"
                }
            ],
            refresh_interval=300,  # 5 minutes
            permissions=["executive", "manager"],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Technical Dashboard
        technical_dashboard = Dashboard(
            dashboard_id="technical_dashboard",
            name="Technical Operations Dashboard",
            description="Detailed technical metrics and system performance",
            dashboard_type="technical",
            widgets=[
                {
                    "widget_id": "performance_metrics",
                    "title": "Performance Metrics",
                    "type": "time_series",
                    "metrics": ["response_latency", "model_accuracy", "error_rate"],
                    "size": "large"
                },
                {
                    "widget_id": "resource_utilization",
                    "title": "Resource Utilization",
                    "type": "gauge_chart",
                    "metrics": ["cpu_usage", "memory_usage", "disk_usage"],
                    "size": "medium"
                },
                {
                    "widget_id": "request_patterns",
                    "title": "Request Patterns",
                    "type": "heatmap",
                    "metrics": ["request_volume"],
                    "size": "medium"
                },
                {
                    "widget_id": "error_analysis",
                    "title": "Error Analysis",
                    "type": "bar_chart",
                    "metrics": ["error_rate"],
                    "size": "medium"
                }
            ],
            refresh_interval=60,  # 1 minute
            permissions=["technical", "operator"],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Operational Dashboard
        operational_dashboard = Dashboard(
            dashboard_id="operational_dashboard",
            name="Operational Monitoring Dashboard",
            description="Real-time operational metrics and alerts",
            dashboard_type="operational",
            widgets=[
                {
                    "widget_id": "real_time_metrics",
                    "title": "Real-time Metrics",
                    "type": "live_metrics",
                    "metrics": ["request_volume", "response_latency", "error_rate"],
                    "size": "large"
                },
                {
                    "widget_id": "active_alerts",
                    "title": "Active Alerts",
                    "type": "alert_list",
                    "size": "medium"
                },
                {
                    "widget_id": "system_status",
                    "title": "System Status",
                    "type": "status_grid",
                    "size": "medium"
                },
                {
                    "widget_id": "recent_anomalies",
                    "title": "Recent Anomalies",
                    "type": "anomaly_timeline",
                    "size": "large"
                }
            ],
            refresh_interval=30,  # 30 seconds
            permissions=["operator", "technical"],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Register dashboards
        self.dashboards[executive_dashboard.dashboard_id] = executive_dashboard
        self.dashboards[technical_dashboard.dashboard_id] = technical_dashboard
        self.dashboards[operational_dashboard.dashboard_id] = operational_dashboard
    
    def create_dashboard(self, dashboard_config: Dict[str, Any]) -> Dashboard:
        """Create a new dashboard"""
        
        dashboard = Dashboard(
            dashboard_id=dashboard_config.get("dashboard_id", f"dashboard_{uuid.uuid4().hex[:8]}"),
            name=dashboard_config["name"],
            description=dashboard_config.get("description", ""),
            dashboard_type=dashboard_config.get("dashboard_type", "custom"),
            widgets=dashboard_config.get("widgets", []),
            refresh_interval=dashboard_config.get("refresh_interval", 300),
            permissions=dashboard_config.get("permissions", []),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.dashboards[dashboard.dashboard_id] = dashboard
        
        logger.info(f"Created dashboard: {dashboard.name} ({dashboard.dashboard_id})")
        
        return dashboard
    
    async def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Get data for a specific dashboard"""
        
        if dashboard_id not in self.dashboards:
            raise ValueError(f"Dashboard not found: {dashboard_id}")
        
        dashboard = self.dashboards[dashboard_id]
        dashboard_data = {
            "dashboard_info": {
                "id": dashboard.dashboard_id,
                "name": dashboard.name,
                "description": dashboard.description,
                "type": dashboard.dashboard_type,
                "refresh_interval": dashboard.refresh_interval,
                "last_updated": datetime.now().isoformat()
            },
            "widgets": []
        }
        
        # Get data for each widget
        for widget in dashboard.widgets:
            widget_data = await self._get_widget_data(widget)
            dashboard_data["widgets"].append(widget_data)
        
        return dashboard_data
    
    async def _get_widget_data(self, widget_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for a specific widget"""
        
        widget_type = widget_config["type"]
        widget_data = {
            "widget_id": widget_config["widget_id"],
            "title": widget_config["title"],
            "type": widget_type,
            "size": widget_config.get("size", "medium"),
            "data": {}
        }
        
        if widget_type == "metrics_grid":
            widget_data["data"] = await self._get_metrics_grid_data(widget_config)
        elif widget_type == "time_series":
            widget_data["data"] = await self._get_time_series_data(widget_config)
        elif widget_type == "status_indicator":
            widget_data["data"] = await self._get_status_indicator_data(widget_config)
        elif widget_type == "alert_summary":
            widget_data["data"] = await self._get_alert_summary_data()
        elif widget_type == "alert_list":
            widget_data["data"] = await self._get_alert_list_data()
        elif widget_type == "gauge_chart":
            widget_data["data"] = await self._get_gauge_chart_data(widget_config)
        elif widget_type == "live_metrics":
            widget_data["data"] = await self._get_live_metrics_data(widget_config)
        else:
            widget_data["data"] = {"message": f"Widget type '{widget_type}' not implemented"}
        
        return widget_data
    
    async def _get_metrics_grid_data(self, widget_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for metrics grid widget"""
        
        metrics = widget_config.get("metrics", [])
        grid_data = {}
        
        for metric_id in metrics:
            latest_metric = self.metrics_collector.get_latest_metric(metric_id)
            stats = self.metrics_collector.get_metric_statistics(metric_id, 60)
            
            if latest_metric and stats:
                grid_data[metric_id] = {
                    "name": latest_metric.metric_name,
                    "value": latest_metric.value,
                    "unit": latest_metric.unit,
                    "trend": stats.get("trend", "stable"),
                    "change_percent": self._calculate_change_percent(stats),
                    "timestamp": latest_metric.timestamp.isoformat()
                }
        
        return {"metrics": grid_data}
    
    async def _get_time_series_data(self, widget_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for time series widget"""
        
        metrics = widget_config.get("metrics", [])
        time_window = widget_config.get("time_window_hours", 24)
        
        start_time = datetime.now() - timedelta(hours=time_window)
        series_data = {}
        
        for metric_id in metrics:
            metric_points = self.metrics_collector.get_metrics(metric_id, start_time)
            
            if metric_points:
                series_data[metric_id] = {
                    "name": metric_points[0].metric_name,
                    "unit": metric_points[0].unit,
                    "data_points": [
                        {
                            "timestamp": point.timestamp.isoformat(),
                            "value": point.value
                        }
                        for point in metric_points
                    ]
                }
        
        return {"series": series_data, "time_window_hours": time_window}
    
    async def _get_status_indicator_data(self, widget_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for status indicator widget"""
        
        metrics = widget_config.get("metrics", [])
        status_data = {}
        
        # Define status thresholds
        status_thresholds = {
            "model_accuracy": {"good": 85, "warning": 80, "critical": 75},
            "error_rate": {"good": 2, "warning": 5, "critical": 10},
            "response_latency": {"good": 200, "warning": 500, "critical": 1000},
            "compliance_score": {"good": 95, "warning": 90, "critical": 85}
        }
        
        for metric_id in metrics:
            latest_metric = self.metrics_collector.get_latest_metric(metric_id)
            
            if latest_metric:
                thresholds = status_thresholds.get(metric_id, {"good": 80, "warning": 60, "critical": 40})
                status = self._determine_status(latest_metric.value, thresholds, metric_id)
                
                status_data[metric_id] = {
                    "name": latest_metric.metric_name,
                    "value": latest_metric.value,
                    "unit": latest_metric.unit,
                    "status": status,
                    "timestamp": latest_metric.timestamp.isoformat()
                }
        
        return {"statuses": status_data}
    
    async def _get_alert_summary_data(self) -> Dict[str, Any]:
        """Get data for alert summary widget"""
        
        active_alerts = self.alert_manager.get_active_alerts()
        alert_stats = self.alert_manager.get_alert_statistics(24)
        
        # Group by severity
        severity_counts = defaultdict(int)
        for alert in active_alerts:
            severity_counts[alert.severity.value] += 1
        
        return {
            "total_active": len(active_alerts),
            "severity_breakdown": dict(severity_counts),
            "statistics": alert_stats,
            "latest_alert": active_alerts[0].created_at.isoformat() if active_alerts else None
        }
    
    async def _get_alert_list_data(self) -> Dict[str, Any]:
        """Get data for alert list widget"""
        
        active_alerts = self.alert_manager.get_active_alerts()
        
        # Get top 10 most recent alerts
        recent_alerts = sorted(active_alerts, key=lambda a: a.created_at, reverse=True)[:10]
        
        alert_list = []
        for alert in recent_alerts:
            alert_list.append({
                "id": alert.alert_id,
                "title": alert.title,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "created_at": alert.created_at.isoformat(),
                "assigned_to": alert.assigned_to
            })
        
        return {"alerts": alert_list}
    
    async def _get_gauge_chart_data(self, widget_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for gauge chart widget"""
        
        metrics = widget_config.get("metrics", [])
        gauge_data = {}
        
        for metric_id in metrics:
            latest_metric = self.metrics_collector.get_latest_metric(metric_id)
            
            if latest_metric:
                # Define gauge ranges (these would be configurable in production)
                max_value = 100
                if "usage" in metric_id:
                    max_value = 100
                elif "latency" in metric_id:
                    max_value = 1000
                elif "accuracy" in metric_id:
                    max_value = 100
                
                gauge_data[metric_id] = {
                    "name": latest_metric.metric_name,
                    "value": latest_metric.value,
                    "max_value": max_value,
                    "unit": latest_metric.unit,
                    "percentage": min(100, (latest_metric.value / max_value) * 100)
                }
        
        return {"gauges": gauge_data}
    
    async def _get_live_metrics_data(self, widget_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for live metrics widget"""
        
        metrics = widget_config.get("metrics", [])
        live_data = {}
        
        for metric_id in metrics:
            latest_metric = self.metrics_collector.get_latest_metric(metric_id)
            stats = self.metrics_collector.get_metric_statistics(metric_id, 5)  # Last 5 minutes
            
            if latest_metric and stats:
                live_data[metric_id] = {
                    "name": latest_metric.metric_name,
                    "current_value": latest_metric.value,
                    "unit": latest_metric.unit,
                    "avg_5min": stats.get("mean", 0),
                    "trend": stats.get("trend", "stable"),
                    "last_updated": latest_metric.timestamp.isoformat()
                }
        
        return {"live_metrics": live_data}
    
    def _calculate_change_percent(self, stats: Dict[str, Any]) -> float:
        """Calculate percentage change for metrics"""
        
        if "mean" not in stats or stats["count"] < 2:
            return 0.0
        
        current = stats.get("latest", 0)
        average = stats.get("mean", 0)
        
        if average == 0:
            return 0.0
        
        return round(((current - average) / average) * 100, 2)
    
    def _determine_status(self, value: float, thresholds: Dict[str, float], metric_id: str) -> str:
        """Determine status based on metric value and thresholds"""
        
        # For metrics where lower is better (like error_rate, latency)
        lower_is_better = any(keyword in metric_id for keyword in ["error", "latency", "cost"])
        
        if lower_is_better:
            if value <= thresholds["good"]:
                return "good"
            elif value <= thresholds["warning"]:
                return "warning"
            else:
                return "critical"
        else:
            # For metrics where higher is better (like accuracy, satisfaction)
            if value >= thresholds["good"]:
                return "good"
            elif value >= thresholds["warning"]:
                return "warning"
            else:
                return "critical"
    
    def get_available_dashboards(self, user_permissions: List[str] = None) -> List[Dict[str, Any]]:
        """Get list of available dashboards for user"""
        
        available_dashboards = []
        
        for dashboard in self.dashboards.values():
            # Check permissions
            if user_permissions and dashboard.permissions:
                if not any(perm in user_permissions for perm in dashboard.permissions):
                    continue
            
            available_dashboards.append({
                "id": dashboard.dashboard_id,
                "name": dashboard.name,
                "description": dashboard.description,
                "type": dashboard.dashboard_type,
                "widget_count": len(dashboard.widgets),
                "refresh_interval": dashboard.refresh_interval
            })
        
        return available_dashboards
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get summary of dashboard system"""
        
        dashboard_types = defaultdict(int)
        total_widgets = 0
        
        for dashboard in self.dashboards.values():
            dashboard_types[dashboard.dashboard_type] += 1
            total_widgets += len(dashboard.widgets)
        
        return {
            "total_dashboards": len(self.dashboards),
            "total_widgets": total_widgets,
            "dashboard_types": dict(dashboard_types),
            "avg_widgets_per_dashboard": round(total_widgets / len(self.dashboards), 2) if self.dashboards else 0
        }


# Factory function for creating monitoring system
def create_comprehensive_monitoring_system() -> Tuple[MetricsCollector, AnomalyDetector, AlertManager, DashboardManager]:
    """Factory function to create complete monitoring system"""
    
    # Create core components
    metrics_collector = MetricsCollector()
    anomaly_detector = AnomalyDetector(metrics_collector)
    alert_manager = AlertManager()
    dashboard_manager = DashboardManager(metrics_collector, alert_manager)
    
    logger.info("Comprehensive Monitoring System created successfully")
    
    return metrics_collector, anomaly_detector, alert_manager, dashboard_manager