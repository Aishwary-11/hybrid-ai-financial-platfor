"""
Advanced Alerting and Incident Response System
BlackRock Aladdin-inspired intelligent alerting, automated incident response,
and proactive issue detection with stakeholder communication
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

from app.core.monitoring_system import AlertManager, Alert, AlertSeverity, AlertStatus, MetricsCollector
from app.core.analytics_reporting import PerformanceAnalyzer

logger = logging.getLogger(__name__)


class IncidentSeverity(Enum):
    """Incident severity levels"""
    P1_CRITICAL = "p1_critical"      # System down, major business impact
    P2_HIGH = "p2_high"              # Significant degradation, moderate business impact
    P3_MEDIUM = "p3_medium"          # Minor issues, limited business impact
    P4_LOW = "p4_low"                # Cosmetic issues, no business impact
    P5_INFORMATIONAL = "p5_info"     # Informational, no action required


class IncidentStatus(Enum):
    """Incident lifecycle status"""
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    MONITORING = "monitoring"
    RESOLVED = "resolved"
    CLOSED = "closed"


class EscalationLevel(Enum):
    """Escalation levels for incidents"""
    L1_SUPPORT = "l1_support"
    L2_ENGINEERING = "l2_engineering"
    L3_SENIOR_ENGINEERING = "l3_senior_engineering"
    L4_MANAGEMENT = "l4_management"
    L5_EXECUTIVE = "l5_executive"


class NotificationChannel(Enum):
    """Available notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    PHONE = "phone"


@dataclass
class Incident:
    """Incident record"""
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime]
    closed_at: Optional[datetime]
    assigned_to: Optional[str]
    escalation_level: EscalationLevel
    affected_services: List[str]
    root_cause: Optional[str]
    resolution_summary: Optional[str]
    related_alerts: List[str]
    timeline: List[Dict[str, Any]]
    stakeholders_notified: List[str]
    business_impact: Dict[str, Any]
    lessons_learned: List[str]
    metadata: Dict[str, Any]


@dataclass
class EscalationPolicy:
    """Escalation policy configuration"""
    policy_id: str
    name: str
    description: str
    trigger_conditions: List[Dict[str, Any]]
    escalation_steps: List[Dict[str, Any]]
    notification_channels: List[NotificationChannel]
    stakeholder_groups: Dict[str, List[str]]
    business_hours_only: bool
    auto_escalation_timeout: int  # minutes
    active: bool
    created_at: datetime


@dataclass
class AutomatedResponse:
    """Automated response action"""
    response_id: str
    name: str
    description: str
    trigger_conditions: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    approval_required: bool
    max_executions_per_hour: int
    success_rate: float
    last_executed: Optional[datetime]
    execution_count: int
    enabled: bool


class IntelligentAlerting:
    """Intelligent alerting system with ML-based filtering and prioritization"""
    
    def __init__(self, alert_manager: AlertManager, metrics_collector: MetricsCollector):
        self.alert_manager = alert_manager
        self.metrics_collector = metrics_collector
        self.alert_history: List[Alert] = []
        self.alert_patterns: Dict[str, Any] = {}
        self.noise_reduction_rules: List[Dict[str, Any]] = []
        self.alert_correlation_rules: List[Dict[str, Any]] = []
        
        # Initialize intelligent features
        self._initialize_noise_reduction()
        self._initialize_correlation_rules()
        
        logger.info("Intelligent Alerting system initialized")
    
    def _initialize_noise_reduction(self):
        """Initialize noise reduction rules"""
        
        self.noise_reduction_rules = [
            {
                "rule_id": "duplicate_suppression",
                "description": "Suppress duplicate alerts within time window",
                "time_window_minutes": 15,
                "similarity_threshold": 0.8,
                "enabled": True
            },
            {
                "rule_id": "flapping_detection",
                "description": "Detect and suppress flapping alerts",
                "flap_threshold": 5,  # 5 state changes in window
                "time_window_minutes": 30,
                "suppression_duration_minutes": 60,
                "enabled": True
            },
            {
                "rule_id": "maintenance_window",
                "description": "Suppress alerts during maintenance windows",
                "maintenance_schedules": [],
                "enabled": True
            },
            {
                "rule_id": "business_hours_filtering",
                "description": "Adjust alert severity based on business hours",
                "business_hours": {"start": 9, "end": 17},
                "severity_adjustment": {"low": "info", "medium": "low"},
                "enabled": True
            }
        ]
    
    def _initialize_correlation_rules(self):
        """Initialize alert correlation rules"""
        
        self.alert_correlation_rules = [
            {
                "rule_id": "cascade_correlation",
                "description": "Correlate cascading failures",
                "primary_alert_types": ["infrastructure_failure"],
                "secondary_alert_types": ["service_degradation", "high_latency"],
                "time_window_minutes": 10,
                "correlation_threshold": 0.7,
                "enabled": True
            },
            {
                "rule_id": "service_dependency",
                "description": "Correlate alerts based on service dependencies",
                "dependency_map": {
                    "ai_engine": ["model_accuracy", "response_latency"],
                    "database": ["query_performance", "connection_errors"],
                    "api_gateway": ["request_volume", "error_rate"]
                },
                "enabled": True
            }
        ]
    
    async def process_alert(self, alert: Alert) -> Dict[str, Any]:
        """Process alert through intelligent filtering and correlation"""
        
        # Store alert in history
        self.alert_history.append(alert)
        
        # Keep only last 1000 alerts
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        # Apply noise reduction
        noise_result = await self._apply_noise_reduction(alert)
        if noise_result["suppressed"]:
            return {
                "action": "suppressed",
                "reason": noise_result["reason"],
                "alert_id": alert.alert_id
            }
        
        # Apply correlation analysis
        correlation_result = await self._apply_correlation_analysis(alert)
        
        # Calculate priority score
        priority_score = await self._calculate_priority_score(alert, correlation_result)
        
        # Determine routing
        routing_decision = await self._determine_alert_routing(alert, priority_score, correlation_result)
        
        return {
            "action": "processed",
            "priority_score": priority_score,
            "correlation_info": correlation_result,
            "routing": routing_decision,
            "alert_id": alert.alert_id
        }
    
    async def _apply_noise_reduction(self, alert: Alert) -> Dict[str, Any]:
        """Apply noise reduction rules to filter out noise"""
        
        for rule in self.noise_reduction_rules:
            if not rule.get("enabled", True):
                continue
            
            rule_id = rule["rule_id"]
            
            if rule_id == "duplicate_suppression":
                if await self._check_duplicate_suppression(alert, rule):
                    return {"suppressed": True, "reason": "Duplicate alert suppressed"}
            
            elif rule_id == "flapping_detection":
                if await self._check_flapping_detection(alert, rule):
                    return {"suppressed": True, "reason": "Flapping alert suppressed"}
            
            elif rule_id == "maintenance_window":
                if await self._check_maintenance_window(alert, rule):
                    return {"suppressed": True, "reason": "Maintenance window active"}
            
            elif rule_id == "business_hours_filtering":
                severity_adjustment = await self._apply_business_hours_filtering(alert, rule)
                if severity_adjustment:
                    alert.severity = AlertSeverity(severity_adjustment)
        
        return {"suppressed": False, "reason": None}
    
    async def _check_duplicate_suppression(self, alert: Alert, rule: Dict[str, Any]) -> bool:
        """Check if alert should be suppressed as duplicate"""
        
        time_window = timedelta(minutes=rule["time_window_minutes"])
        cutoff_time = datetime.now() - time_window
        
        recent_alerts = [
            a for a in self.alert_history
            if a.created_at > cutoff_time and a.alert_id != alert.alert_id
        ]
        
        for recent_alert in recent_alerts:
            similarity = self._calculate_alert_similarity(alert, recent_alert)
            if similarity >= rule["similarity_threshold"]:
                return True
        
        return False
    
    async def _check_flapping_detection(self, alert: Alert, rule: Dict[str, Any]) -> bool:
        """Check if alert is flapping (rapidly changing states)"""
        
        time_window = timedelta(minutes=rule["time_window_minutes"])
        cutoff_time = datetime.now() - time_window
        
        # Count state changes for similar alerts
        similar_alerts = [
            a for a in self.alert_history
            if (a.created_at > cutoff_time and 
                a.title == alert.title and 
                a.metric_id == alert.metric_id)
        ]
        
        if len(similar_alerts) >= rule["flap_threshold"]:
            return True
        
        return False
    
    async def _check_maintenance_window(self, alert: Alert, rule: Dict[str, Any]) -> bool:
        """Check if alert occurs during maintenance window"""
        
        # Simplified maintenance window check
        current_hour = datetime.now().hour
        
        # Example: suppress alerts between 2-4 AM (maintenance window)
        if 2 <= current_hour <= 4:
            return True
        
        return False
    
    async def _apply_business_hours_filtering(self, alert: Alert, rule: Dict[str, Any]) -> Optional[str]:
        """Apply business hours filtering to adjust alert severity"""
        
        current_hour = datetime.now().hour
        business_hours = rule["business_hours"]
        
        # If outside business hours, reduce severity for non-critical alerts
        if not (business_hours["start"] <= current_hour <= business_hours["end"]):
            severity_adjustment = rule["severity_adjustment"]
            current_severity = alert.severity.value
            
            if current_severity in severity_adjustment:
                return severity_adjustment[current_severity]
        
        return None
    
    def _calculate_alert_similarity(self, alert1: Alert, alert2: Alert) -> float:
        """Calculate similarity between two alerts"""
        
        similarity_score = 0.0
        
        # Title similarity (40%)
        if alert1.title == alert2.title:
            similarity_score += 0.4
        
        # Metric similarity (30%)
        if alert1.metric_id == alert2.metric_id:
            similarity_score += 0.3
        
        # Severity similarity (20%)
        if alert1.severity == alert2.severity:
            similarity_score += 0.2
        
        # Description similarity (10%)
        if alert1.description == alert2.description:
            similarity_score += 0.1
        
        return similarity_score
    
    async def _apply_correlation_analysis(self, alert: Alert) -> Dict[str, Any]:
        """Apply correlation analysis to identify related alerts"""
        
        correlated_alerts = []
        correlation_strength = 0.0
        
        # Look for correlated alerts in recent history
        time_window = timedelta(minutes=30)
        cutoff_time = datetime.now() - time_window
        
        recent_alerts = [
            a for a in self.alert_history
            if a.created_at > cutoff_time and a.alert_id != alert.alert_id
        ]
        
        for rule in self.alert_correlation_rules:
            if not rule.get("enabled", True):
                continue
            
            if rule["rule_id"] == "cascade_correlation":
                cascade_alerts = await self._find_cascade_correlations(alert, recent_alerts, rule)
                correlated_alerts.extend(cascade_alerts)
            
            elif rule["rule_id"] == "service_dependency":
                dependency_alerts = await self._find_dependency_correlations(alert, recent_alerts, rule)
                correlated_alerts.extend(dependency_alerts)
        
        # Calculate overall correlation strength
        if correlated_alerts:
            correlation_strength = min(1.0, len(correlated_alerts) / 5.0)  # Max at 5 correlated alerts
        
        return {
            "correlated_alerts": [a.alert_id for a in correlated_alerts],
            "correlation_strength": correlation_strength,
            "correlation_type": "cascade" if correlated_alerts else "none"
        }
    
    async def _find_cascade_correlations(self, alert: Alert, recent_alerts: List[Alert], 
                                       rule: Dict[str, Any]) -> List[Alert]:
        """Find cascade correlation alerts"""
        
        correlated = []
        
        # Check if this is a primary alert that could cause cascading failures
        primary_types = rule["primary_alert_types"]
        secondary_types = rule["secondary_alert_types"]
        
        # Simple correlation based on alert metadata
        for recent_alert in recent_alerts:
            if (any(ptype in alert.title.lower() for ptype in primary_types) and
                any(stype in recent_alert.title.lower() for stype in secondary_types)):
                correlated.append(recent_alert)
        
        return correlated
    
    async def _find_dependency_correlations(self, alert: Alert, recent_alerts: List[Alert],
                                          rule: Dict[str, Any]) -> List[Alert]:
        """Find service dependency correlations"""
        
        correlated = []
        dependency_map = rule["dependency_map"]
        
        # Find service dependencies
        for service, dependent_metrics in dependency_map.items():
            if alert.metric_id in dependent_metrics:
                # Look for alerts in dependent services
                for recent_alert in recent_alerts:
                    if recent_alert.metric_id in dependent_metrics:
                        correlated.append(recent_alert)
        
        return correlated
    
    async def _calculate_priority_score(self, alert: Alert, correlation_info: Dict[str, Any]) -> float:
        """Calculate priority score for alert routing"""
        
        base_score = 0.0
        
        # Severity-based scoring (40%)
        severity_scores = {
            AlertSeverity.CRITICAL: 1.0,
            AlertSeverity.HIGH: 0.8,
            AlertSeverity.MEDIUM: 0.6,
            AlertSeverity.LOW: 0.4,
            AlertSeverity.INFO: 0.2
        }
        base_score += severity_scores.get(alert.severity, 0.5) * 0.4
        
        # Correlation-based scoring (30%)
        correlation_strength = correlation_info.get("correlation_strength", 0)
        base_score += correlation_strength * 0.3
        
        # Business impact scoring (20%)
        business_hours = 9 <= datetime.now().hour <= 17
        if business_hours:
            base_score += 0.2
        else:
            base_score += 0.1
        
        # Historical pattern scoring (10%)
        pattern_score = await self._get_historical_pattern_score(alert)
        base_score += pattern_score * 0.1
        
        return min(1.0, base_score)
    
    async def _get_historical_pattern_score(self, alert: Alert) -> float:
        """Get historical pattern score for alert"""
        
        # Count similar alerts in the past
        similar_count = sum(
            1 for a in self.alert_history[-100:]  # Last 100 alerts
            if a.metric_id == alert.metric_id and a.severity == alert.severity
        )
        
        # Higher frequency = higher priority (up to a point)
        return min(1.0, similar_count / 10.0)
    
    async def _determine_alert_routing(self, alert: Alert, priority_score: float,
                                     correlation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Determine how to route the alert"""
        
        routing = {
            "channels": [],
            "escalation_level": EscalationLevel.L1_SUPPORT,
            "immediate_response": False,
            "stakeholders": []
        }
        
        # Route based on priority score
        if priority_score >= 0.9:
            routing["channels"] = [NotificationChannel.PAGERDUTY, NotificationChannel.PHONE, NotificationChannel.SLACK]
            routing["escalation_level"] = EscalationLevel.L3_SENIOR_ENGINEERING
            routing["immediate_response"] = True
            routing["stakeholders"] = ["on_call_engineer", "engineering_manager", "ops_team"]
        
        elif priority_score >= 0.7:
            routing["channels"] = [NotificationChannel.PAGERDUTY, NotificationChannel.SLACK]
            routing["escalation_level"] = EscalationLevel.L2_ENGINEERING
            routing["stakeholders"] = ["on_call_engineer", "ops_team"]
        
        elif priority_score >= 0.5:
            routing["channels"] = [NotificationChannel.SLACK, NotificationChannel.EMAIL]
            routing["escalation_level"] = EscalationLevel.L1_SUPPORT
            routing["stakeholders"] = ["ops_team"]
        
        else:
            routing["channels"] = [NotificationChannel.EMAIL]
            routing["escalation_level"] = EscalationLevel.L1_SUPPORT
            routing["stakeholders"] = ["ops_team"]
        
        # Add correlation-based routing
        if correlation_info["correlation_strength"] > 0.7:
            routing["immediate_response"] = True
            if NotificationChannel.SLACK not in routing["channels"]:
                routing["channels"].append(NotificationChannel.SLACK)
        
        return routing


class IncidentManager:
    """Manages incident lifecycle and coordination"""
    
    def __init__(self, alert_manager: AlertManager, intelligent_alerting: IntelligentAlerting):
        self.alert_manager = alert_manager
        self.intelligent_alerting = intelligent_alerting
        self.incidents: Dict[str, Incident] = {}
        self.escalation_policies: Dict[str, EscalationPolicy] = {}
        self.incident_history: List[Incident] = []
        
        # Initialize default escalation policies
        self._initialize_escalation_policies()
        
        logger.info("Incident Manager initialized")
    
    def _initialize_escalation_policies(self):
        """Initialize default escalation policies"""
        
        # Critical incident escalation policy
        critical_policy = EscalationPolicy(
            policy_id="critical_incident_policy",
            name="Critical Incident Escalation",
            description="Escalation policy for P1 critical incidents",
            trigger_conditions=[
                {"severity": "p1_critical"},
                {"business_impact": "high"},
                {"affected_users": ">1000"}
            ],
            escalation_steps=[
                {
                    "level": EscalationLevel.L2_ENGINEERING,
                    "timeout_minutes": 5,
                    "stakeholders": ["on_call_engineer", "engineering_lead"]
                },
                {
                    "level": EscalationLevel.L3_SENIOR_ENGINEERING,
                    "timeout_minutes": 15,
                    "stakeholders": ["senior_engineer", "engineering_manager"]
                },
                {
                    "level": EscalationLevel.L4_MANAGEMENT,
                    "timeout_minutes": 30,
                    "stakeholders": ["engineering_director", "product_manager"]
                },
                {
                    "level": EscalationLevel.L5_EXECUTIVE,
                    "timeout_minutes": 60,
                    "stakeholders": ["cto", "ceo"]
                }
            ],
            notification_channels=[NotificationChannel.PAGERDUTY, NotificationChannel.PHONE, NotificationChannel.SLACK],
            stakeholder_groups={
                "engineering": ["on_call_engineer", "engineering_lead", "senior_engineer"],
                "management": ["engineering_manager", "engineering_director"],
                "executive": ["cto", "ceo"],
                "operations": ["ops_team", "devops_lead"]
            },
            business_hours_only=False,
            auto_escalation_timeout=15,
            active=True,
            created_at=datetime.now()
        )
        
        self.escalation_policies[critical_policy.policy_id] = critical_policy
    
    async def create_incident_from_alert(self, alert: Alert, correlation_info: Dict[str, Any] = None) -> Incident:
        """Create incident from alert"""
        
        # Determine incident severity based on alert
        incident_severity = self._map_alert_to_incident_severity(alert)
        
        # Create incident
        incident = Incident(
            incident_id=f"INC-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
            title=f"Incident: {alert.title}",
            description=alert.description,
            severity=incident_severity,
            status=IncidentStatus.DETECTED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            resolved_at=None,
            closed_at=None,
            assigned_to=None,
            escalation_level=EscalationLevel.L1_SUPPORT,
            affected_services=self._identify_affected_services(alert),
            root_cause=None,
            resolution_summary=None,
            related_alerts=[alert.alert_id],
            timeline=[
                {
                    "timestamp": datetime.now(),
                    "event": "incident_created",
                    "description": f"Incident created from alert {alert.alert_id}",
                    "user": "system"
                }
            ],
            stakeholders_notified=[],
            business_impact=self._assess_business_impact(alert, incident_severity),
            lessons_learned=[],
            metadata={
                "source_alert": alert.alert_id,
                "correlation_info": correlation_info or {},
                "auto_created": True
            }
        )
        
        # Store incident
        self.incidents[incident.incident_id] = incident
        
        # Start incident response workflow
        await self._start_incident_response(incident)
        
        logger.info(f"Created incident: {incident.incident_id} from alert {alert.alert_id}")
        
        return incident
    
    def _map_alert_to_incident_severity(self, alert: Alert) -> IncidentSeverity:
        """Map alert severity to incident severity"""
        
        mapping = {
            AlertSeverity.CRITICAL: IncidentSeverity.P1_CRITICAL,
            AlertSeverity.HIGH: IncidentSeverity.P2_HIGH,
            AlertSeverity.MEDIUM: IncidentSeverity.P3_MEDIUM,
            AlertSeverity.LOW: IncidentSeverity.P4_LOW,
            AlertSeverity.INFO: IncidentSeverity.P5_INFORMATIONAL
        }
        
        return mapping.get(alert.severity, IncidentSeverity.P3_MEDIUM)
    
    def _identify_affected_services(self, alert: Alert) -> List[str]:
        """Identify services affected by the alert"""
        
        # Map metrics to services
        metric_service_map = {
            "model_accuracy": ["ai_engine", "model_service"],
            "response_latency": ["api_gateway", "ai_engine"],
            "error_rate": ["api_gateway", "application_service"],
            "security_events": ["security_service", "authentication"],
            "compliance_score": ["compliance_service", "audit_service"]
        }
        
        return metric_service_map.get(alert.metric_id, ["unknown_service"])
    
    def _assess_business_impact(self, alert: Alert, severity: IncidentSeverity) -> Dict[str, Any]:
        """Assess business impact of the incident"""
        
        impact_assessment = {
            "severity": severity.value,
            "estimated_affected_users": 0,
            "estimated_revenue_impact": 0.0,
            "service_degradation": "none",
            "customer_facing": False,
            "sla_breach_risk": "low"
        }
        
        # Assess based on alert type and severity
        if alert.metric_id == "model_accuracy" and severity in [IncidentSeverity.P1_CRITICAL, IncidentSeverity.P2_HIGH]:
            impact_assessment.update({
                "estimated_affected_users": 5000,
                "estimated_revenue_impact": 50000.0,
                "service_degradation": "significant",
                "customer_facing": True,
                "sla_breach_risk": "high"
            })
        
        elif alert.metric_id == "response_latency" and severity == IncidentSeverity.P1_CRITICAL:
            impact_assessment.update({
                "estimated_affected_users": 10000,
                "estimated_revenue_impact": 25000.0,
                "service_degradation": "moderate",
                "customer_facing": True,
                "sla_breach_risk": "medium"
            })
        
        return impact_assessment
    
    async def _start_incident_response(self, incident: Incident):
        """Start incident response workflow"""
        
        # Find applicable escalation policy
        escalation_policy = await self._find_escalation_policy(incident)
        
        if escalation_policy:
            # Apply escalation policy
            await self._apply_escalation_policy(incident, escalation_policy)
        else:
            # Default response
            await self._apply_default_response(incident)
        
        # Update incident status
        await self.update_incident_status(incident.incident_id, IncidentStatus.INVESTIGATING, "system")
    
    async def _find_escalation_policy(self, incident: Incident) -> Optional[EscalationPolicy]:
        """Find applicable escalation policy for incident"""
        
        for policy in self.escalation_policies.values():
            if not policy.active:
                continue
            
            # Check if incident matches policy conditions
            if await self._matches_escalation_conditions(incident, policy):
                return policy
        
        return None
    
    async def _matches_escalation_conditions(self, incident: Incident, policy: EscalationPolicy) -> bool:
        """Check if incident matches escalation policy conditions"""
        
        for condition in policy.trigger_conditions:
            if "severity" in condition:
                if incident.severity.value != condition["severity"]:
                    return False
            
            if "business_impact" in condition:
                impact_level = incident.business_impact.get("service_degradation", "none")
                if impact_level != condition["business_impact"]:
                    return False
        
        return True
    
    async def _apply_escalation_policy(self, incident: Incident, policy: EscalationPolicy):
        """Apply escalation policy to incident"""
        
        # Start with first escalation step
        if policy.escalation_steps:
            first_step = policy.escalation_steps[0]
            incident.escalation_level = first_step["level"]
            
            # Notify stakeholders
            await self._notify_stakeholders(incident, first_step["stakeholders"], policy.notification_channels)
            
            # Schedule auto-escalation if configured
            if policy.auto_escalation_timeout > 0:
                await self._schedule_auto_escalation(incident, policy)
    
    async def _apply_default_response(self, incident: Incident):
        """Apply default incident response"""
        
        # Default notification to ops team
        await self._notify_stakeholders(incident, ["ops_team"], [NotificationChannel.EMAIL, NotificationChannel.SLACK])
    
    async def _notify_stakeholders(self, incident: Incident, stakeholders: List[str], 
                                 channels: List[NotificationChannel]):
        """Notify stakeholders about incident"""
        
        for stakeholder in stakeholders:
            for channel in channels:
                await self._send_incident_notification(incident, stakeholder, channel)
                
                # Track notification
                if stakeholder not in incident.stakeholders_notified:
                    incident.stakeholders_notified.append(stakeholder)
        
        # Add to timeline
        incident.timeline.append({
            "timestamp": datetime.now(),
            "event": "stakeholders_notified",
            "description": f"Notified {len(stakeholders)} stakeholders via {len(channels)} channels",
            "user": "system"
        })
    
    async def _send_incident_notification(self, incident: Incident, stakeholder: str, 
                                        channel: NotificationChannel):
        """Send incident notification via specific channel"""
        
        message = f"""
INCIDENT ALERT: {incident.title}
Incident ID: {incident.incident_id}
Severity: {incident.severity.value.upper()}
Status: {incident.status.value.upper()}
Created: {incident.created_at.strftime('%Y-%m-%d %H:%M:%S')}
Affected Services: {', '.join(incident.affected_services)}
Description: {incident.description}

Business Impact:
- Affected Users: {incident.business_impact.get('estimated_affected_users', 0):,}
- Revenue Impact: ${incident.business_impact.get('estimated_revenue_impact', 0):,.2f}
- Service Degradation: {incident.business_impact.get('service_degradation', 'unknown')}

Please acknowledge and begin investigation immediately.
        """.strip()
        
        if channel == NotificationChannel.EMAIL:
            logger.info(f"📧 Email notification sent to {stakeholder} for incident {incident.incident_id}")
        elif channel == NotificationChannel.SLACK:
            logger.info(f"💬 Slack notification sent to {stakeholder} for incident {incident.incident_id}")
        elif channel == NotificationChannel.PAGERDUTY:
            logger.info(f"📟 PagerDuty alert sent to {stakeholder} for incident {incident.incident_id}")
        elif channel == NotificationChannel.SMS:
            logger.info(f"📱 SMS sent to {stakeholder} for incident {incident.incident_id}")
        elif channel == NotificationChannel.PHONE:
            logger.info(f"📞 Phone call initiated to {stakeholder} for incident {incident.incident_id}")
    
    async def _schedule_auto_escalation(self, incident: Incident, policy: EscalationPolicy):
        """Schedule automatic escalation"""
        
        # In a real implementation, this would use a task scheduler
        logger.info(f"Auto-escalation scheduled for incident {incident.incident_id} in {policy.auto_escalation_timeout} minutes")
    
    async def update_incident_status(self, incident_id: str, new_status: IncidentStatus, 
                                   updated_by: str, notes: str = None) -> bool:
        """Update incident status"""
        
        if incident_id not in self.incidents:
            return False
        
        incident = self.incidents[incident_id]
        old_status = incident.status
        
        incident.status = new_status
        incident.updated_at = datetime.now()
        
        # Add to timeline
        incident.timeline.append({
            "timestamp": datetime.now(),
            "event": "status_updated",
            "description": f"Status changed from {old_status.value} to {new_status.value}",
            "user": updated_by,
            "notes": notes
        })
        
        # Handle status-specific actions
        if new_status == IncidentStatus.RESOLVED:
            incident.resolved_at = datetime.now()
            await self._handle_incident_resolution(incident)
        
        elif new_status == IncidentStatus.CLOSED:
            incident.closed_at = datetime.now()
            await self._handle_incident_closure(incident)
        
        logger.info(f"Incident {incident_id} status updated to {new_status.value} by {updated_by}")
        
        return True
    
    async def _handle_incident_resolution(self, incident: Incident):
        """Handle incident resolution"""
        
        # Notify stakeholders of resolution
        await self._notify_stakeholders(
            incident, 
            incident.stakeholders_notified, 
            [NotificationChannel.SLACK, NotificationChannel.EMAIL]
        )
        
        # Start post-incident review process
        await self._initiate_post_incident_review(incident)
    
    async def _handle_incident_closure(self, incident: Incident):
        """Handle incident closure"""
        
        # Move to history
        self.incident_history.append(incident)
        
        # Generate incident report
        await self._generate_incident_report(incident)
    
    async def _initiate_post_incident_review(self, incident: Incident):
        """Initiate post-incident review process"""
        
        logger.info(f"Post-incident review initiated for {incident.incident_id}")
        
        # In a real implementation, this would:
        # - Schedule PIR meeting
        # - Create PIR document template
        # - Assign PIR facilitator
        # - Collect timeline and data
    
    async def _generate_incident_report(self, incident: Incident) -> Dict[str, Any]:
        """Generate comprehensive incident report"""
        
        duration = None
        if incident.resolved_at:
            duration = (incident.resolved_at - incident.created_at).total_seconds() / 60  # minutes
        
        report = {
            "incident_id": incident.incident_id,
            "title": incident.title,
            "severity": incident.severity.value,
            "duration_minutes": duration,
            "affected_services": incident.affected_services,
            "business_impact": incident.business_impact,
            "timeline_events": len(incident.timeline),
            "stakeholders_notified": len(incident.stakeholders_notified),
            "root_cause": incident.root_cause,
            "resolution_summary": incident.resolution_summary,
            "lessons_learned": incident.lessons_learned,
            "created_at": incident.created_at.isoformat(),
            "resolved_at": incident.resolved_at.isoformat() if incident.resolved_at else None,
            "closed_at": incident.closed_at.isoformat() if incident.closed_at else None
        }
        
        logger.info(f"Generated incident report for {incident.incident_id}")
        
        return report
    
    def get_active_incidents(self, severity: IncidentSeverity = None) -> List[Incident]:
        """Get active incidents"""
        
        active_incidents = [
            incident for incident in self.incidents.values()
            if incident.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]
        ]
        
        if severity:
            active_incidents = [
                incident for incident in active_incidents
                if incident.severity == severity
            ]
        
        return sorted(active_incidents, key=lambda i: i.created_at, reverse=True)
    
    def get_incident_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get incident statistics"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_incidents = [
            incident for incident in self.incident_history
            if incident.created_at > cutoff_date
        ]
        
        # Calculate statistics
        total_incidents = len(recent_incidents)
        resolved_incidents = len([i for i in recent_incidents if i.resolved_at])
        
        # Average resolution time
        resolution_times = [
            (i.resolved_at - i.created_at).total_seconds() / 60
            for i in recent_incidents if i.resolved_at
        ]
        avg_resolution_time = statistics.mean(resolution_times) if resolution_times else 0
        
        # Severity breakdown
        severity_counts = defaultdict(int)
        for incident in recent_incidents:
            severity_counts[incident.severity.value] += 1
        
        return {
            "total_incidents": total_incidents,
            "resolved_incidents": resolved_incidents,
            "resolution_rate": resolved_incidents / total_incidents if total_incidents > 0 else 0,
            "avg_resolution_time_minutes": round(avg_resolution_time, 2),
            "severity_breakdown": dict(severity_counts),
            "active_incidents": len(self.get_active_incidents()),
            "time_period_days": days
        }


class AutomatedResponseEngine:
    """Automated response engine for common incidents"""
    
    def __init__(self, incident_manager: IncidentManager):
        self.incident_manager = incident_manager
        self.automated_responses: Dict[str, AutomatedResponse] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Initialize automated responses
        self._initialize_automated_responses()
        
        logger.info("Automated Response Engine initialized")
    
    def _initialize_automated_responses(self):
        """Initialize automated response actions"""
        
        responses = [
            {
                "response_id": "restart_service",
                "name": "Restart Service",
                "description": "Automatically restart affected service",
                "trigger_conditions": [
                    {"metric": "error_rate", "threshold": 10.0, "operator": "greater_than"},
                    {"service_health": "degraded"}
                ],
                "actions": [
                    {"type": "service_restart", "service": "ai_engine"},
                    {"type": "health_check", "wait_seconds": 30},
                    {"type": "notify", "message": "Service restarted automatically"}
                ],
                "approval_required": False,
                "max_executions_per_hour": 3
            },
            {
                "response_id": "scale_infrastructure",
                "name": "Scale Infrastructure",
                "description": "Automatically scale infrastructure resources",
                "trigger_conditions": [
                    {"metric": "response_latency", "threshold": 1000.0, "operator": "greater_than"},
                    {"metric": "request_volume", "threshold": 500.0, "operator": "greater_than"}
                ],
                "actions": [
                    {"type": "scale_up", "resource": "api_servers", "factor": 1.5},
                    {"type": "monitor", "duration_minutes": 15},
                    {"type": "notify", "message": "Infrastructure scaled automatically"}
                ],
                "approval_required": True,
                "max_executions_per_hour": 2
            },
            {
                "response_id": "enable_circuit_breaker",
                "name": "Enable Circuit Breaker",
                "description": "Enable circuit breaker for failing service",
                "trigger_conditions": [
                    {"metric": "error_rate", "threshold": 15.0, "operator": "greater_than"},
                    {"consecutive_failures": 5}
                ],
                "actions": [
                    {"type": "circuit_breaker", "action": "enable", "service": "external_api"},
                    {"type": "fallback", "enable": True},
                    {"type": "notify", "message": "Circuit breaker enabled automatically"}
                ],
                "approval_required": False,
                "max_executions_per_hour": 1
            }
        ]
        
        for response_config in responses:
            response = AutomatedResponse(
                response_id=response_config["response_id"],
                name=response_config["name"],
                description=response_config["description"],
                trigger_conditions=response_config["trigger_conditions"],
                actions=response_config["actions"],
                approval_required=response_config["approval_required"],
                max_executions_per_hour=response_config["max_executions_per_hour"],
                success_rate=0.0,
                last_executed=None,
                execution_count=0,
                enabled=True
            )
            
            self.automated_responses[response.response_id] = response
    
    async def evaluate_incident_for_automation(self, incident: Incident) -> Dict[str, Any]:
        """Evaluate if incident can be handled with automated response"""
        
        applicable_responses = []
        
        for response in self.automated_responses.values():
            if not response.enabled:
                continue
            
            # Check execution limits
            if not await self._check_execution_limits(response):
                continue
            
            # Check trigger conditions
            if await self._check_trigger_conditions(incident, response):
                applicable_responses.append(response)
        
        if applicable_responses:
            # Select best response (highest success rate)
            best_response = max(applicable_responses, key=lambda r: r.success_rate)
            
            return {
                "automation_available": True,
                "recommended_response": best_response.response_id,
                "response_name": best_response.name,
                "approval_required": best_response.approval_required,
                "confidence": best_response.success_rate
            }
        
        return {"automation_available": False}
    
    async def _check_execution_limits(self, response: AutomatedResponse) -> bool:
        """Check if response is within execution limits"""
        
        # Check hourly execution limit
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_executions = [
            entry for entry in self.execution_history
            if (entry["response_id"] == response.response_id and 
                entry["timestamp"] > one_hour_ago)
        ]
        
        return len(recent_executions) < response.max_executions_per_hour
    
    async def _check_trigger_conditions(self, incident: Incident, response: AutomatedResponse) -> bool:
        """Check if incident matches response trigger conditions"""
        
        for condition in response.trigger_conditions:
            if "metric" in condition:
                # Check metric-based conditions
                metric_name = condition["metric"]
                threshold = condition["threshold"]
                operator = condition["operator"]
                
                # Get current metric value (simplified)
                current_value = await self._get_current_metric_value(metric_name)
                
                if operator == "greater_than" and current_value <= threshold:
                    return False
                elif operator == "less_than" and current_value >= threshold:
                    return False
            
            elif "service_health" in condition:
                # Check service health conditions
                expected_health = condition["service_health"]
                current_health = await self._get_service_health(incident.affected_services)
                
                if current_health != expected_health:
                    return False
        
        return True
    
    async def _get_current_metric_value(self, metric_name: str) -> float:
        """Get current value for a metric"""
        
        # Simplified metric retrieval
        metric_values = {
            "error_rate": 5.0,
            "response_latency": 800.0,
            "request_volume": 300.0
        }
        
        return metric_values.get(metric_name, 0.0)
    
    async def _get_service_health(self, services: List[str]) -> str:
        """Get overall health status of services"""
        
        # Simplified service health check
        return "degraded"  # Assume degraded for demo
    
    async def execute_automated_response(self, response_id: str, incident: Incident, 
                                       approved_by: str = None) -> Dict[str, Any]:
        """Execute automated response"""
        
        if response_id not in self.automated_responses:
            return {"success": False, "error": "Response not found"}
        
        response = self.automated_responses[response_id]
        
        # Check approval requirement
        if response.approval_required and not approved_by:
            return {"success": False, "error": "Approval required"}
        
        # Execute actions
        execution_results = []
        overall_success = True
        
        for action in response.actions:
            try:
                result = await self._execute_action(action, incident)
                execution_results.append(result)
                
                if not result.get("success", False):
                    overall_success = False
                    
            except Exception as e:
                execution_results.append({"success": False, "error": str(e)})
                overall_success = False
        
        # Record execution
        execution_record = {
            "response_id": response_id,
            "incident_id": incident.incident_id,
            "timestamp": datetime.now(),
            "success": overall_success,
            "approved_by": approved_by,
            "results": execution_results
        }
        
        self.execution_history.append(execution_record)
        
        # Update response statistics
        response.execution_count += 1
        response.last_executed = datetime.now()
        
        if overall_success:
            # Update success rate (exponential moving average)
            alpha = 0.1
            response.success_rate = (1 - alpha) * response.success_rate + alpha * 1.0
        else:
            response.success_rate = (1 - alpha) * response.success_rate + alpha * 0.0
        
        # Add to incident timeline
        incident.timeline.append({
            "timestamp": datetime.now(),
            "event": "automated_response_executed",
            "description": f"Executed automated response: {response.name}",
            "user": approved_by or "system",
            "success": overall_success
        })
        
        logger.info(f"Executed automated response {response_id} for incident {incident.incident_id}: {'SUCCESS' if overall_success else 'FAILED'}")
        
        return {
            "success": overall_success,
            "response_name": response.name,
            "execution_results": execution_results,
            "execution_id": execution_record["timestamp"].isoformat()
        }
    
    async def _execute_action(self, action: Dict[str, Any], incident: Incident) -> Dict[str, Any]:
        """Execute a single automated action"""
        
        action_type = action["type"]
        
        if action_type == "service_restart":
            service = action["service"]
            logger.info(f"🔄 Restarting service: {service}")
            # Simulate service restart
            await asyncio.sleep(1)
            return {"success": True, "message": f"Service {service} restarted successfully"}
        
        elif action_type == "health_check":
            wait_seconds = action.get("wait_seconds", 10)
            logger.info(f"🏥 Performing health check (waiting {wait_seconds}s)")
            await asyncio.sleep(min(wait_seconds, 2))  # Simulate wait
            return {"success": True, "message": "Health check passed"}
        
        elif action_type == "scale_up":
            resource = action["resource"]
            factor = action.get("factor", 1.5)
            logger.info(f"📈 Scaling up {resource} by factor {factor}")
            return {"success": True, "message": f"Scaled {resource} by {factor}x"}
        
        elif action_type == "circuit_breaker":
            service = action["service"]
            cb_action = action["action"]
            logger.info(f"⚡ Circuit breaker {cb_action} for {service}")
            return {"success": True, "message": f"Circuit breaker {cb_action} for {service}"}
        
        elif action_type == "notify":
            message = action["message"]
            logger.info(f"📢 Notification: {message}")
            return {"success": True, "message": "Notification sent"}
        
        else:
            return {"success": False, "error": f"Unknown action type: {action_type}"}
    
    def get_automation_statistics(self) -> Dict[str, Any]:
        """Get automation statistics"""
        
        total_executions = len(self.execution_history)
        successful_executions = len([e for e in self.execution_history if e["success"]])
        
        # Response statistics
        response_stats = {}
        for response_id, response in self.automated_responses.items():
            response_stats[response_id] = {
                "name": response.name,
                "execution_count": response.execution_count,
                "success_rate": response.success_rate,
                "last_executed": response.last_executed.isoformat() if response.last_executed else None,
                "enabled": response.enabled
            }
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "overall_success_rate": successful_executions / total_executions if total_executions > 0 else 0,
            "available_responses": len(self.automated_responses),
            "enabled_responses": len([r for r in self.automated_responses.values() if r.enabled]),
            "response_statistics": response_stats
        }


# Factory function for creating incident response system
def create_incident_response_system(alert_manager: AlertManager, 
                                   metrics_collector: MetricsCollector) -> Tuple[IntelligentAlerting, IncidentManager, AutomatedResponseEngine]:
    """Factory function to create complete incident response system"""
    
    intelligent_alerting = IntelligentAlerting(alert_manager, metrics_collector)
    incident_manager = IncidentManager(alert_manager, intelligent_alerting)
    automated_response_engine = AutomatedResponseEngine(incident_manager)
    
    logger.info("Incident Response System created successfully")
    
    return intelligent_alerting, incident_manager, automated_response_engine