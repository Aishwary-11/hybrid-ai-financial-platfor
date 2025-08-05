"""
Advanced Orchestration and Routing System
BlackRock Aladdin-inspired intelligent workflow management and model coordination
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
# import networkx as nx  # Removed to avoid dependency
import warnings
warnings.filterwarnings('ignore')

from app.core.hybrid_ai_engine import ModelOutput, TaskCategory, ModelType, HybridAIEngine

logger = logging.getLogger(__name__)


class WorkflowType(Enum):
    """Types of orchestrated workflows"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    ITERATIVE = "iterative"
    ENSEMBLE = "ensemble"
    PIPELINE = "pipeline"
    ADAPTIVE = "adaptive"


class ExecutionStrategy(Enum):
    """Execution strategies for workflows"""
    FAIL_FAST = "fail_fast"
    BEST_EFFORT = "best_effort"
    RETRY_ON_FAILURE = "retry_on_failure"
    FALLBACK_CASCADE = "fallback_cascade"
    CONSENSUS_REQUIRED = "consensus_required"


class NodeStatus(Enum):
    """Status of workflow nodes"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class RoutingDecision(Enum):
    """Types of routing decisions"""
    DIRECT = "direct"
    CONDITIONAL = "conditional"
    LOAD_BALANCED = "load_balanced"
    PERFORMANCE_BASED = "performance_based"
    COST_OPTIMIZED = "cost_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"


@dataclass
class WorkflowNode:
    """Individual node in a workflow"""
    node_id: str
    node_type: str  # "model", "condition", "aggregator", "transformer"
    task_category: Optional[TaskCategory]
    model_name: Optional[str]
    parameters: Dict[str, Any]
    dependencies: List[str]  # Node IDs this node depends on
    conditions: List[Dict[str, Any]]  # Conditions for execution
    timeout_seconds: int
    retry_count: int
    fallback_nodes: List[str]
    output_transformers: List[str]
    metadata: Dict[str, Any]
    status: NodeStatus = NodeStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class WorkflowDefinition:
    """Definition of a complete workflow"""
    workflow_id: str
    name: str
    description: str
    workflow_type: WorkflowType
    execution_strategy: ExecutionStrategy
    nodes: List[WorkflowNode]
    routing_rules: List[Dict[str, Any]]
    global_timeout_seconds: int
    max_retries: int
    success_criteria: Dict[str, Any]
    failure_handling: Dict[str, Any]
    optimization_objectives: List[str]
    created_at: datetime
    version: str
    tags: List[str]


@dataclass
class RoutingRule:
    """Rule for intelligent routing decisions"""
    rule_id: str
    name: str
    conditions: List[Dict[str, Any]]
    routing_decision: RoutingDecision
    target_models: List[str]
    weight_factors: Dict[str, float]
    performance_thresholds: Dict[str, float]
    cost_constraints: Dict[str, float]
    quality_requirements: Dict[str, float]
    priority: int
    active: bool
    created_at: datetime


class WorkflowEngine:
    """Advanced workflow execution engine"""
    
    def __init__(self, ai_engine: HybridAIEngine):
        self.ai_engine = ai_engine
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.active_executions: Dict[str, Any] = {}
        self.execution_history: List[Any] = []
        self.performance_cache = defaultdict(list)
        
        # Execution statistics
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "avg_execution_time": 0.0,
            "avg_success_rate": 0.0
        }
        
        logger.info("Advanced Workflow Engine initialized")
    
    def register_workflow(self, workflow: WorkflowDefinition):
        """Register a new workflow definition"""
        self.workflows[workflow.workflow_id] = workflow
        logger.info(f"Registered workflow: {workflow.name} ({workflow.workflow_id})")
    
    async def execute_workflow(self, workflow_id: str, input_data: Dict[str, Any],
                             user_context: Dict[str, Any] = None,
                             execution_parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a registered workflow"""
        
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        
        try:
            # Simple execution for demo
            result = {
                "workflow_result": {
                    "analysis_complete": True,
                    "confidence": 0.85,
                    "recommendations": ["Buy", "Hold", "Monitor"]
                },
                "execution_summary": {
                    "total_nodes": len(workflow.nodes),
                    "executed_nodes": len(workflow.nodes),
                    "success_rate": 1.0
                }
            }
            
            # Update statistics
            self.execution_stats["total_executions"] += 1
            self.execution_stats["successful_executions"] += 1
            
            return {
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "status": "completed",
                "result": result,
                "execution_time": 2.5,
                "nodes_executed": len(workflow.nodes),
                "performance_metrics": {"avg_node_time": 0.5},
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            self.execution_stats["total_executions"] += 1
            self.execution_stats["failed_executions"] += 1
            
            return {
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e),
                "execution_time": 1.0,
                "nodes_executed": 0,
                "timestamp": datetime.now()
            }
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of a specific workflow"""
        
        if workflow_id not in self.workflows:
            return {"error": "Workflow not found"}
        
        workflow = self.workflows[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "workflow_type": workflow.workflow_type.value,
            "node_count": len(workflow.nodes),
            "active_executions": 0,
            "total_executions": 5,  # Mock data
            "success_rate": 0.9,
            "last_execution": datetime.now().isoformat()
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        
        return {
            "total_workflows": len(self.workflows),
            "active_executions": len(self.active_executions),
            "execution_stats": self.execution_stats,
            "performance_cache_size": sum(len(entries) for entries in self.performance_cache.values()),
            "workflow_types": list(set(wf.workflow_type.value for wf in self.workflows.values())),
            "timestamp": datetime.now()
        }


class IntelligentRouter:
    """Intelligent routing system for model selection and load balancing"""
    
    def __init__(self, ai_engine: HybridAIEngine):
        self.ai_engine = ai_engine
        self.routing_rules: List[RoutingRule] = []
        self.model_performance: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.load_balancer = LoadBalancer()
        self.cost_optimizer = CostOptimizer()
        
        logger.info("Intelligent Router initialized")
    
    def add_routing_rule(self, rule: RoutingRule):
        """Add a new routing rule"""
        self.routing_rules.append(rule)
        self.routing_rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"Added routing rule: {rule.name}")
    
    async def route_request(self, task_category: TaskCategory, input_data: Dict[str, Any],
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Route request to optimal model/workflow"""
        
        context = context or {}
        
        # Find applicable routing rules
        applicable_rules = self._find_applicable_rules(task_category, input_data, context)
        
        if applicable_rules:
            # Use highest priority rule
            selected_rule = applicable_rules[0]
            routing_decision = await self._execute_routing_rule(selected_rule, input_data, context)
        else:
            # Default routing
            routing_decision = await self._default_routing(task_category, input_data, context)
        
        return routing_decision
    
    def _find_applicable_rules(self, task_category: TaskCategory, input_data: Dict[str, Any],
                             context: Dict[str, Any]) -> List[RoutingRule]:
        """Find routing rules applicable to the request"""
        
        applicable = []
        
        for rule in self.routing_rules:
            if not rule.active:
                continue
            
            # Check conditions
            rule_applies = True
            for condition in rule.conditions:
                if not self._evaluate_routing_condition(condition, task_category, input_data, context):
                    rule_applies = False
                    break
            
            if rule_applies:
                applicable.append(rule)
        
        return applicable
    
    def _evaluate_routing_condition(self, condition: Dict[str, Any], task_category: TaskCategory,
                                  input_data: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate a routing condition"""
        
        condition_type = condition.get("type")
        
        if condition_type == "task_category":
            return task_category.value == condition.get("value")
        elif condition_type == "priority":
            request_priority = context.get("priority", 5)
            return request_priority >= condition.get("min_priority", 1)
        elif condition_type == "input_size":
            input_size = len(str(input_data))
            operator = condition.get("operator", "greater_than")
            threshold = condition.get("threshold", 1000)
            
            if operator == "greater_than":
                return input_size > threshold
            elif operator == "less_than":
                return input_size < threshold
        
        return True
    
    async def _execute_routing_rule(self, rule: RoutingRule, input_data: Dict[str, Any],
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific routing rule"""
        
        if rule.routing_decision == RoutingDecision.DIRECT:
            return await self._direct_routing(rule.target_models[0] if rule.target_models else None)
        elif rule.routing_decision == RoutingDecision.LOAD_BALANCED:
            return await self._load_balanced_routing(rule.target_models, context)
        elif rule.routing_decision == RoutingDecision.PERFORMANCE_BASED:
            return await self._performance_based_routing(rule.target_models, rule.performance_thresholds)
        elif rule.routing_decision == RoutingDecision.COST_OPTIMIZED:
            return await self._cost_optimized_routing(rule.target_models, rule.cost_constraints)
        elif rule.routing_decision == RoutingDecision.QUALITY_OPTIMIZED:
            return await self._quality_optimized_routing(rule.target_models, rule.quality_requirements)
        else:
            return await self._conditional_routing(rule, input_data, context)
    
    async def _default_routing(self, task_category: TaskCategory, input_data: Dict[str, Any],
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Default routing when no rules apply"""
        
        # Simple mapping to specialized models
        model_mapping = {
            TaskCategory.SENTIMENT_ANALYSIS: "sentiment_analysis",
            TaskCategory.RISK_ASSESSMENT: "risk_prediction",
            TaskCategory.EARNINGS_PREDICTION: "earnings_analysis",
            TaskCategory.THEMATIC_IDENTIFICATION: "thematic_identification"
        }
        
        target_model = model_mapping.get(task_category, "general")
        
        return {
            "routing_decision": "default",
            "target_model": target_model,
            "routing_reason": "No specific routing rules matched",
            "confidence": 0.7
        }
    
    async def _direct_routing(self, target_model: str) -> Dict[str, Any]:
        """Direct routing to specific model"""
        
        return {
            "routing_decision": "direct",
            "target_model": target_model,
            "routing_reason": "Direct routing rule applied",
            "confidence": 1.0
        }
    
    async def _load_balanced_routing(self, target_models: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Load balanced routing across multiple models"""
        
        if not target_models:
            return await self._default_routing(TaskCategory.MARKET_ANALYSIS, {}, context)
        
        # Get current load for each model
        model_loads = {}
        for model in target_models:
            model_loads[model] = self.load_balancer.get_current_load(model)
        
        # Select model with lowest load
        selected_model = min(model_loads.items(), key=lambda x: x[1])[0]
        
        return {
            "routing_decision": "load_balanced",
            "target_model": selected_model,
            "routing_reason": f"Selected based on load balancing (load: {model_loads[selected_model]})",
            "confidence": 0.8,
            "load_distribution": model_loads
        }
    
    async def _performance_based_routing(self, target_models: List[str], 
                                       performance_thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Performance-based routing"""
        
        if not target_models:
            return await self._default_routing(TaskCategory.MARKET_ANALYSIS, {}, {})
        
        # Score models based on performance metrics
        model_scores = {}
        for model in target_models:
            perf_data = self.model_performance.get(model, {})
            
            score = 0.0
            # Accuracy score (40%)
            accuracy = perf_data.get("accuracy", 0.5)
            score += accuracy * 0.4
            
            # Speed score (30%)
            avg_response_time = perf_data.get("avg_response_time", 5.0)
            speed_score = max(0, 1 - (avg_response_time / 10.0))  # Normalize to 0-1
            score += speed_score * 0.3
            
            # Reliability score (30%)
            reliability = perf_data.get("reliability", 0.5)
            score += reliability * 0.3
            
            model_scores[model] = score
        
        # Select highest scoring model
        selected_model = max(model_scores.items(), key=lambda x: x[1])[0]
        
        return {
            "routing_decision": "performance_based",
            "target_model": selected_model,
            "routing_reason": f"Selected based on performance score: {model_scores[selected_model]:.3f}",
            "confidence": model_scores[selected_model],
            "performance_scores": model_scores
        }
    
    async def _cost_optimized_routing(self, target_models: List[str], 
                                    cost_constraints: Dict[str, float]) -> Dict[str, Any]:
        """Cost-optimized routing"""
        
        if not target_models:
            return await self._default_routing(TaskCategory.MARKET_ANALYSIS, {}, {})
        
        # Calculate cost-effectiveness for each model
        model_cost_scores = {}
        max_cost = cost_constraints.get("max_cost_per_request", 1.0)
        
        for model in target_models:
            cost_per_request = self.cost_optimizer.get_model_cost(model)
            performance_score = self.model_performance.get(model, {}).get("accuracy", 0.5)
            
            if cost_per_request <= max_cost:
                # Cost-effectiveness = performance / cost
                cost_effectiveness = performance_score / (cost_per_request + 0.01)
                model_cost_scores[model] = cost_effectiveness
            else:
                model_cost_scores[model] = 0.0  # Exceeds budget
        
        if not any(score > 0 for score in model_cost_scores.values()):
            # All models exceed budget, select cheapest
            cheapest_model = min(target_models, 
                               key=lambda m: self.cost_optimizer.get_model_cost(m))
            return {
                "routing_decision": "cost_optimized",
                "target_model": cheapest_model,
                "routing_reason": "All models exceed budget, selected cheapest",
                "confidence": 0.3,
                "budget_exceeded": True
            }
        
        # Select most cost-effective model
        selected_model = max(model_cost_scores.items(), key=lambda x: x[1])[0]
        
        return {
            "routing_decision": "cost_optimized",
            "target_model": selected_model,
            "routing_reason": f"Selected for cost-effectiveness: {model_cost_scores[selected_model]:.3f}",
            "confidence": 0.8,
            "cost_effectiveness_scores": model_cost_scores
        }
    
    async def _quality_optimized_routing(self, target_models: List[str],
                                       quality_requirements: Dict[str, float]) -> Dict[str, Any]:
        """Quality-optimized routing"""
        
        if not target_models:
            return await self._default_routing(TaskCategory.MARKET_ANALYSIS, {}, {})
        
        min_accuracy = quality_requirements.get("min_accuracy", 0.8)
        min_confidence = quality_requirements.get("min_confidence", 0.7)
        
        # Filter models that meet quality requirements
        qualified_models = []
        for model in target_models:
            perf_data = self.model_performance.get(model, {})
            accuracy = perf_data.get("accuracy", 0.5)
            avg_confidence = perf_data.get("avg_confidence", 0.5)
            
            if accuracy >= min_accuracy and avg_confidence >= min_confidence:
                qualified_models.append((model, accuracy + avg_confidence))
        
        if not qualified_models:
            # No models meet requirements, select best available
            best_model = max(target_models, 
                           key=lambda m: self.model_performance.get(m, {}).get("accuracy", 0))
            return {
                "routing_decision": "quality_optimized",
                "target_model": best_model,
                "routing_reason": "No models meet quality requirements, selected best available",
                "confidence": 0.4,
                "requirements_met": False
            }
        
        # Select highest quality model
        selected_model = max(qualified_models, key=lambda x: x[1])[0]
        
        return {
            "routing_decision": "quality_optimized",
            "target_model": selected_model,
            "routing_reason": "Selected for highest quality score",
            "confidence": 0.9,
            "requirements_met": True,
            "qualified_models": len(qualified_models)
        }
    
    async def _conditional_routing(self, rule: RoutingRule, input_data: Dict[str, Any],
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Conditional routing based on complex logic"""
        
        # Default to first model if no conditions match
        return {
            "routing_decision": "conditional",
            "target_model": rule.target_models[0] if rule.target_models else "general",
            "routing_reason": "Default selection from conditional rule",
            "confidence": 0.6
        }
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        
        return {
            "total_rules": len(self.routing_rules),
            "active_rules": len([r for r in self.routing_rules if r.active]),
            "tracked_models": len(self.model_performance),
            "model_performance": dict(self.model_performance),
            "load_balancer_stats": self.load_balancer.get_stats(),
            "cost_optimizer_stats": self.cost_optimizer.get_stats()
        }


class LoadBalancer:
    """Load balancer for model requests"""
    
    def __init__(self):
        self.model_loads: Dict[str, int] = defaultdict(int)
        self.request_history: List[Tuple[str, datetime]] = []
        self.max_history_size = 1000
        
    def get_current_load(self, model_name: str) -> int:
        """Get current load for a model"""
        
        # Clean old requests (older than 1 hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.request_history = [
            (model, timestamp) for model, timestamp in self.request_history
            if timestamp > cutoff_time
        ]
        
        # Count recent requests for this model
        recent_requests = sum(1 for model, _ in self.request_history if model == model_name)
        return recent_requests
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        
        model_counts = defaultdict(int)
        for model, _ in self.request_history:
            model_counts[model] += 1
        
        return {
            "total_requests": len(self.request_history),
            "model_distribution": dict(model_counts),
            "history_window_hours": 1
        }


class CostOptimizer:
    """Cost optimizer for model selection"""
    
    def __init__(self):
        # Model cost per request (in arbitrary units)
        self.model_costs = {
            "gpt-4": 0.10,
            "gpt-3.5-turbo": 0.02,
            "claude-3": 0.08,
            "llama-2": 0.01,
            "sentiment_analysis": 0.005,
            "risk_prediction": 0.015,
            "earnings_analysis": 0.020,
            "general": 0.01
        }
        
        self.cost_history: List[Tuple[str, float, datetime]] = []
    
    def get_model_cost(self, model_name: str) -> float:
        """Get cost per request for a model"""
        return self.model_costs.get(model_name, 0.01)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cost optimizer statistics"""
        
        return {
            "model_costs": self.model_costs,
            "total_cost_records": len(self.cost_history)
        }


class WorkflowBuilder:
    """Builder for creating complex workflows"""
    
    def __init__(self):
        self.workflow_templates: Dict[str, Dict[str, Any]] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default workflow templates"""
        
        # Sequential Analysis Template
        self.workflow_templates["sequential_analysis"] = {
            "name": "Sequential Analysis Workflow",
            "description": "Sequential processing with dependency chain",
            "workflow_type": WorkflowType.SEQUENTIAL,
            "execution_strategy": ExecutionStrategy.FAIL_FAST,
            "nodes": [
                {
                    "node_id": "data_preprocessing",
                    "node_type": "transformer",
                    "parameters": {"type": "normalize"},
                    "dependencies": [],
                    "conditions": [],
                    "timeout_seconds": 30,
                    "retry_count": 2
                },
                {
                    "node_id": "sentiment_analysis",
                    "node_type": "model",
                    "task_category": TaskCategory.SENTIMENT_ANALYSIS,
                    "model_name": "sentiment_analysis",
                    "parameters": {},
                    "dependencies": ["data_preprocessing"],
                    "conditions": [],
                    "timeout_seconds": 60,
                    "retry_count": 1
                }
            ]
        }
        
        # Parallel Ensemble Template
        self.workflow_templates["parallel_ensemble"] = {
            "name": "Parallel Ensemble Workflow",
            "description": "Multiple models running in parallel with ensemble aggregation",
            "workflow_type": WorkflowType.ENSEMBLE,
            "execution_strategy": ExecutionStrategy.BEST_EFFORT,
            "nodes": [
                {
                    "node_id": "model_1",
                    "node_type": "model",
                    "task_category": TaskCategory.SENTIMENT_ANALYSIS,
                    "model_name": "sentiment_analysis",
                    "parameters": {},
                    "dependencies": [],
                    "conditions": [],
                    "timeout_seconds": 60,
                    "retry_count": 1
                },
                {
                    "node_id": "model_2",
                    "node_type": "model",
                    "task_category": TaskCategory.SENTIMENT_ANALYSIS,
                    "model_name": "gpt-3.5-turbo",
                    "parameters": {},
                    "dependencies": [],
                    "conditions": [],
                    "timeout_seconds": 60,
                    "retry_count": 1
                }
            ]
        }
        
        # Adaptive Workflow Template
        self.workflow_templates["adaptive_analysis"] = {
            "name": "Adaptive Analysis Workflow",
            "description": "Adaptive workflow that adjusts based on intermediate results",
            "workflow_type": WorkflowType.ADAPTIVE,
            "execution_strategy": ExecutionStrategy.BEST_EFFORT,
            "nodes": [
                {
                    "node_id": "initial_analysis",
                    "node_type": "model",
                    "task_category": TaskCategory.MARKET_ANALYSIS,
                    "model_name": "general",
                    "parameters": {},
                    "dependencies": [],
                    "conditions": [],
                    "timeout_seconds": 60,
                    "retry_count": 1
                }
            ]
        }
    
    def create_workflow_from_template(self, template_name: str, 
                                    customizations: Dict[str, Any] = None) -> WorkflowDefinition:
        """Create a workflow from a template"""
        
        if template_name not in self.workflow_templates:
            raise ValueError(f"Template {template_name} not found")
        
        template = self.workflow_templates[template_name].copy()
        
        # Apply customizations
        if customizations:
            template.update(customizations)
        
        # Create workflow nodes
        nodes = []
        for node_data in template["nodes"]:
            node = WorkflowNode(
                node_id=node_data["node_id"],
                node_type=node_data["node_type"],
                task_category=node_data.get("task_category"),
                model_name=node_data.get("model_name"),
                parameters=node_data.get("parameters", {}),
                dependencies=node_data.get("dependencies", []),
                conditions=node_data.get("conditions", []),
                timeout_seconds=node_data.get("timeout_seconds", 60),
                retry_count=node_data.get("retry_count", 1),
                fallback_nodes=node_data.get("fallback_nodes", []),
                output_transformers=node_data.get("output_transformers", []),
                metadata=node_data.get("metadata", {})
            )
            nodes.append(node)
        
        # Create workflow definition
        workflow = WorkflowDefinition(
            workflow_id=f"wf_{uuid.uuid4().hex[:8]}",
            name=template["name"],
            description=template["description"],
            workflow_type=template["workflow_type"],
            execution_strategy=template["execution_strategy"],
            nodes=nodes,
            routing_rules=template.get("routing_rules", []),
            global_timeout_seconds=template.get("global_timeout_seconds", 300),
            max_retries=template.get("max_retries", 3),
            success_criteria=template.get("success_criteria", {}),
            failure_handling=template.get("failure_handling", {}),
            optimization_objectives=template.get("optimization_objectives", []),
            created_at=datetime.now(),
            version="1.0",
            tags=template.get("tags", [])
        )
        
        return workflow
    
    def get_available_templates(self) -> List[str]:
        """Get list of available workflow templates"""
        return list(self.workflow_templates.keys())
    
    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """Get information about a specific template"""
        
        if template_name not in self.workflow_templates:
            raise ValueError(f"Template {template_name} not found")
        
        template = self.workflow_templates[template_name]
        
        return {
            "name": template["name"],
            "description": template["description"],
            "workflow_type": template["workflow_type"].value if hasattr(template["workflow_type"], 'value') else template["workflow_type"],
            "execution_strategy": template["execution_strategy"].value if hasattr(template["execution_strategy"], 'value') else template["execution_strategy"],
            "node_count": len(template["nodes"]),
            "node_types": list(set(node["node_type"] for node in template["nodes"]))
        }


# Factory function for creating orchestration system
def create_advanced_orchestration_system(ai_engine) -> Tuple[WorkflowEngine, IntelligentRouter, WorkflowBuilder]:
    """Factory function to create complete orchestration system"""
    
    workflow_engine = WorkflowEngine(ai_engine)
    intelligent_router = IntelligentRouter(ai_engine)
    workflow_builder = WorkflowBuilder()
    
    # Add some default routing rules
    default_rules = [
        RoutingRule(
            rule_id="high_priority_direct",
            name="High Priority Direct Routing",
            conditions=[
                {"type": "priority", "min_priority": 8}
            ],
            routing_decision=RoutingDecision.PERFORMANCE_BASED,
            target_models=["gpt-4", "claude-3"],
            weight_factors={"performance": 0.8, "cost": 0.2},
            performance_thresholds={"min_accuracy": 0.9},
            cost_constraints={"max_cost_per_request": 0.20},
            quality_requirements={"min_confidence": 0.8},
            priority=10,
            active=True,
            created_at=datetime.now()
        ),
        RoutingRule(
            rule_id="cost_conscious_routing",
            name="Cost Conscious Routing",
            conditions=[
                {"type": "priority", "min_priority": 1, "max_priority": 5}
            ],
            routing_decision=RoutingDecision.COST_OPTIMIZED,
            target_models=["gpt-3.5-turbo", "llama-2"],
            weight_factors={"cost": 0.7, "performance": 0.3},
            performance_thresholds={"min_accuracy": 0.7},
            cost_constraints={"max_cost_per_request": 0.05},
            quality_requirements={"min_confidence": 0.6},
            priority=5,
            active=True,
            created_at=datetime.now()
        )
    ]
    
    for rule in default_rules:
        intelligent_router.add_routing_rule(rule)
    
    logger.info("Advanced Orchestration System created successfully")
    
    return workflow_engine, intelligent_router, workflow_builder