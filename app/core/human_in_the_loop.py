"""
Enhanced Human-in-the-Loop Validation System
BlackRock Aladdin-inspired expert collaboration and feedback integration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import json
import asyncio
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from abc import ABC, abstractmethod
import uuid
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from app.core.hybrid_ai_engine import ModelOutput, TaskCategory, ModelType

logger = logging.getLogger(__name__)


class ExpertType(Enum):
    """Types of human experts"""
    PORTFOLIO_MANAGER = "portfolio_manager"
    SENIOR_ANALYST = "senior_analyst"
    RISK_MANAGER = "risk_manager"
    COMPLIANCE_OFFICER = "compliance_officer"
    QUANTITATIVE_RESEARCHER = "quantitative_researcher"
    SECTOR_SPECIALIST = "sector_specialist"
    CHIEF_INVESTMENT_OFFICER = "chief_investment_officer"


class ReviewPriority(Enum):
    """Priority levels for expert review"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class ReviewStatus(Enum):
    """Status of expert review"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    CANCELLED = "cancelled"


class FeedbackType(Enum):
    """Types of expert feedback"""
    APPROVAL = "approval"
    REJECTION = "rejection"
    MODIFICATION = "modification"
    CLARIFICATION = "clarification"
    ESCALATION = "escalation"


@dataclass
class ExpertProfile:
    """Expert profile information"""
    expert_id: str
    name: str
    expert_type: ExpertType
    specializations: List[str]
    experience_years: int
    certifications: List[str]
    performance_rating: float
    availability_status: str
    contact_info: Dict[str, str]
    preferred_categories: List[TaskCategory]
    review_capacity: int  # Max concurrent reviews
    avg_review_time: float  # Hours
    created_at: datetime
    last_active: datetime


@dataclass
class ReviewRequest:
    """Request for expert review"""
    review_id: str
    model_output: ModelOutput
    expert_type: ExpertType
    priority: ReviewPriority
    context: Dict[str, Any]
    deadline: Optional[datetime]
    special_instructions: Optional[str]
    routing_reason: str
    created_at: datetime
    assigned_expert: Optional[str] = None
    status: ReviewStatus = ReviewStatus.PENDING
    estimated_completion: Optional[datetime] = None


@dataclass
class ExpertFeedback:
    """Expert feedback on model output"""
    feedback_id: str
    review_id: str
    expert_id: str
    feedback_type: FeedbackType
    overall_rating: int  # 1-10 scale
    agreement_level: float  # 0-1 scale
    confidence_in_feedback: float  # 0-1 scale
    detailed_comments: str
    specific_corrections: List[Dict[str, Any]]
    improvement_suggestions: List[str]
    risk_assessment: Optional[str]
    regulatory_concerns: List[str]
    alternative_recommendations: List[str]
    time_spent_minutes: int
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class CollaborationSession:
    """Interactive collaboration session between expert and AI"""
    session_id: str
    expert_id: str
    model_output: ModelOutput
    interaction_history: List[Dict[str, Any]]
    current_iteration: int
    session_status: str
    started_at: datetime
    last_interaction: datetime
    final_output: Optional[ModelOutput] = None
    session_notes: Optional[str] = None


class ExpertRouter:
    """Routes review requests to appropriate experts"""
    
    def __init__(self):
        self.experts: Dict[str, ExpertProfile] = {}
        self.routing_rules = self._initialize_routing_rules()
        self.workload_tracker = defaultdict(int)
        self.performance_history = defaultdict(list)
        
    def register_expert(self, expert: ExpertProfile):
        """Register a new expert in the system"""
        self.experts[expert.expert_id] = expert
        logger.info(f"Registered expert: {expert.name} ({expert.expert_type.value})")
    
    async def route_review_request(self, request: ReviewRequest) -> Optional[str]:
        """Route review request to best available expert"""
        
        # Find eligible experts
        eligible_experts = self._find_eligible_experts(request)
        
        if not eligible_experts:
            logger.warning(f"No eligible experts found for review {request.review_id}")
            return None
        
        # Score and rank experts
        scored_experts = await self._score_experts(eligible_experts, request)
        
        # Select best available expert
        selected_expert = self._select_best_expert(scored_experts, request)
        
        if selected_expert:
            # Assign the review
            request.assigned_expert = selected_expert.expert_id
            request.status = ReviewStatus.IN_PROGRESS
            request.estimated_completion = self._estimate_completion_time(selected_expert, request)
            
            # Update workload
            self.workload_tracker[selected_expert.expert_id] += 1
            
            logger.info(f"Assigned review {request.review_id} to expert {selected_expert.name}")
            return selected_expert.expert_id
        
        return None
    
    def _find_eligible_experts(self, request: ReviewRequest) -> List[ExpertProfile]:
        """Find experts eligible for the review request"""
        
        eligible = []
        
        for expert in self.experts.values():
            # Check expert type match
            if expert.expert_type != request.expert_type:
                continue
            
            # Check availability
            if expert.availability_status != "available":
                continue
            
            # Check workload capacity
            current_workload = self.workload_tracker[expert.expert_id]
            if current_workload >= expert.review_capacity:
                continue
            
            # Check specialization match
            if request.model_output.task_category in expert.preferred_categories:
                eligible.append(expert)
            elif not expert.preferred_categories:  # General expert
                eligible.append(expert)
        
        return eligible
    
    async def _score_experts(self, experts: List[ExpertProfile], 
                           request: ReviewRequest) -> List[Tuple[ExpertProfile, float]]:
        """Score experts based on suitability for the request"""
        
        scored_experts = []
        
        for expert in experts:
            score = 0.0
            
            # Performance rating (40% weight)
            score += expert.performance_rating * 0.4
            
            # Specialization match (30% weight)
            if request.model_output.task_category in expert.preferred_categories:
                score += 0.3
            
            # Experience (20% weight)
            experience_score = min(expert.experience_years / 20, 1.0)  # Cap at 20 years
            score += experience_score * 0.2
            
            # Workload factor (10% weight) - prefer less loaded experts
            workload_factor = 1 - (self.workload_tracker[expert.expert_id] / expert.review_capacity)
            score += workload_factor * 0.1
            
            # Priority boost for urgent requests
            if request.priority in [ReviewPriority.URGENT, ReviewPriority.CRITICAL]:
                if expert.expert_type in [ExpertType.SENIOR_ANALYST, ExpertType.CHIEF_INVESTMENT_OFFICER]:
                    score += 0.1
            
            scored_experts.append((expert, score))
        
        # Sort by score (descending)
        scored_experts.sort(key=lambda x: x[1], reverse=True)
        return scored_experts
    
    def _select_best_expert(self, scored_experts: List[Tuple[ExpertProfile, float]], 
                          request: ReviewRequest) -> Optional[ExpertProfile]:
        """Select the best expert from scored list"""
        
        if not scored_experts:
            return None
        
        # For critical requests, always select the highest scored expert
        if request.priority == ReviewPriority.CRITICAL:
            return scored_experts[0][0]
        
        # For other requests, consider availability and recent performance
        for expert, score in scored_experts:
            # Check if expert is currently active
            time_since_active = datetime.now() - expert.last_active
            if time_since_active > timedelta(hours=8):  # Not active in last 8 hours
                continue
            
            return expert
        
        # Fallback to highest scored expert
        return scored_experts[0][0]
    
    def _estimate_completion_time(self, expert: ExpertProfile, 
                                request: ReviewRequest) -> datetime:
        """Estimate when the expert will complete the review"""
        
        base_time = expert.avg_review_time
        
        # Adjust based on priority
        priority_multipliers = {
            ReviewPriority.LOW: 1.5,
            ReviewPriority.MEDIUM: 1.0,
            ReviewPriority.HIGH: 0.7,
            ReviewPriority.URGENT: 0.5,
            ReviewPriority.CRITICAL: 0.3
        }
        
        adjusted_time = base_time * priority_multipliers.get(request.priority, 1.0)
        
        # Add current workload delay
        workload_delay = self.workload_tracker[expert.expert_id] * 0.5  # 30 min per existing review
        
        total_hours = adjusted_time + workload_delay
        return datetime.now() + timedelta(hours=total_hours)
    
    def _initialize_routing_rules(self) -> Dict[str, Any]:
        """Initialize routing rules for different scenarios"""
        
        return {
            "high_confidence_threshold": 0.9,
            "auto_approve_threshold": 0.95,
            "mandatory_review_categories": [TaskCategory.RISK_ASSESSMENT],
            "escalation_rules": {
                "disagreement_threshold": 0.3,
                "low_rating_threshold": 4,
                "regulatory_concern_keywords": ["compliance", "regulation", "legal"]
            }
        }


class ReviewQueue:
    """Manages the queue of pending reviews"""
    
    def __init__(self):
        self.pending_reviews: Dict[str, ReviewRequest] = {}
        self.in_progress_reviews: Dict[str, ReviewRequest] = {}
        self.completed_reviews: Dict[str, ReviewRequest] = {}
        self.queue_metrics = {
            "total_submitted": 0,
            "total_completed": 0,
            "avg_wait_time": 0.0,
            "avg_review_time": 0.0
        }
    
    def add_review_request(self, request: ReviewRequest):
        """Add a new review request to the queue"""
        self.pending_reviews[request.review_id] = request
        self.queue_metrics["total_submitted"] += 1
        logger.info(f"Added review request {request.review_id} to queue")
    
    def start_review(self, review_id: str, expert_id: str):
        """Move review from pending to in-progress"""
        if review_id in self.pending_reviews:
            request = self.pending_reviews.pop(review_id)
            request.status = ReviewStatus.IN_PROGRESS
            request.assigned_expert = expert_id
            self.in_progress_reviews[review_id] = request
            logger.info(f"Started review {review_id} with expert {expert_id}")
    
    def complete_review(self, review_id: str, feedback: ExpertFeedback):
        """Move review from in-progress to completed"""
        if review_id in self.in_progress_reviews:
            request = self.in_progress_reviews.pop(review_id)
            request.status = ReviewStatus.COMPLETED
            self.completed_reviews[review_id] = request
            
            # Update metrics
            self.queue_metrics["total_completed"] += 1
            self._update_queue_metrics(request)
            
            logger.info(f"Completed review {review_id}")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return {
            "pending_count": len(self.pending_reviews),
            "in_progress_count": len(self.in_progress_reviews),
            "completed_count": len(self.completed_reviews),
            "metrics": self.queue_metrics,
            "priority_breakdown": self._get_priority_breakdown(),
            "expert_workload": self._get_expert_workload()
        }
    
    def _update_queue_metrics(self, request: ReviewRequest):
        """Update queue performance metrics"""
        
        # Calculate wait time (time from creation to assignment)
        if request.assigned_expert and request.status == ReviewStatus.COMPLETED:
            # This would be calculated from actual assignment time
            # For now, using estimated values
            wait_time = 2.0  # hours
            review_time = 1.5  # hours
            
            # Update running averages
            total_completed = self.queue_metrics["total_completed"]
            
            self.queue_metrics["avg_wait_time"] = (
                (self.queue_metrics["avg_wait_time"] * (total_completed - 1) + wait_time) / total_completed
            )
            
            self.queue_metrics["avg_review_time"] = (
                (self.queue_metrics["avg_review_time"] * (total_completed - 1) + review_time) / total_completed
            )
    
    def _get_priority_breakdown(self) -> Dict[str, int]:
        """Get breakdown of reviews by priority"""
        
        breakdown = defaultdict(int)
        
        for request in self.pending_reviews.values():
            breakdown[request.priority.value] += 1
        
        for request in self.in_progress_reviews.values():
            breakdown[request.priority.value] += 1
        
        return dict(breakdown)
    
    def _get_expert_workload(self) -> Dict[str, int]:
        """Get current workload per expert"""
        
        workload = defaultdict(int)
        
        for request in self.in_progress_reviews.values():
            if request.assigned_expert:
                workload[request.assigned_expert] += 1
        
        return dict(workload)


class FeedbackProcessor:
    """Processes expert feedback and integrates it into model improvement"""
    
    def __init__(self):
        self.feedback_history: List[ExpertFeedback] = []
        self.improvement_patterns = defaultdict(list)
        self.expert_reliability = defaultdict(float)
        
    async def process_feedback(self, feedback: ExpertFeedback, 
                             original_output: ModelOutput) -> Dict[str, Any]:
        """Process expert feedback and generate improvement insights"""
        
        # Store feedback
        self.feedback_history.append(feedback)
        
        # Analyze feedback patterns
        patterns = await self._analyze_feedback_patterns(feedback, original_output)
        
        # Update expert reliability scores
        self._update_expert_reliability(feedback)
        
        # Generate model improvement suggestions
        improvements = await self._generate_improvement_suggestions(feedback, patterns)
        
        # Create updated output if modifications suggested
        updated_output = None
        if feedback.feedback_type == FeedbackType.MODIFICATION:
            updated_output = await self._apply_expert_corrections(original_output, feedback)
        
        return {
            "feedback_processed": True,
            "patterns_identified": patterns,
            "improvement_suggestions": improvements,
            "updated_output": updated_output,
            "expert_reliability": self.expert_reliability[feedback.expert_id],
            "processing_timestamp": datetime.now()
        }
    
    async def _analyze_feedback_patterns(self, feedback: ExpertFeedback, 
                                       original_output: ModelOutput) -> Dict[str, Any]:
        """Analyze patterns in expert feedback"""
        
        patterns = {
            "common_corrections": [],
            "recurring_issues": [],
            "expert_preferences": [],
            "model_weaknesses": []
        }
        
        # Analyze corrections
        for correction in feedback.specific_corrections:
            correction_type = correction.get("type", "unknown")
            patterns["common_corrections"].append(correction_type)
        
        # Look for recurring issues across feedback history
        recent_feedback = [f for f in self.feedback_history[-10:] 
                          if f.review_id != feedback.review_id]
        
        for recent in recent_feedback:
            if recent.overall_rating < 6:  # Low rating
                patterns["recurring_issues"].append("low_quality_output")
            
            if recent.agreement_level < 0.5:  # Low agreement
                patterns["recurring_issues"].append("expert_disagreement")
        
        return patterns
    
    def _update_expert_reliability(self, feedback: ExpertFeedback):
        """Update expert reliability scores based on feedback quality"""
        
        expert_id = feedback.expert_id
        
        # Calculate reliability score based on multiple factors
        reliability_score = 0.0
        
        # Confidence in feedback (30% weight)
        reliability_score += feedback.confidence_in_feedback * 0.3
        
        # Consistency with other experts (40% weight)
        consistency_score = self._calculate_expert_consistency(feedback)
        reliability_score += consistency_score * 0.4
        
        # Feedback quality indicators (30% weight)
        quality_score = self._assess_feedback_quality(feedback)
        reliability_score += quality_score * 0.3
        
        # Update running average
        if expert_id in self.expert_reliability:
            current_score = self.expert_reliability[expert_id]
            self.expert_reliability[expert_id] = (current_score * 0.8 + reliability_score * 0.2)
        else:
            self.expert_reliability[expert_id] = reliability_score
    
    def _calculate_expert_consistency(self, feedback: ExpertFeedback) -> float:
        """Calculate how consistent this expert is with others"""
        
        # Find similar reviews from other experts
        similar_reviews = [
            f for f in self.feedback_history[-20:]  # Last 20 reviews
            if f.expert_id != feedback.expert_id and
            abs(f.overall_rating - feedback.overall_rating) <= 2
        ]
        
        if not similar_reviews:
            return 0.5  # Neutral score if no comparison data
        
        # Calculate average agreement
        agreements = []
        for similar in similar_reviews:
            agreement = 1 - abs(similar.agreement_level - feedback.agreement_level)
            agreements.append(agreement)
        
        return sum(agreements) / len(agreements)
    
    def _assess_feedback_quality(self, feedback: ExpertFeedback) -> float:
        """Assess the quality of the feedback provided"""
        
        quality_score = 0.0
        
        # Detailed comments (25% weight)
        if len(feedback.detailed_comments) > 100:
            quality_score += 0.25
        elif len(feedback.detailed_comments) > 50:
            quality_score += 0.15
        
        # Specific corrections (25% weight)
        if len(feedback.specific_corrections) > 0:
            quality_score += 0.25
        
        # Improvement suggestions (25% weight)
        if len(feedback.improvement_suggestions) > 0:
            quality_score += 0.25
        
        # Time spent (25% weight) - more time generally means more thorough review
        if feedback.time_spent_minutes > 30:
            quality_score += 0.25
        elif feedback.time_spent_minutes > 15:
            quality_score += 0.15
        
        return min(quality_score, 1.0)
    
    async def _generate_improvement_suggestions(self, feedback: ExpertFeedback, 
                                             patterns: Dict[str, Any]) -> List[str]:
        """Generate model improvement suggestions based on feedback"""
        
        suggestions = []
        
        # Based on rating
        if feedback.overall_rating < 6:
            suggestions.append("Consider retraining model with additional expert-validated data")
        
        # Based on agreement level
        if feedback.agreement_level < 0.5:
            suggestions.append("Review model output calibration and confidence scoring")
        
        # Based on specific corrections
        correction_types = [c.get("type") for c in feedback.specific_corrections]
        if "numerical_accuracy" in correction_types:
            suggestions.append("Improve numerical precision and validation")
        
        if "classification_error" in correction_types:
            suggestions.append("Enhance classification logic and decision boundaries")
        
        # Based on patterns
        if "low_quality_output" in patterns.get("recurring_issues", []):
            suggestions.append("Implement additional quality checks before output generation")
        
        return suggestions
    
    async def _apply_expert_corrections(self, original_output: ModelOutput, 
                                      feedback: ExpertFeedback) -> ModelOutput:
        """Apply expert corrections to create improved output"""
        
        # Create a copy of the original output
        corrected_result = original_output.result.copy()
        
        # Apply specific corrections
        for correction in feedback.specific_corrections:
            field = correction.get("field")
            corrected_value = correction.get("corrected_value")
            
            if field and corrected_value is not None:
                corrected_result[field] = corrected_value
        
        # Create new ModelOutput with corrections
        corrected_output = ModelOutput(
            result=corrected_result,
            confidence=original_output.confidence * 0.9,  # Slightly reduce confidence
            model_type=original_output.model_type,
            task_category=original_output.task_category,
            timestamp=datetime.now(),
            validation_score=original_output.validation_score,
            guardrail_passed=original_output.guardrail_passed,
            human_reviewed=True,
            metadata={
                **(original_output.metadata or {}),
                "expert_corrected": True,
                "expert_id": feedback.expert_id,
                "correction_timestamp": datetime.now().isoformat()
            }
        )
        
        return corrected_output


class HumanInTheLoopSystem:
    """Enhanced Human-in-the-Loop validation and collaboration system"""
    
    def __init__(self):
        self.expert_router = ExpertRouter()
        self.review_queue = ReviewQueue()
        self.feedback_processor = FeedbackProcessor()
        self.collaboration_sessions: Dict[str, CollaborationSession] = {}
        
        # System configuration
        self.config = {
            "auto_review_threshold": 0.7,
            "mandatory_review_categories": [TaskCategory.RISK_ASSESSMENT],
            "max_review_time_hours": 24,
            "escalation_enabled": True
        }
        
        # Performance tracking
        self.system_metrics = {
            "total_reviews": 0,
            "avg_expert_rating": 0.0,
            "avg_agreement_level": 0.0,
            "improvement_rate": 0.0,
            "expert_satisfaction": 0.0
        }
        
        logger.info("Enhanced Human-in-the-Loop System initialized")
    
    async def submit_for_review(self, output: ModelOutput, expert_type: ExpertType,
                              priority: ReviewPriority = ReviewPriority.MEDIUM,
                              context: Dict[str, Any] = None,
                              special_instructions: str = None) -> str:
        """Submit model output for expert review"""
        
        # Generate review ID
        review_id = f"review_{uuid.uuid4().hex[:8]}"
        
        # Determine routing reason
        routing_reason = self._determine_routing_reason(output, priority)
        
        # Create review request
        request = ReviewRequest(
            review_id=review_id,
            model_output=output,
            expert_type=expert_type,
            priority=priority,
            context=context or {},
            deadline=self._calculate_deadline(priority),
            special_instructions=special_instructions,
            routing_reason=routing_reason,
            created_at=datetime.now()
        )
        
        # Add to queue
        self.review_queue.add_review_request(request)
        
        # Route to expert
        assigned_expert = await self.expert_router.route_review_request(request)
        
        if assigned_expert:
            self.review_queue.start_review(review_id, assigned_expert)
            logger.info(f"Review {review_id} assigned to expert {assigned_expert}")
        else:
            logger.warning(f"Could not assign expert for review {review_id}")
        
        return review_id
    
    async def provide_expert_feedback(self, review_id: str, expert_id: str,
                                    feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process expert feedback on a review"""
        
        # Create feedback object
        feedback = ExpertFeedback(
            feedback_id=f"feedback_{uuid.uuid4().hex[:8]}",
            review_id=review_id,
            expert_id=expert_id,
            feedback_type=FeedbackType(feedback_data.get("feedback_type", "approval")),
            overall_rating=feedback_data.get("overall_rating", 5),
            agreement_level=feedback_data.get("agreement_level", 0.5),
            confidence_in_feedback=feedback_data.get("confidence_in_feedback", 0.8),
            detailed_comments=feedback_data.get("detailed_comments", ""),
            specific_corrections=feedback_data.get("specific_corrections", []),
            improvement_suggestions=feedback_data.get("improvement_suggestions", []),
            risk_assessment=feedback_data.get("risk_assessment"),
            regulatory_concerns=feedback_data.get("regulatory_concerns", []),
            alternative_recommendations=feedback_data.get("alternative_recommendations", []),
            time_spent_minutes=feedback_data.get("time_spent_minutes", 30),
            created_at=datetime.now(),
            metadata=feedback_data.get("metadata", {})
        )
        
        # Get original output
        if review_id in self.review_queue.in_progress_reviews:
            original_output = self.review_queue.in_progress_reviews[review_id].model_output
        elif review_id in self.review_queue.completed_reviews:
            original_output = self.review_queue.completed_reviews[review_id].model_output
        else:
            raise ValueError(f"Review {review_id} not found")
        
        # Process feedback
        processing_result = await self.feedback_processor.process_feedback(feedback, original_output)
        
        # Complete the review
        self.review_queue.complete_review(review_id, feedback)
        
        # Update system metrics
        self._update_system_metrics(feedback)
        
        # Check for escalation
        escalation_needed = self._check_escalation_needed(feedback)
        
        result = {
            "feedback_id": feedback.feedback_id,
            "review_completed": True,
            "processing_result": processing_result,
            "escalation_needed": escalation_needed,
            "timestamp": datetime.now()
        }
        
        return result 
   
    async def start_collaboration_session(self, expert_id: str, model_output: ModelOutput) -> str:
        """Start an interactive collaboration session"""
        
        session_id = f"collab_{uuid.uuid4().hex[:8]}"
        
        session = CollaborationSession(
            session_id=session_id,
            expert_id=expert_id,
            model_output=model_output,
            interaction_history=[],
            current_iteration=1,
            session_status="active",
            started_at=datetime.now(),
            last_interaction=datetime.now()
        )
        
        self.collaboration_sessions[session_id] = session
        
        logger.info(f"Started collaboration session {session_id} with expert {expert_id}")
        return session_id
    
    async def add_collaboration_interaction(self, session_id: str, 
                                          interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add an interaction to a collaboration session"""
        
        if session_id not in self.collaboration_sessions:
            raise ValueError(f"Collaboration session {session_id} not found")
        
        session = self.collaboration_sessions[session_id]
        
        # Add interaction to history
        interaction = {
            "iteration": session.current_iteration,
            "timestamp": datetime.now(),
            "type": interaction_data.get("type", "comment"),
            "content": interaction_data.get("content", ""),
            "expert_input": interaction_data.get("expert_input"),
            "ai_response": interaction_data.get("ai_response"),
            "metadata": interaction_data.get("metadata", {})
        }
        
        session.interaction_history.append(interaction)
        session.current_iteration += 1
        session.last_interaction = datetime.now()
        
        return {
            "interaction_added": True,
            "session_id": session_id,
            "current_iteration": session.current_iteration,
            "timestamp": datetime.now()
        }
    
    async def finalize_collaboration_session(self, session_id: str, 
                                           final_output: ModelOutput,
                                           session_notes: str = None) -> Dict[str, Any]:
        """Finalize a collaboration session"""
        
        if session_id not in self.collaboration_sessions:
            raise ValueError(f"Collaboration session {session_id} not found")
        
        session = self.collaboration_sessions[session_id]
        session.final_output = final_output
        session.session_notes = session_notes
        session.session_status = "completed"
        
        # Generate collaboration summary
        summary = {
            "session_id": session_id,
            "expert_id": session.expert_id,
            "total_interactions": len(session.interaction_history),
            "session_duration_minutes": (datetime.now() - session.started_at).total_seconds() / 60,
            "improvement_achieved": self._calculate_improvement(session.model_output, final_output),
            "collaboration_quality": self._assess_collaboration_quality(session),
            "final_output": final_output,
            "session_notes": session_notes
        }
        
        return summary
    
    def get_expert_dashboard_data(self, expert_id: str) -> Dict[str, Any]:
        """Get dashboard data for an expert"""
        
        # Get pending reviews for this expert
        pending_reviews = [
            request for request in self.review_queue.pending_reviews.values()
            if request.assigned_expert == expert_id
        ]
        
        # Get in-progress reviews
        in_progress_reviews = [
            request for request in self.review_queue.in_progress_reviews.values()
            if request.assigned_expert == expert_id
        ]
        
        # Get recent feedback history
        recent_feedback = [
            feedback for feedback in self.feedback_processor.feedback_history[-20:]
            if feedback.expert_id == expert_id
        ]
        
        # Calculate expert performance metrics
        performance_metrics = self._calculate_expert_performance(expert_id, recent_feedback)
        
        return {
            "expert_id": expert_id,
            "pending_reviews": len(pending_reviews),
            "in_progress_reviews": len(in_progress_reviews),
            "recent_feedback_count": len(recent_feedback),
            "performance_metrics": performance_metrics,
            "workload_status": self._get_expert_workload_status(expert_id),
            "upcoming_deadlines": self._get_upcoming_deadlines(expert_id),
            "collaboration_sessions": self._get_active_collaborations(expert_id),
            "timestamp": datetime.now()
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        
        queue_status = self.review_queue.get_queue_status()
        
        return {
            "queue_status": queue_status,
            "system_metrics": self.system_metrics,
            "expert_count": len(self.expert_router.experts),
            "active_collaborations": len([s for s in self.collaboration_sessions.values() 
                                        if s.session_status == "active"]),
            "configuration": self.config,
            "timestamp": datetime.now()
        }
    
    def _determine_routing_reason(self, output: ModelOutput, priority: ReviewPriority) -> str:
        """Determine why this output needs review"""
        
        reasons = []
        
        # Low confidence
        if output.confidence < self.config["auto_review_threshold"]:
            reasons.append("low_confidence")
        
        # Mandatory review category
        if output.task_category in self.config["mandatory_review_categories"]:
            reasons.append("mandatory_category")
        
        # High priority
        if priority in [ReviewPriority.URGENT, ReviewPriority.CRITICAL]:
            reasons.append("high_priority")
        
        # Guardrail issues
        if not output.guardrail_passed:
            reasons.append("guardrail_violation")
        
        return ", ".join(reasons) if reasons else "routine_review"
    
    def _calculate_deadline(self, priority: ReviewPriority) -> datetime:
        """Calculate review deadline based on priority"""
        
        hours_map = {
            ReviewPriority.LOW: 48,
            ReviewPriority.MEDIUM: 24,
            ReviewPriority.HIGH: 8,
            ReviewPriority.URGENT: 4,
            ReviewPriority.CRITICAL: 2
        }
        
        hours = hours_map.get(priority, 24)
        return datetime.now() + timedelta(hours=hours)
    
    def _update_system_metrics(self, feedback: ExpertFeedback):
        """Update system-wide metrics"""
        
        self.system_metrics["total_reviews"] += 1
        total_reviews = self.system_metrics["total_reviews"]
        
        # Update running averages
        current_rating = self.system_metrics["avg_expert_rating"]
        self.system_metrics["avg_expert_rating"] = (
            (current_rating * (total_reviews - 1) + feedback.overall_rating) / total_reviews
        )
        
        current_agreement = self.system_metrics["avg_agreement_level"]
        self.system_metrics["avg_agreement_level"] = (
            (current_agreement * (total_reviews - 1) + feedback.agreement_level) / total_reviews
        )
    
    def _check_escalation_needed(self, feedback: ExpertFeedback) -> bool:
        """Check if feedback indicates need for escalation"""
        
        # Low rating
        if feedback.overall_rating <= 3:
            return True
        
        # Low agreement
        if feedback.agreement_level < 0.3:
            return True
        
        # Regulatory concerns
        if feedback.regulatory_concerns:
            return True
        
        # Risk assessment indicates high risk
        if feedback.risk_assessment and "high" in feedback.risk_assessment.lower():
            return True
        
        return False
    
    def _calculate_improvement(self, original: ModelOutput, final: ModelOutput) -> float:
        """Calculate improvement achieved through collaboration"""
        
        # Simple improvement metric based on confidence and validation scores
        original_score = (original.confidence + original.validation_score) / 2
        final_score = (final.confidence + final.validation_score) / 2
        
        return final_score - original_score
    
    def _assess_collaboration_quality(self, session: CollaborationSession) -> float:
        """Assess the quality of a collaboration session"""
        
        quality_score = 0.0
        
        # Number of interactions (more is generally better up to a point)
        interaction_count = len(session.interaction_history)
        if 3 <= interaction_count <= 10:
            quality_score += 0.3
        elif interaction_count > 10:
            quality_score += 0.2
        
        # Session duration (not too short, not too long)
        duration_minutes = (session.last_interaction - session.started_at).total_seconds() / 60
        if 15 <= duration_minutes <= 120:
            quality_score += 0.3
        
        # Variety of interaction types
        interaction_types = set(interaction.get("type", "comment") for interaction in session.interaction_history)
        if len(interaction_types) > 1:
            quality_score += 0.2
        
        # Final output improvement
        if session.final_output and session.model_output:
            improvement = self._calculate_improvement(session.model_output, session.final_output)
            if improvement > 0:
                quality_score += 0.2
        
        return min(quality_score, 1.0)
    
    def _calculate_expert_performance(self, expert_id: str, 
                                    recent_feedback: List[ExpertFeedback]) -> Dict[str, Any]:
        """Calculate performance metrics for an expert"""
        
        if not recent_feedback:
            return {"insufficient_data": True}
        
        # Average rating given by this expert
        avg_rating = sum(f.overall_rating for f in recent_feedback) / len(recent_feedback)
        
        # Average agreement level
        avg_agreement = sum(f.agreement_level for f in recent_feedback) / len(recent_feedback)
        
        # Average confidence in feedback
        avg_confidence = sum(f.confidence_in_feedback for f in recent_feedback) / len(recent_feedback)
        
        # Average time spent
        avg_time = sum(f.time_spent_minutes for f in recent_feedback) / len(recent_feedback)
        
        # Reliability score
        reliability = self.feedback_processor.expert_reliability.get(expert_id, 0.5)
        
        return {
            "avg_rating_given": avg_rating,
            "avg_agreement_level": avg_agreement,
            "avg_confidence": avg_confidence,
            "avg_time_spent_minutes": avg_time,
            "reliability_score": reliability,
            "total_reviews": len(recent_feedback),
            "feedback_quality": self._assess_expert_feedback_quality(recent_feedback)
        }
    
    def _assess_expert_feedback_quality(self, feedback_list: List[ExpertFeedback]) -> float:
        """Assess overall quality of expert's feedback"""
        
        if not feedback_list:
            return 0.0
        
        quality_scores = []
        
        for feedback in feedback_list:
            score = self.feedback_processor._assess_feedback_quality(feedback)
            quality_scores.append(score)
        
        return sum(quality_scores) / len(quality_scores)
    
    def _get_expert_workload_status(self, expert_id: str) -> Dict[str, Any]:
        """Get current workload status for an expert"""
        
        if expert_id not in self.expert_router.experts:
            return {"error": "Expert not found"}
        
        expert = self.expert_router.experts[expert_id]
        current_workload = self.expert_router.workload_tracker[expert_id]
        
        return {
            "current_reviews": current_workload,
            "capacity": expert.review_capacity,
            "utilization": current_workload / expert.review_capacity if expert.review_capacity > 0 else 0,
            "status": "overloaded" if current_workload > expert.review_capacity else "available"
        }
    
    def _get_upcoming_deadlines(self, expert_id: str) -> List[Dict[str, Any]]:
        """Get upcoming deadlines for an expert"""
        
        deadlines = []
        
        for request in self.review_queue.in_progress_reviews.values():
            if request.assigned_expert == expert_id and request.deadline:
                deadlines.append({
                    "review_id": request.review_id,
                    "deadline": request.deadline.isoformat(),
                    "priority": request.priority.value,
                    "time_remaining_hours": (request.deadline - datetime.now()).total_seconds() / 3600
                })
        
        # Sort by deadline
        deadlines.sort(key=lambda x: x["deadline"])
        return deadlines
    
    def _get_active_collaborations(self, expert_id: str) -> List[Dict[str, Any]]:
        """Get active collaboration sessions for an expert"""
        
        active_sessions = []
        
        for session in self.collaboration_sessions.values():
            if session.expert_id == expert_id and session.session_status == "active":
                active_sessions.append({
                    "session_id": session.session_id,
                    "started_at": session.started_at.isoformat(),
                    "interactions": len(session.interaction_history),
                    "last_interaction": session.last_interaction.isoformat()
                })
        
        return active_sessions


# Factory function for creating human-in-the-loop system
def create_human_in_the_loop_system() -> HumanInTheLoopSystem:
    """Factory function to create human-in-the-loop system"""
    return HumanInTheLoopSystem()


# Utility functions for expert management
def create_expert_profile(name: str, expert_type: ExpertType, 
                         specializations: List[str],
                         experience_years: int = 5) -> ExpertProfile:
    """Create an expert profile"""
    
    return ExpertProfile(
        expert_id=f"expert_{uuid.uuid4().hex[:8]}",
        name=name,
        expert_type=expert_type,
        specializations=specializations,
        experience_years=experience_years,
        certifications=["CFA", "FRM"],  # Default certifications
        performance_rating=0.8,  # Default rating
        availability_status="available",
        contact_info={"email": f"{name.lower().replace(' ', '.')}@company.com"},
        preferred_categories=[TaskCategory.SENTIMENT_ANALYSIS, TaskCategory.RISK_ASSESSMENT],
        review_capacity=5,  # Max 5 concurrent reviews
        avg_review_time=2.0,  # 2 hours average
        created_at=datetime.now(),
        last_active=datetime.now()
    )


def create_sample_feedback(review_id: str, expert_id: str, 
                         rating: int = 7, agreement: float = 0.8) -> Dict[str, Any]:
    """Create sample expert feedback"""
    
    return {
        "feedback_type": "approval",
        "overall_rating": rating,
        "agreement_level": agreement,
        "confidence_in_feedback": 0.85,
        "detailed_comments": "The analysis is well-structured and provides valuable insights. Minor adjustments suggested for risk assessment.",
        "specific_corrections": [
            {
                "field": "risk_level",
                "original_value": "moderate",
                "corrected_value": "moderate-high",
                "type": "classification_adjustment",
                "reason": "Given current market volatility, risk should be slightly elevated"
            }
        ],
        "improvement_suggestions": [
            "Consider incorporating more recent market data",
            "Enhance risk factor correlation analysis"
        ],
        "risk_assessment": "Acceptable with minor adjustments",
        "regulatory_concerns": [],
        "alternative_recommendations": [
            "Consider diversification across sectors",
            "Monitor regulatory changes in target markets"
        ],
        "time_spent_minutes": 45,
        "metadata": {
            "review_method": "detailed_analysis",
            "tools_used": ["risk_calculator", "market_data_terminal"]
        }
    }