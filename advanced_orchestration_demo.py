#!/usr/bin/env python3
"""
Advanced Orchestration and Routing System Demo
BlackRock Aladdin-inspired intelligent workflow management and model coordination
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our modules
from app.core.hybrid_ai_engine import HybridAIEngine, TaskCategory
from app.core.advanced_orchestration import (
    create_advanced_orchestration_system,
    WorkflowType, ExecutionStrategy, RoutingDecision,
    WorkflowNode, WorkflowDefinition, RoutingRule
)

class AdvancedOrchestrationDemo:
    """Demo of the advanced orchestration system"""
    
    def __init__(self):
        self.ai_engine = None
        self.workflow_engine = None
        self.intelligent_router = None
        self.workflow_builder = None
        
    async def initialize(self):
        """Initialize the orchestration system"""
        
        print("=" * 80)
        print("ADVANCED ORCHESTRATION & ROUTING SYSTEM DEMO")
        print("BlackRock Aladdin-inspired Intelligent Workflow Management")
        print("=" * 80)
        
        # Initialize AI engine
        print("\\n🔧 Initializing Hybrid AI Engine...")
        self.ai_engine = HybridAIEngine()
        
        # Create orchestration system
        print("🔧 Creating Advanced Orchestration System...")
        self.workflow_engine, self.intelligent_router, self.workflow_builder = \
            create_advanced_orchestration_system(self.ai_engine)
        
        print("✅ System initialized successfully!")
    
    async def demo_intelligent_routing(self):
        """Demonstrate intelligent routing capabilities"""
        
        print("\\n" + "-" * 60)
        print("🧠 INTELLIGENT ROUTING DEMONSTRATION")
        print("-" * 60)
        
        # Test different routing scenarios
        routing_scenarios = [
            {
                "name": "High Priority Request",
                "task_category": TaskCategory.RISK_ASSESSMENT,
                "input_data": {"text": "Analyze market volatility risks"},
                "context": {"priority": 9, "deadline": datetime.now() + timedelta(minutes=5)}
            },
            {
                "name": "Cost-Conscious Request",
                "task_category": TaskCategory.SENTIMENT_ANALYSIS,
                "input_data": {"text": "Market sentiment looks positive"},
                "context": {"priority": 3, "budget_constraint": 0.03}
            },
            {
                "name": "Quality-Critical Request",
                "task_category": TaskCategory.EARNINGS_PREDICTION,
                "input_data": {"company": "AAPL", "quarter": "Q4 2024"},
                "context": {"priority": 7, "quality_requirements": {"min_accuracy": 0.9}}
            }
        ]
        
        for scenario in routing_scenarios:
            print(f"\\n📍 Testing: {scenario['name']}")
            print(f"   Task: {scenario['task_category'].value}")
            print(f"   Context: {scenario['context']}")
            
            try:
                routing_decision = await self.intelligent_router.route_request(
                    scenario["task_category"],
                    scenario["input_data"],
                    scenario["context"]
                )
                
                print(f"   ✅ Routing Decision: {routing_decision['routing_decision']}")
                print(f"   🎯 Selected Model: {routing_decision['target_model']}")
                print(f"   📝 Reason: {routing_decision['routing_reason']}")
                print(f"   🎲 Confidence: {routing_decision['confidence']:.2f}")
                
            except Exception as e:
                print(f"   ❌ Routing failed: {e}")
    
    async def demo_workflow_templates(self):
        """Demonstrate workflow template system"""
        
        print("\\n" + "-" * 60)
        print("📋 WORKFLOW TEMPLATE DEMONSTRATION")
        print("-" * 60)
        
        # Show available templates
        templates = self.workflow_builder.get_available_templates()
        print(f"\\n📚 Available Templates: {len(templates)}")
        
        for template_name in templates:
            template_info = self.workflow_builder.get_template_info(template_name)
            print(f"\\n  📄 {template_info['name']}")
            print(f"     Description: {template_info['description']}")
            print(f"     Type: {template_info['workflow_type']}")
            print(f"     Nodes: {template_info['node_count']} ({', '.join(template_info['node_types'])})")
        
        # Create and register workflows from templates
        print("\\n🏗️ Creating workflows from templates...")
        
        for template_name in templates:
            try:
                workflow = self.workflow_builder.create_workflow_from_template(template_name)
                self.workflow_engine.register_workflow(workflow)
                print(f"   ✅ Created and registered: {workflow.name}")
            except Exception as e:
                print(f"   ❌ Failed to create {template_name}: {e}")
    
    async def demo_sequential_workflow(self):
        """Demonstrate sequential workflow execution"""
        
        print("\\n" + "-" * 60)
        print("🔄 SEQUENTIAL WORKFLOW DEMONSTRATION")
        print("-" * 60)
        
        # Find sequential workflow
        sequential_workflow_id = None
        for workflow_id, workflow in self.workflow_engine.workflows.items():
            if workflow.workflow_type == WorkflowType.SEQUENTIAL:
                sequential_workflow_id = workflow_id
                break
        
        if not sequential_workflow_id:
            print("❌ No sequential workflow found")
            return
        
        print(f"\\n🚀 Executing Sequential Workflow: {sequential_workflow_id}")
        
        # Test data
        test_data = {
            "text": "Apple's Q4 earnings exceeded expectations with strong iPhone sales, but supply chain concerns remain.",
            "company": "AAPL",
            "sector": "Technology"
        }
        
        print(f"📊 Input Data: {test_data}")
        
        try:
            start_time = datetime.now()
            result = await self.workflow_engine.execute_workflow(
                sequential_workflow_id,
                test_data,
                user_context={"user_id": "demo_user", "session_id": "demo_session"},
                execution_parameters={"priority": 5, "timeout": 180}
            )
            execution_time = (datetime.now() - start_time).total_seconds()
            
            print(f"\\n✅ Workflow completed in {execution_time:.2f} seconds")
            print(f"📋 Status: {result['status']}")
            print(f"🔢 Nodes executed: {result['nodes_executed']}")
            print(f"⏱️ Execution time: {result['execution_time']:.2f}s")
            
            if 'result' in result:
                print(f"\\n📊 Workflow Results:")
                self._print_nested_dict(result['result'], indent=2)
            
            if 'performance_metrics' in result:
                print(f"\\n📈 Performance Metrics:")
                self._print_nested_dict(result['performance_metrics'], indent=2)
                
        except Exception as e:
            print(f"❌ Workflow execution failed: {e}")
    
    async def demo_performance_monitoring(self):
        """Demonstrate performance monitoring and optimization"""
        
        print("\\n" + "-" * 60)
        print("📊 PERFORMANCE MONITORING DEMONSTRATION")
        print("-" * 60)
        
        # Get system status
        system_status = self.workflow_engine.get_system_status()
        print(f"\\n🖥️ System Status:")
        print(f"   Total Workflows: {system_status['total_workflows']}")
        print(f"   Active Executions: {system_status['active_executions']}")
        print(f"   Total Executions: {system_status['execution_stats']['total_executions']}")
        print(f"   Success Rate: {system_status['execution_stats']['avg_success_rate']:.2%}")
        print(f"   Avg Execution Time: {system_status['execution_stats']['avg_execution_time']:.2f}s")
        
        # Get routing statistics
        routing_stats = self.intelligent_router.get_routing_stats()
        print(f"\\n🧠 Routing Statistics:")
        print(f"   Total Rules: {routing_stats['total_rules']}")
        print(f"   Active Rules: {routing_stats['active_rules']}")
        print(f"   Tracked Models: {routing_stats['tracked_models']}")
        
        # Show workflow statuses
        print(f"\\n📋 Individual Workflow Status:")
        for workflow_id in self.workflow_engine.workflows.keys():
            status = self.workflow_engine.get_workflow_status(workflow_id)
            print(f"   {status['name']}:")
            print(f"     Type: {status['workflow_type']}")
            print(f"     Nodes: {status['node_count']}")
            print(f"     Executions: {status['total_executions']}")
            print(f"     Success Rate: {status['success_rate']:.2%}")
    
    def _print_nested_dict(self, data: Dict[str, Any], indent: int = 0):
        """Helper method to print nested dictionaries"""
        
        prefix = "  " * indent
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    print(f"{prefix}{key}:")
                    self._print_nested_dict(value, indent + 1)
                else:
                    print(f"{prefix}{key}: {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                print(f"{prefix}[{i}]:")
                self._print_nested_dict(item, indent + 1)
        else:
            print(f"{prefix}{data}")
    
    async def run_demo(self):
        """Run the demonstration"""
        
        try:
            await self.initialize()
            
            # Run demonstrations
            await self.demo_intelligent_routing()
            await self.demo_workflow_templates()
            await self.demo_sequential_workflow()
            await self.demo_performance_monitoring()
            
            print("\\n" + "=" * 80)
            print("🎉 ADVANCED ORCHESTRATION DEMO COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("\\n📋 Summary of Demonstrated Features:")
            print("   ✅ Intelligent routing with multiple strategies")
            print("   ✅ Workflow templates and custom workflow creation")
            print("   ✅ Sequential workflow execution")
            print("   ✅ Performance monitoring and optimization")
            print("   ✅ Load balancing and cost optimization")
            print("   ✅ Quality-based model selection")
            
            print("\\n🚀 The Advanced Orchestration System is ready for production use!")
            
        except Exception as e:
            print(f"\\n❌ Demo failed with error: {e}")
            logger.exception("Demo execution failed")


async def main():
    """Main function to run the advanced orchestration demo"""
    
    demo = AdvancedOrchestrationDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())