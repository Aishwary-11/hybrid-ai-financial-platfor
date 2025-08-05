"""
AI Orchestrator Demo
Demonstrates the BlackRock Aladdin-inspired hybrid AI architecture
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List

BASE_URL = "http://localhost:8000/api/v1/ai"

class AIOrchestratorDemo:
    """Demo client for AI Orchestrator system"""
    
    def __init__(self):
        self.base_url = BASE_URL
    
    def test_server_connection(self):
        """Test if the AI orchestrator is available"""
        try:
            response = requests.get(f"{self.base_url}/system/status")
            if response.status_code == 200:
                print("✅ AI Orchestrator is operational!")
                return True
            else:
                print("❌ AI Orchestrator responded with error")
                return False
        except:
            print("❌ AI Orchestrator is not available. Please start the server.")
            return False
    
    def system_status_demo(self):
        """Demonstrate system status and model availability"""
        print("\n🔍 SYSTEM STATUS & MODEL INVENTORY")
        print("=" * 45)
        
        try:
            response = requests.get(f"{self.base_url}/system/status")
            if response.status_code == 200:
                status = response.json()
                
                print(f"🏥 System Health: {status['system_health'].upper()}")
                print(f"🤖 Total Models: {status['total_models']}")
                
                print(f"\n🏛️ Foundation Models:")
                for model_id, info in status['foundation_models'].items():
                    health_icon = "✅" if info['healthy'] else "❌"
                    print(f"   {health_icon} {model_id}")
                    print(f"      Capabilities: {len(info['capabilities'])} query types")
                
                print(f"\n🎯 Specialized Models:")
                for model_id, info in status['specialized_models'].items():
                    health_icon = "✅" if info['healthy'] else "❌"
                    print(f"   {health_icon} {model_id}")
                    print(f"      Specialization: {info['specialization']}")
                    print(f"      Training Data: {info['training_data_size']:,} samples")
            else:
                print("❌ Failed to get system status")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def foundation_models_demo(self):
        """Demonstrate foundation model capabilities"""
        print("\n🏛️ FOUNDATION MODELS DEMONSTRATION")
        print("=" * 45)
        
        try:
            response = requests.get(f"{self.base_url}/models/foundation")
            if response.status_code == 200:
                models_info = response.json()
                
                print(f"📊 Available Foundation Models: {models_info['total_models']}")
                
                for model_id, info in models_info['foundation_models'].items():
                    print(f"\n🤖 {model_id.upper()}:")
                    print(f"   Health: {'✅ Healthy' if info['healthy'] else '❌ Unhealthy'}")
                    print(f"   Type: {info['model_type'].title()}")
                    print(f"   Capabilities: {len(info['capabilities'])} query types")
                    
                    # Show some capabilities
                    capabilities = info['capabilities'][:3]  # Show first 3
                    for cap in capabilities:
                        print(f"     • {cap.replace('_', ' ').title()}")
                    if len(info['capabilities']) > 3:
                        print(f"     • ... and {len(info['capabilities']) - 3} more")
            else:
                print("❌ Failed to get foundation models info")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def specialized_models_demo(self):
        """Demonstrate specialized model capabilities"""
        print("\n🎯 SPECIALIZED MODELS DEMONSTRATION")
        print("=" * 45)
        
        try:
            response = requests.get(f"{self.base_url}/models/specialized")
            if response.status_code == 200:
                models_info = response.json()
                
                print(f"🧠 Available Specialized Models: {models_info['total_models']}")
                
                for model_id, info in models_info['specialized_models'].items():
                    print(f"\n🎯 {model_id.upper()}:")
                    print(f"   Health: {'✅ Healthy' if info['healthy'] else '❌ Unhealthy'}")
                    print(f"   Specialization: {info['specialization'].replace('_', ' ').title()}")
                    print(f"   Training Data: {info['training_data_size']:,} samples")
                    print(f"   Model Path: {info['model_path']}")
                    
                    if info['accuracy_metrics']:
                        print(f"   Accuracy Metrics: {info['accuracy_metrics']}")
            else:
                print("❌ Failed to get specialized models info")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def earnings_analysis_demo(self):
        """Demonstrate specialized earnings analysis"""
        print("\n📊 EARNINGS ANALYSIS - SPECIALIZED MODEL")
        print("=" * 45)
        
        sample_earnings = {
            "transcript": """
            Thank you for joining our Q4 2024 earnings call. I'm excited to report outstanding results 
            with revenue growth of 22% year-over-year, significantly beating our guidance of 18-20%. 
            Our AI and cloud initiatives are driving exceptional performance, with cloud revenue up 40% 
            and AI services showing 60% growth. We're raising our 2025 guidance and expect continued 
            strong momentum. Our balance sheet remains robust, and we're well-positioned for strategic 
            investments in emerging technologies. However, we're monitoring macroeconomic headwinds 
            and competitive pressures in certain segments.
            """,
            "symbols": ["TECH"],
            "time_horizon": "short_term",
            "context": {"quarter": "Q4", "year": 2024}
        }
        
        print("🎯 Analyzing earnings transcript with specialized model...")
        
        try:
            response = requests.post(f"{self.base_url}/analyze/earnings", json=sample_earnings)
            
            if response.status_code == 200:
                result = response.json()
                analysis = result['earnings_analysis']
                
                print(f"\n📈 EARNINGS ANALYSIS RESULTS:")
                print(f"   Confidence: {result['confidence']:.0%}")
                print(f"   Specialized Model Used: {'✅ Yes' if result['specialized_model_used'] else '❌ No'}")
                print(f"   Human Review Required: {'⚠️ Yes' if result['human_review_required'] else '✅ No'}")
                print(f"   Processing Time: {result['processing_time']:.2f}s")
                
                if 'primary_prediction' in analysis:
                    pred = analysis['primary_prediction']
                    print(f"\n💡 Key Insights:")
                    print(f"   Investment Signal: {pred.get('investment_signal', 'N/A').upper()}")
                    print(f"   Signal Strength: {pred.get('signal_strength', 0):.0%}")
                    print(f"   Key Themes: {', '.join(pred.get('key_themes', []))}")
                    print(f"   Price Target: {pred.get('price_target_adjustment', 'maintain').replace('_', ' ').title()}")
                
                print(f"\n🤖 Model Information:")
                print(f"   Primary Model: {analysis.get('primary_model', 'Unknown')}")
                print(f"   Model Count: {analysis.get('model_count', 1)}")
                
            else:
                print(f"❌ Earnings analysis failed: {response.json()}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def thematic_analysis_demo(self):
        """Demonstrate thematic investment identification"""
        print("\n🎯 THEMATIC INVESTMENT IDENTIFICATION")
        print("=" * 45)
        
        market_data = {
            "symbols": ["TECH", "CLEAN", "CYBER"],
            "time_horizon": "long_term",
            "news_data": [
                "AI breakthrough in quantum computing shows promise",
                "Renewable energy adoption accelerates globally",
                "Cybersecurity threats drive enterprise spending"
            ],
            "patent_data": ["quantum_computing", "battery_tech", "ai_chips"],
            "regulatory_data": ["clean_energy_incentives", "data_privacy_laws"]
        }
        
        print("🔍 Identifying thematic investment opportunities...")
        
        try:
            response = requests.post(f"{self.base_url}/analyze/thematic", json=market_data)
            
            if response.status_code == 200:
                result = response.json()
                analysis = result['thematic_analysis']
                
                print(f"\n🚀 THEMATIC OPPORTUNITIES:")
                print(f"   Overall Confidence: {result['confidence']:.0%}")
                print(f"   Processing Time: {result['processing_time']:.2f}s")
                
                themes = result.get('themes_identified', [])
                if themes:
                    print(f"\n🎯 Top Investment Themes:")
                    for i, theme in enumerate(themes[:3], 1):
                        if isinstance(theme, dict):
                            print(f"   {i}. {theme.get('name', 'Unknown').replace('_', ' ').title()}")
                            print(f"      Strength: {theme.get('strength', 0):.0%}")
                            print(f"      Sources: {', '.join(theme.get('sources', []))}")
                
                if 'primary_prediction' in analysis:
                    pred = analysis['primary_prediction']
                    if 'investment_vehicles' in pred:
                        print(f"\n💼 Investment Vehicles:")
                        for theme, vehicles in list(pred['investment_vehicles'].items())[:2]:
                            print(f"   {theme.replace('_', ' ').title()}:")
                            for vehicle in vehicles[:2]:
                                print(f"     • {vehicle}")
                
            else:
                print(f"❌ Thematic analysis failed: {response.json()}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def routing_strategies_demo(self):
        """Demonstrate different routing strategies"""
        print("\n🔀 ROUTING STRATEGIES COMPARISON")
        print("=" * 40)
        
        test_query = {
            "query": "Analyze the investment potential of AI companies",
            "query_type": "general_analysis",
            "symbols": ["NVDA", "GOOGL", "MSFT"]
        }
        
        print("🧪 Testing different routing strategies on the same query...")
        
        try:
            response = requests.post(f"{self.base_url}/routing/test", json=test_query)
            
            if response.status_code == 200:
                results = response.json()
                comparison = results['routing_comparison']
                
                print(f"\n📊 ROUTING STRATEGY COMPARISON:")
                print("-" * 60)
                print(f"{'Strategy':<20} {'Confidence':<12} {'Models':<8} {'Time':<8} {'Review':<8}")
                print("-" * 60)
                
                for strategy, metrics in comparison.items():
                    if 'error' not in metrics:
                        strategy_name = strategy.replace('_', ' ').title()
                        confidence = f"{metrics['confidence']:.0%}"
                        models = str(metrics['models_used'])
                        time_taken = f"{metrics['processing_time']:.2f}s"
                        review = "Yes" if metrics['human_review_required'] else "No"
                        
                        print(f"{strategy_name:<20} {confidence:<12} {models:<8} {time_taken:<8} {review:<8}")
                    else:
                        print(f"{strategy:<20} ERROR: {metrics['error']}")
                
                print("\n💡 Strategy Insights:")
                print("   • Specialization First: Prioritizes domain expertise")
                print("   • Parallel Ensemble: Uses multiple models simultaneously")
                print("   • Confidence Based: Routes based on historical performance")
                print("   • Foundation Fallback: Systematic fallback approach")
                
            else:
                print(f"❌ Routing test failed: {response.json()}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def comprehensive_analysis_demo(self):
        """Demonstrate comprehensive multi-model analysis"""
        print("\n🎯 COMPREHENSIVE MULTI-MODEL ANALYSIS")
        print("=" * 45)
        
        comprehensive_query = {
            "query": "Provide comprehensive investment analysis for technology sector",
            "query_type": "general_analysis",
            "symbols": ["AAPL", "GOOGL", "MSFT", "NVDA"],
            "time_horizon": "medium_term",
            "risk_tolerance": "moderate",
            "user_context": {
                "portfolio_size": 1000000,
                "investment_style": "growth",
                "sector_focus": "technology"
            }
        }
        
        print("🔄 Running comprehensive analysis with hybrid AI system...")
        
        try:
            response = requests.post(f"{self.base_url}/analyze", json=comprehensive_query)
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"\n📊 COMPREHENSIVE ANALYSIS RESULTS:")
                print(f"   Query ID: {result['query_id']}")
                print(f"   Overall Confidence: {result['confidence']:.0%}")
                print(f"   Processing Time: {result['processing_time']:.2f}s")
                print(f"   Human Review Required: {'⚠️ Yes' if result['human_review_required'] else '✅ No'}")
                print(f"   Routing Strategy: {result['routing_strategy'].replace('_', ' ').title()}")
                
                print(f"\n🤖 Models Utilized:")
                for model in result['models_used']:
                    model_type_icon = "🏛️" if model['model_type'] == 'foundation' else "🎯"
                    print(f"   {model_type_icon} {model['model_id']}")
                    print(f"      Type: {model['model_type'].title()}")
                    print(f"      Confidence: {model['confidence']:.0%}")
                    print(f"      Processing Time: {model['processing_time']:.2f}s")
                
                analysis = result['analysis']
                if 'primary_prediction' in analysis:
                    pred = analysis['primary_prediction']
                    print(f"\n💡 Key Analysis Points:")
                    if isinstance(pred, dict):
                        for key, value in list(pred.items())[:3]:
                            if isinstance(value, (str, int, float)):
                                print(f"   • {key.replace('_', ' ').title()}: {value}")
                
                print(f"\n📈 Investment Implications:")
                print(f"   • Multi-model consensus provides robust analysis")
                print(f"   • Hybrid approach combines general and specialized insights")
                print(f"   • Confidence level indicates reliability of recommendations")
                
            else:
                print(f"❌ Comprehensive analysis failed: {response.json()}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def performance_metrics_demo(self):
        """Demonstrate system performance monitoring"""
        print("\n📈 SYSTEM PERFORMANCE METRICS")
        print("=" * 35)
        
        try:
            response = requests.get(f"{self.base_url}/performance/metrics")
            
            if response.status_code == 200:
                metrics = response.json()
                
                print(f"📊 Performance Overview:")
                print(f"   Total Queries Processed: {metrics['total_queries_processed']}")
                print(f"   System Status: {metrics['system_uptime'].title()}")
                
                if metrics['performance_metrics']:
                    print(f"\n🎯 Model Performance Insights:")
                    print(f"   • {len(metrics['performance_metrics'])} performance data points collected")
                    print(f"   • Continuous monitoring active")
                    print(f"   • Performance optimization in progress")
                else:
                    print(f"\n🎯 Performance Monitoring:")
                    print(f"   • System ready for performance tracking")
                    print(f"   • Metrics will be collected as queries are processed")
                
            else:
                print(f"❌ Performance metrics failed: {response.json()}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def run_full_demo(self):
        """Run complete AI orchestrator demonstration"""
        print("🤖" * 25)
        print("🏛️  AI ORCHESTRATOR - BLACKROCK ALADDIN ARCHITECTURE")
        print("🤖" * 25)
        print("\n🎯 Hybrid AI System with Foundation + Specialized Models")
        print("🧠 GPT-4, Gemini, Claude + Custom Financial Models")
        
        # Test connection
        if not self.test_server_connection():
            return
        
        try:
            # System overview
            self.system_status_demo()
            time.sleep(1)
            
            # Model demonstrations
            self.foundation_models_demo()
            time.sleep(1)
            
            self.specialized_models_demo()
            time.sleep(1)
            
            # Analysis demonstrations
            self.earnings_analysis_demo()
            time.sleep(1)
            
            self.thematic_analysis_demo()
            time.sleep(1)
            
            # Advanced features
            self.routing_strategies_demo()
            time.sleep(1)
            
            self.comprehensive_analysis_demo()
            time.sleep(1)
            
            self.performance_metrics_demo()
            
            print("\n" + "🎉" * 25)
            print("✅ AI ORCHESTRATOR DEMONSTRATION COMPLETE!")
            print("🎯 Key Features Demonstrated:")
            print("   • 🏛️ Foundation Models (GPT-4, Gemini, Claude)")
            print("   • 🎯 Specialized Models (Earnings, Thematic, Sentiment)")
            print("   • 🔀 Intelligent Routing Strategies")
            print("   • 🤖 Multi-Model Orchestration")
            print("   • 📊 Comprehensive Analysis Pipeline")
            print("   • 📈 Performance Monitoring")
            print("   • 🛡️ Quality Assurance & Confidence Scoring")
            print("\n🏆 INSTITUTIONAL-GRADE AI FOR INVESTMENT MANAGEMENT")
            print("🎉" * 25)
            
        except KeyboardInterrupt:
            print("\n🛑 Demo stopped by user")
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")


if __name__ == "__main__":
    demo = AIOrchestratorDemo()
    demo.run_full_demo()