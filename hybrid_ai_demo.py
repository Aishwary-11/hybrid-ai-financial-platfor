"""
Hybrid AI System Demo
Demonstrates BlackRock Aladdin-inspired AI architecture with specialized models
"""

import requests
import json
import asyncio
from datetime import datetime
from typing import Dict, List

BASE_URL = "http://localhost:8000/api/v1/hybrid-ai"

class HybridAIDemo:
    """Demo client for hybrid AI system"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
    
    def earnings_analysis_demo(self):
        """Demonstrate earnings call analysis with specialized model"""
        print("📊 EARNINGS CALL ANALYSIS - SPECIALIZED MODEL")
        print("=" * 50)
        
        # Sample earnings call transcript
        earnings_transcript = """
        Thank you for joining our Q3 2024 earnings call. I'm pleased to report exceptional results 
        this quarter with revenue growth of 18% year-over-year, significantly exceeding our guidance 
        of 12-15%. Our digital transformation initiatives continue to drive operational efficiency, 
        with cloud services revenue up 35% and AI-powered solutions showing tremendous adoption.
        
        Looking ahead, we're optimistic about Q4 and expect continued momentum. We're raising our 
        full-year guidance to 16-18% revenue growth, up from our previous 12-15% range. Our strong 
        balance sheet positions us well for strategic investments in emerging technologies.
        
        However, we do see some headwinds from supply chain normalization and increased competition 
        in our core markets. We're monitoring these factors closely and remain confident in our 
        long-term strategy. Our focus on innovation and customer experience will continue to 
        differentiate us in the marketplace.
        """
        
        print("🎯 Analyzing earnings transcript...")
        
        try:
            response = requests.post(
                f"{self.base_url}/analyze/earnings",
                json={
                    "transcript": earnings_transcript,
                    "require_human_review": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result["analysis"]
                
                print(f"\n📈 INVESTMENT SIGNAL: {analysis['investment_signal'].upper()}")
                print(f"🎯 Signal Strength: {analysis['signal_strength']:.2f}")
                print(f"📊 Confidence: {result['confidence']:.2f}")
                print(f"✅ Validation Score: {result['validation_score']:.2f}")
                print(f"🛡️ Guardrails Passed: {result['guardrails_passed']}")
                
                print(f"\n🔍 Key Themes Identified:")
                for theme in analysis['key_themes']:
                    print(f"   • {theme.replace('_', ' ').title()}")
                
                print(f"\n⚠️ Risk Factors:")
                for risk in analysis['risk_factors']:
                    print(f"   • {risk.replace('_', ' ').title()}")
                
                print(f"\n💰 Price Target Adjustment: {analysis['price_target_adjustment'].replace('_', ' ').title()}")
                
                print(f"\n🤖 Model Details:")
                print(f"   Model Type: {result['model_type']}")
                print(f"   Processing Time: {result['metadata']['processing_time']}s")
                print(f"   Transcript Length: {result['metadata']['transcript_length']} chars")
                
            else:
                print(f"❌ Analysis failed: {response.json()}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def thematic_analysis_demo(self):
        """Demonstrate thematic investment identification"""
        print("\n🎯 THEMATIC INVESTMENT IDENTIFICATION")
        print("=" * 45)
        
        # Sample market data for thematic analysis
        market_data = {
            "news": [
                {
                    "content": "Major breakthrough in quantum computing achieved by tech giants, promising revolutionary advances in cryptography and drug discovery"
                },
                {
                    "content": "Renewable energy adoption accelerates globally with new solar efficiency records and massive wind farm investments"
                },
                {
                    "content": "Cybersecurity threats evolve as AI-powered attacks become more sophisticated, driving enterprise security spending"
                },
                {
                    "content": "Biotechnology companies report promising results in gene therapy trials for rare diseases"
                },
                {
                    "content": "Space technology sector sees unprecedented investment as satellite internet and space tourism gain momentum"
                }
            ],
            "patents": [
                {"category": "quantum_computing", "count": 150},
                {"category": "renewable_energy", "count": 200},
                {"category": "biotechnology", "count": 180}
            ],
            "regulations": [
                {"type": "data_privacy", "impact": "high"},
                {"type": "environmental_compliance", "impact": "medium"}
            ],
            "price_data": {
                "technology_sector": {"momentum": 0.7},
                "healthcare_sector": {"momentum": 0.6},
                "energy_sector": {"momentum": 0.8}
            }
        }
        
        print("🔍 Analyzing market data for thematic opportunities...")
        
        try:
            response = requests.post(
                f"{self.base_url}/analyze/thematic",
                json=market_data
            )
            
            if response.status_code == 200:
                result = response.json()
                themes = result["themes"]
                
                print(f"\n🚀 TOP THEMATIC OPPORTUNITIES:")
                print("-" * 40)
                
                for i, theme in enumerate(themes["top_themes"][:5], 1):
                    print(f"{i}. {theme['name'].replace('_', ' ').title()}")
                    print(f"   Strength: {theme['strength']:.2f}")
                    print(f"   Sources: {', '.join(theme['sources'])}")
                
                print(f"\n💼 INVESTMENT VEHICLES:")
                for theme_name, vehicles in themes["investment_vehicles"].items():
                    print(f"\n{theme_name.replace('_', ' ').title()}:")
                    for vehicle in vehicles:
                        print(f"   • {vehicle}")
                
                print(f"\n⚠️ RISK ASSESSMENT:")
                for theme_name, risks in themes["risk_assessment"].items():
                    print(f"\n{theme_name.replace('_', ' ').title()}:")
                    for risk in risks:
                        print(f"   • {risk}")
                
                print(f"\n⏰ TIME HORIZONS:")
                for theme_name, horizon in themes["time_horizon"].items():
                    print(f"   {theme_name.replace('_', ' ').title()}: {horizon}")
                
                print(f"\n📊 Analysis Confidence: {result['confidence']:.2f}")
                print(f"✅ Validation Score: {result['validation_score']:.2f}")
                
            else:
                print(f"❌ Analysis failed: {response.json()}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def comprehensive_analysis_demo(self):
        """Demonstrate comprehensive investment analysis"""
        print("\n🎯 COMPREHENSIVE INVESTMENT ANALYSIS")
        print("=" * 45)
        
        # Comprehensive analysis request
        analysis_request = {
            "earnings_transcript": """
            We delivered outstanding Q3 results with 22% revenue growth, driven by strong demand 
            for our AI and cloud solutions. Our digital transformation is accelerating, and we're 
            seeing excellent traction in enterprise markets. We're confident about the future and 
            are raising our full-year guidance. However, we're monitoring macroeconomic headwinds 
            and competitive pressures in certain segments.
            """,
            "market_data": {
                "news": [
                    {"content": "AI revolution transforms enterprise software landscape with unprecedented adoption rates"},
                    {"content": "Cloud computing demand surges as remote work becomes permanent fixture"},
                    {"content": "Cybersecurity investments reach record highs amid increasing threat landscape"}
                ],
                "price_data": {"technology_sector": {"momentum": 0.8}}
            }
        }
        
        print("🔄 Running comprehensive multi-model analysis...")
        
        try:
            response = requests.post(
                f"{self.base_url}/analyze/comprehensive",
                json=analysis_request
            )
            
            if response.status_code == 200:
                result = response.json()
                comprehensive = result["comprehensive_analysis"]
                components = result["component_analyses"]
                
                print(f"\n🎯 COMPREHENSIVE RECOMMENDATION:")
                print("-" * 40)
                print(f"Overall Signal: {comprehensive['overall_signal'].upper()}")
                print(f"Confidence: {comprehensive['confidence']:.2f}")
                print(f"Time Horizon: {comprehensive['time_horizon'].replace('_', ' ').title()}")
                
                print(f"\n💡 Key Insights:")
                for insight in comprehensive['key_insights']:
                    print(f"   • {insight}")
                
                print(f"\n📋 Recommended Actions:")
                for action in comprehensive['recommended_actions']:
                    print(f"   • {action}")
                
                print(f"\n⚠️ Risk Considerations:")
                for risk in comprehensive['risk_considerations']:
                    print(f"   • {risk}")
                
                print(f"\n🔍 COMPONENT ANALYSES:")
                print("-" * 25)
                
                if "earnings_analysis" in components:
                    earnings = components["earnings_analysis"]
                    print(f"Earnings Signal: {earnings['signal']} (Strength: {earnings['strength']:.2f})")
                    print(f"Key Themes: {', '.join(earnings['themes'])}")
                
                if "thematic_analysis" in components:
                    thematic = components["thematic_analysis"]
                    top_themes = [theme['name'] for theme in thematic['top_themes'][:3]]
                    print(f"Top Themes: {', '.join(top_themes)}")
                
                if "review_id" in comprehensive:
                    print(f"\n👥 Submitted for Expert Review: {comprehensive['review_id']}")
                
            else:
                print(f"❌ Analysis failed: {response.json()}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def human_review_demo(self):
        """Demonstrate human-in-the-loop review process"""
        print("\n👥 HUMAN-IN-THE-LOOP REVIEW SYSTEM")
        print("=" * 45)
        
        # Submit analysis for human review
        print("📤 Submitting analysis for expert review...")
        
        review_request = {
            "output": {
                "result": {
                    "investment_signal": "bullish",
                    "signal_strength": 0.75,
                    "recommendation": "Strong buy based on earnings momentum and thematic alignment"
                },
                "confidence": 0.8,
                "model_type": "specialized",
                "task_category": "market_analysis",
                "validation_score": 0.85,
                "guardrail_passed": True
            },
            "expert_type": "senior_portfolio_manager"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/human-review/submit",
                json=review_request
            )
            
            if response.status_code == 200:
                result = response.json()
                review_id = result["review_id"]
                
                print(f"✅ Review submitted successfully")
                print(f"   Review ID: {review_id}")
                print(f"   Expert Type: {result['expert_type']}")
                print(f"   Priority: {result['priority']}")
                print(f"   Estimated Review Time: {result['estimated_review_time']}")
                
                # Simulate expert feedback
                print(f"\n🧠 Simulating expert feedback...")
                
                expert_feedback = {
                    "reviewer": "Dr. Sarah Chen, Senior Portfolio Manager",
                    "rating": 8,
                    "agreement": 0.85,
                    "suggestions": [
                        "Consider sector rotation risks",
                        "Monitor regulatory developments",
                        "Assess correlation with market volatility"
                    ],
                    "additional_notes": "Strong analysis with good thematic alignment. Recommend monitoring macro factors."
                }
                
                feedback_response = requests.post(
                    f"{self.base_url}/human-review/feedback/{review_id}",
                    json=expert_feedback
                )
                
                if feedback_response.status_code == 200:
                    feedback_result = feedback_response.json()
                    print(f"✅ Expert feedback recorded")
                    print(f"   Reviewer: {feedback_result['reviewer']}")
                    print(f"   Rating: {feedback_result['rating']}/10")
                    print(f"   Agreement Level: {expert_feedback['agreement']:.0%}")
                    
                    print(f"\n💡 Expert Suggestions:")
                    for suggestion in expert_feedback['suggestions']:
                        print(f"   • {suggestion}")
                
            else:
                print(f"❌ Review submission failed: {response.json()}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def performance_report_demo(self):
        """Demonstrate system performance reporting"""
        print("\n📈 SYSTEM PERFORMANCE REPORT")
        print("=" * 35)
        
        try:
            response = requests.get(f"{self.base_url}/performance/report")
            
            if response.status_code == 200:
                report = response.json()
                
                print("🤖 SPECIALIZED MODELS:")
                print("-" * 25)
                for model_name, model_info in report["specialized_models"].items():
                    training_data = model_info["training_data"]
                    print(f"\n{model_name.replace('_', ' ').title()}:")
                    if training_data["name"]:
                        print(f"   Dataset: {training_data['name']}")
                        print(f"   Size: {training_data['size']:,} records")
                        print(f"   Quality Score: {training_data['quality_score']:.2f}")
                
                print(f"\n🛡️ GUARDRAIL SYSTEM:")
                print("-" * 20)
                guardrails = report["guardrail_effectiveness"]
                print(f"   Total Validation Rules: {guardrails['total_validations']}")
                print(f"   Protected Categories: {len(guardrails['validation_rules'])}")
                
                print(f"\n👥 HUMAN COLLABORATION:")
                print("-" * 25)
                collaboration = report["human_collaboration"]
                print(f"   Pending Reviews: {collaboration['pending_reviews']}")
                print(f"   Completed Reviews: {collaboration['completed_reviews']}")
                print(f"   Collaboration History: {collaboration['collaboration_history_size']} interactions")
                
                print(f"\n📊 MODEL PERFORMANCE:")
                print("-" * 20)
                for task_model, metrics in report["model_performance"].items():
                    print(f"\n{task_model.replace('_', ' ').title()}:")
                    print(f"   Total Requests: {metrics['total_requests']}")
                    print(f"   Avg Confidence: {metrics['avg_confidence']:.2f}")
                    print(f"   Validation Score: {metrics['avg_validation_score']:.2f}")
                    print(f"   Guardrail Pass Rate: {metrics['guardrail_pass_rate']:.1%}")
                    print(f"   Human Review Rate: {metrics['human_review_rate']:.1%}")
                
            else:
                print(f"❌ Report generation failed: {response.json()}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def model_validation_demo(self):
        """Demonstrate model output validation"""
        print("\n🛡️ MODEL OUTPUT VALIDATION")
        print("=" * 35)
        
        # Test with valid output
        print("✅ Testing valid output...")
        
        valid_output = {
            "output": {
                "result": {
                    "investment_signal": "bullish",
                    "signal_strength": 0.75,
                    "key_themes": ["digital_transformation", "ai_adoption"],
                    "risk_factors": ["competition", "regulatory"]
                },
                "confidence": 0.8,
                "model_type": "specialized",
                "task_category": "earnings_prediction",
                "validation_score": 0.85
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/models/validate",
                json=valid_output
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   Validation Passed: {result['validation_passed']}")
                print(f"   Issues Found: {len(result['issues'])}")
                print(f"   Validation Score: {result['validation_score']:.2f}")
                
                if result['recommendations']:
                    print(f"   Recommendations:")
                    for rec in result['recommendations']:
                        print(f"     • {rec}")
            
            # Test with invalid output
            print(f"\n❌ Testing invalid output...")
            
            invalid_output = {
                "output": {
                    "result": {
                        "investment_signal": "invalid_signal",  # Invalid signal
                        "signal_strength": 1.5,  # Out of range
                        "key_themes": "not_a_list"  # Wrong type
                    },
                    "confidence": -0.2,  # Invalid confidence
                    "model_type": "specialized",
                    "task_category": "earnings_prediction",
                    "validation_score": 0.85
                }
            }
            
            response = requests.post(
                f"{self.base_url}/models/validate",
                json=invalid_output
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   Validation Passed: {result['validation_passed']}")
                print(f"   Issues Found: {len(result['issues'])}")
                
                if result['issues']:
                    print(f"   Issues:")
                    for issue in result['issues']:
                        print(f"     • {issue}")
                
                if result['recommendations']:
                    print(f"   Recommendations:")
                    for rec in result['recommendations']:
                        print(f"     • {rec}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def run_full_demo(self):
        """Run complete hybrid AI system demonstration"""
        print("🤖" * 25)
        print("🏛️  HYBRID AI ENGINE - BLACKROCK ALADDIN ARCHITECTURE")
        print("🤖" * 25)
        print("\n🎯 Institutional-grade AI with specialized models and human expertise")
        print("📊 Proprietary datasets • 🛡️ Robust guardrails • 👥 Human-in-the-loop")
        
        try:
            # Core AI demonstrations
            self.earnings_analysis_demo()
            self.thematic_analysis_demo()
            self.comprehensive_analysis_demo()
            
            # Human collaboration
            self.human_review_demo()
            
            # System monitoring
            self.performance_report_demo()
            self.model_validation_demo()
            
            print("\n" + "🎉" * 25)
            print("✅ HYBRID AI SYSTEM DEMONSTRATION COMPLETE!")
            print("🎯 Key Features Demonstrated:")
            print("   • 🧠 Specialized Models (Earnings Analysis, Thematic Identification)")
            print("   • 🏛️ Foundation Model Integration (GPT-4 style reasoning)")
            print("   • 🛡️ Comprehensive Guardrail System")
            print("   • 👥 Human-in-the-Loop Validation")
            print("   • 📊 Performance Monitoring & Reporting")
            print("   • 🔍 Output Validation & Quality Control")
            print("   • 🤝 Human-AI Collaboration History")
            print("\n🏆 INSTITUTIONAL-GRADE AI FOR INVESTMENT MANAGEMENT")
            print("🎉" * 25)
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            print("Make sure the server is running: python main.py")


if __name__ == "__main__":
    demo = HybridAIDemo()
    demo.run_full_demo()