"""
Financial Sentiment Analysis Model Demo
Test the specialized sentiment analysis capabilities
"""

import asyncio
import json
from datetime import datetime
from app.core.sentiment_analysis_model import (
    FinancialSentimentModel, 
    analyze_news_sentiment,
    batch_sentiment_analysis,
    SentimentSource
)


async def demo_sentiment_analysis():
    """Demonstrate financial sentiment analysis capabilities"""
    
    print("🔍 Financial Sentiment Analysis Model Demo")
    print("=" * 50)
    
    # Initialize the sentiment model
    sentiment_model = FinancialSentimentModel()
    
    # Test cases with different types of financial content
    test_cases = [
        {
            "name": "Positive Earnings News",
            "text": """Apple Inc. reported strong quarterly earnings that beat analyst estimates by 15%. 
                      Revenue grew 12% year-over-year to $95.3 billion, driven by robust iPhone sales 
                      and expanding services revenue. The company raised its full-year guidance and 
                      announced a $10 billion share buyback program. Management expressed confidence 
                      in continued growth momentum.""",
            "symbols": ["AAPL"],
            "source": "news_articles"
        },
        {
            "name": "Negative Market Outlook",
            "text": """Tesla faces mounting challenges as competition intensifies in the EV market. 
                      Recent price cuts have pressured margins, and the company missed delivery 
                      targets for the third consecutive quarter. Regulatory concerns over 
                      autonomous driving features add uncertainty. Analysts are lowering 
                      price targets amid weakening demand.""",
            "symbols": ["TSLA"],
            "source": "analyst_reports"
        },
        {
            "name": "Mixed Thematic Analysis",
            "text": """The artificial intelligence sector shows mixed signals. While major tech 
                      companies continue heavy AI investments, concerns about regulation and 
                      high valuations persist. NVIDIA's data center revenue remains strong, 
                      but competition from AMD and custom chips poses risks. The sector 
                      faces both tremendous opportunity and significant headwinds.""",
            "symbols": ["NVDA", "AMD"],
            "source": "news_articles"
        },
        {
            "name": "Earnings Call Transcript",
            "text": """We're pleased to report another strong quarter with revenue exceeding 
                      expectations. Our digital transformation initiatives are paying off, 
                      with cloud revenue up 25% and margin expansion across all segments. 
                      Looking ahead, we're optimistic about market conditions and expect 
                      continued growth. We're raising our full-year guidance and remain 
                      confident in our strategic direction.""",
            "symbols": ["MSFT"],
            "source": "earnings_calls"
        }
    ]
    
    print("\n📊 Running Sentiment Analysis Tests")
    print("-" * 40)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Symbols: {', '.join(test_case['symbols'])}")
        print(f"   Source: {test_case['source']}")
        
        # Prepare input data
        input_data = {
            "text": test_case["text"],
            "symbols": test_case["symbols"],
            "source": test_case["source"],
            "include_market_impact": True
        }
        
        # Run sentiment analysis
        try:
            result = sentiment_model.predict(input_data)
            
            print(f"   ✅ Analysis completed")
            print(f"   📈 Overall Sentiment: {result.result['overall_sentiment']}")
            print(f"   📊 Sentiment Score: {result.result['sentiment_score']:.3f}")
            print(f"   🎯 Confidence: {result.confidence:.3f}")
            print(f"   ⚡ Processing Time: {result.metadata.get('processing_method', 'N/A')}")
            
            # Show market impact predictions
            if result.result.get("market_impact_prediction"):
                print(f"   💹 Market Impact Predictions:")
                for symbol, impact in result.result["market_impact_prediction"].items():
                    print(f"      {symbol}: {impact['expected_price_impact_1d']:.2%} (1d), "
                          f"Confidence: {impact['confidence']:.2f}")
            
            # Show sentiment drivers
            if result.result.get("sentiment_drivers"):
                print(f"   🔍 Key Drivers: {len(result.result['sentiment_drivers'])} identified")
                for driver in result.result["sentiment_drivers"][:3]:  # Show top 3
                    print(f"      - {driver['driver']} ({driver['sentiment_direction']})")
            
        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
    
    # Test batch processing
    print(f"\n🔄 Testing Batch Processing")
    print("-" * 30)
    
    batch_data = [
        {
            "text": "Strong earnings beat expectations with revenue growth of 15%",
            "symbols": ["GOOGL"],
            "source": "news_articles"
        },
        {
            "text": "Company faces regulatory challenges and margin pressure",
            "symbols": ["META"],
            "source": "news_articles"
        },
        {
            "text": "Innovative product launch drives customer excitement and pre-orders",
            "symbols": ["AMZN"],
            "source": "news_articles"
        }
    ]
    
    try:
        batch_results = batch_sentiment_analysis(batch_data)
        print(f"   ✅ Batch processing completed")
        print(f"   📊 Processed: {len(batch_results)} items")
        
        for i, result in enumerate(batch_results, 1):
            if "error" not in result:
                sentiment = result["output"]["overall_sentiment"]
                confidence = result["confidence"]
                print(f"   {i}. Sentiment: {sentiment}, Confidence: {confidence:.2f}")
            else:
                print(f"   {i}. Error: {result['error']}")
                
    except Exception as e:
        print(f"   ❌ Batch processing failed: {e}")
    
    # Test sentiment trends
    print(f"\n📈 Testing Sentiment Trends")
    print("-" * 25)
    
    try:
        # First add some sentiment data
        for symbol in ["AAPL", "TSLA", "NVDA"]:
            trends = sentiment_model.get_sentiment_trends(symbol, days=30)
            if "error" not in trends:
                print(f"   {symbol}: {trends['data_points']} data points, "
                      f"Avg: {trends['average_sentiment']:.3f}, "
                      f"Trend: {trends['trend_direction']}")
            else:
                print(f"   {symbol}: {trends['error']}")
                
    except Exception as e:
        print(f"   ❌ Trend analysis failed: {e}")
    
    # Test model validation
    print(f"\n🔍 Testing Model Validation")
    print("-" * 25)
    
    # Test with valid output
    valid_output = {
        "overall_sentiment": "bullish",
        "sentiment_score": 0.65,
        "confidence": 0.82,
        "entity_sentiments": {"overall": 0.65, "entity_aapl": 0.70},
        "sentiment_drivers": [{"driver": "earnings_beat", "strength": 0.8}],
        "market_impact_prediction": {}
    }
    
    is_valid, validation_score = sentiment_model.validate_output(valid_output)
    print(f"   Valid Output: {'✅ Passed' if is_valid else '❌ Failed'} "
          f"(Score: {validation_score:.2f})")
    
    # Test with invalid output
    invalid_output = {
        "overall_sentiment": "bullish",
        "sentiment_score": 1.5,  # Invalid range
        "confidence": -0.1,      # Invalid range
        "entity_sentiments": "invalid_type"  # Wrong type
    }
    
    is_valid, validation_score = sentiment_model.validate_output(invalid_output)
    print(f"   Invalid Output: {'✅ Passed' if is_valid else '❌ Failed'} "
          f"(Score: {validation_score:.2f})")
    
    print(f"\n🎯 Demo Summary")
    print("-" * 15)
    print(f"   • Financial sentiment analysis model successfully tested")
    print(f"   • Supports multiple data sources and market impact prediction")
    print(f"   • Includes comprehensive validation and error handling")
    print(f"   • Ready for integration with hybrid AI architecture")
    print(f"   • Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def test_quick_sentiment():
    """Quick sentiment analysis test"""
    
    print("\n🚀 Quick Sentiment Test")
    print("-" * 20)
    
    # Test the utility function
    news_text = """Microsoft reported exceptional quarterly results with cloud revenue 
                   surging 30% and AI services showing tremendous adoption. The company 
                   raised guidance and announced strategic partnerships in the AI space."""
    
    result = analyze_news_sentiment(news_text, ["MSFT"])
    
    print(f"   Text: {news_text[:100]}...")
    print(f"   Sentiment: {result['overall_sentiment']}")
    print(f"   Score: {result['sentiment_score']:.3f}")
    print(f"   Confidence: {result['confidence']:.3f}")


if __name__ == "__main__":
    print("Starting Financial Sentiment Analysis Demo...")
    
    # Run the main demo
    asyncio.run(demo_sentiment_analysis())
    
    # Run quick test
    test_quick_sentiment()
    
    print("\n✅ Demo completed successfully!")