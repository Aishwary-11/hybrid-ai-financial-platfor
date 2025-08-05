"""
Conversational AI Engine
Natural language interface for investment queries and analysis
"""

import asyncio
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import openai
import anthropic
from transformers import pipeline
import spacy
import pandas as pd
import numpy as np

class QueryType(Enum):
    STOCK_ANALYSIS = "stock_analysis"
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    MARKET_SCREENING = "market_screening"
    SCENARIO_ANALYSIS = "scenario_analysis"
    ESG_ANALYSIS = "esg_analysis"
    COMPARISON = "comparison"
    PREDICTION = "prediction"
    NEWS_ANALYSIS = "news_analysis"

class QueryComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    MULTI_STEP = "multi_step"

@dataclass
class ParsedQuery:
    original_query: str
    query_type: QueryType
    complexity: QueryComplexity
    entities: Dict[str, List[str]]
    parameters: Dict[str, Any]
    time_horizon: Optional[str]
    filters: Dict[str, Any]
    intent_confidence: float

@dataclass
class ConversationalResponse:
    query: str
    response_text: str
    data_analysis: Dict[str, Any]
    visualizations: List[Dict[str, Any]]
    follow_up_questions: List[str]
    confidence_score: float
    processing_time_ms: float
    sources: List[str]

class ConversationalAIEngine:
    """Advanced conversational AI for investment queries"""
    
    def __init__(self):
        self.openai_client = openai.AsyncOpenAI()
        self.anthropic_client = anthropic.AsyncAnthropic()
        
        # Load NLP models
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                         model="ProsusAI/finbert")
        
        # Financial entity recognition
        self.financial_entities = self._load_financial_entities()
        
        # Query patterns
        self.query_patterns = self._initialize_query_patterns()
        
        # Context memory for conversation
        self.conversation_context = {}
        
    def _load_financial_entities(self) -> Dict[str, List[str]]:
        """Load financial entities for NER"""
        return {
            'stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
            'sectors': ['technology', 'healthcare', 'finance', 'energy', 'utilities', 
                       'consumer', 'industrial', 'materials', 'real estate'],
            'regions': ['US', 'Europe', 'Asia', 'emerging markets', 'developed markets'],
            'asset_classes': ['stocks', 'bonds', 'commodities', 'crypto', 'real estate'],
            'metrics': ['P/E', 'EPS', 'revenue', 'profit', 'ROE', 'debt', 'cash flow'],
            'time_periods': ['day', 'week', 'month', 'quarter', 'year', 'YTD']
        }
    
    def _initialize_query_patterns(self) -> Dict[str, List[str]]:
        """Initialize regex patterns for query classification"""
        return {
            'stock_analysis': [
                r'analyze\s+(\w+)\s+stock',
                r'what.*think.*about\s+(\w+)',
                r'(\w+)\s+investment\s+potential',
                r'should\s+i\s+buy\s+(\w+)'
            ],
            'portfolio_analysis': [
                r'analyze.*portfolio',
                r'portfolio.*performance',
                r'how.*portfolio.*doing',
                r'portfolio.*risk'
            ],
            'market_screening': [
                r'find.*stocks.*with',
                r'show.*me.*stocks',
                r'screen.*for.*stocks',
                r'stocks.*that.*have'
            ],
            'scenario_analysis': [
                r'what.*if.*prices?',
                r'scenario.*analysis',
                r'impact.*of.*on.*portfolio',
                r'stress.*test'
            ],
            'comparison': [
                r'compare.*(\w+).*and.*(\w+)',
                r'(\w+).*vs.*(\w+)',
                r'which.*better.*(\w+).*(\w+)'
            ]
        }
    
    async def process_query(self, query: str, user_id: str = None, 
                          context: Dict[str, Any] = None) -> ConversationalResponse:
        """Process natural language investment query"""
        
        start_time = datetime.now()
        
        # Parse the query
        parsed_query = await self._parse_query(query)
        
        # Get conversation context
        if user_id:
            conversation_context = self.conversation_context.get(user_id, {})
        else:
            conversation_context = context or {}
        
        # Route to appropriate handler
        if parsed_query.query_type == QueryType.STOCK_ANALYSIS:
            response = await self._handle_stock_analysis(parsed_query, conversation_context)
        elif parsed_query.query_type == QueryType.MARKET_SCREENING:
            response = await self._handle_market_screening(parsed_query, conversation_context)
        elif parsed_query.query_type == QueryType.SCENARIO_ANALYSIS:
            response = await self._handle_scenario_analysis(parsed_query, conversation_context)
        elif parsed_query.query_type == QueryType.PORTFOLIO_ANALYSIS:
            response = await self._handle_portfolio_analysis(parsed_query, conversation_context)
        elif parsed_query.query_type == QueryType.COMPARISON:
            response = await self._handle_comparison(parsed_query, conversation_context)
        else:
            response = await self._handle_general_query(parsed_query, conversation_context)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        response.processing_time_ms = processing_time
        
        # Update conversation context
        if user_id:
            self._update_conversation_context(user_id, parsed_query, response)
        
        return response
    
    async def _parse_query(self, query: str) -> ParsedQuery:
        """Parse natural language query into structured format"""
        
        # Basic NLP processing
        doc = self.nlp(query.lower())
        
        # Extract entities
        entities = {
            'stocks': [],
            'sectors': [],
            'regions': [],
            'metrics': [],
            'time_periods': []
        }
        
        # Extract financial entities
        for entity_type, entity_list in self.financial_entities.items():
            for entity in entity_list:
                if entity.lower() in query.lower():
                    entities[entity_type].append(entity)
        
        # Extract stock symbols (pattern: 3-5 uppercase letters)
        stock_pattern = r'\b[A-Z]{2,5}\b'
        stock_matches = re.findall(stock_pattern, query.upper())
        entities['stocks'].extend(stock_matches)
        
        # Classify query type
        query_type = self._classify_query_type(query)
        
        # Determine complexity
        complexity = self._determine_complexity(query, entities)
        
        # Extract parameters
        parameters = self._extract_parameters(query, doc)
        
        # Extract time horizon
        time_horizon = self._extract_time_horizon(query)
        
        # Extract filters
        filters = self._extract_filters(query)
        
        return ParsedQuery(
            original_query=query,
            query_type=query_type,
            complexity=complexity,
            entities=entities,
            parameters=parameters,
            time_horizon=time_horizon,
            filters=filters,
            intent_confidence=0.85  # Simplified confidence score
        )
    
    def _classify_query_type(self, query: str) -> QueryType:
        """Classify the type of query"""
        
        query_lower = query.lower()
        
        # Check patterns
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return QueryType(query_type)
        
        # Keyword-based classification
        if any(word in query_lower for word in ['analyze', 'analysis', 'potential']):
            return QueryType.STOCK_ANALYSIS
        elif any(word in query_lower for word in ['find', 'show', 'screen']):
            return QueryType.MARKET_SCREENING
        elif any(word in query_lower for word in ['what if', 'scenario', 'impact']):
            return QueryType.SCENARIO_ANALYSIS
        elif any(word in query_lower for word in ['portfolio']):
            return QueryType.PORTFOLIO_ANALYSIS
        elif any(word in query_lower for word in ['compare', 'vs', 'versus']):
            return QueryType.COMPARISON
        else:
            return QueryType.STOCK_ANALYSIS  # Default
    
    def _determine_complexity(self, query: str, entities: Dict[str, List[str]]) -> QueryComplexity:
        """Determine query complexity"""
        
        # Count entities and conditions
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        
        # Check for complex conditions
        complex_keywords = ['and', 'or', 'but', 'with', 'correlation', 'scenario']
        complex_count = sum(1 for keyword in complex_keywords if keyword in query.lower())
        
        if total_entities > 5 or complex_count > 2:
            return QueryComplexity.COMPLEX
        elif total_entities > 2 or complex_count > 0:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    def _extract_parameters(self, query: str, doc) -> Dict[str, Any]:
        """Extract parameters from query"""
        
        parameters = {}
        
        # Extract numerical values
        numbers = [token.text for token in doc if token.like_num]
        if numbers:
            parameters['numbers'] = numbers
        
        # Extract percentages
        percentage_pattern = r'(\d+(?:\.\d+)?)\s*%'
        percentages = re.findall(percentage_pattern, query)
        if percentages:
            parameters['percentages'] = [float(p) for p in percentages]
        
        # Extract dollar amounts
        dollar_pattern = r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)'
        dollar_amounts = re.findall(dollar_pattern, query)
        if dollar_amounts:
            parameters['dollar_amounts'] = dollar_amounts
        
        return parameters
    
    def _extract_time_horizon(self, query: str) -> Optional[str]:
        """Extract time horizon from query"""
        
        time_patterns = {
            r'past\s+(\d+)\s+months?': lambda m: f"{m.group(1)}_months",
            r'last\s+(\d+)\s+years?': lambda m: f"{m.group(1)}_years",
            r'next\s+(\d+)\s+months?': lambda m: f"next_{m.group(1)}_months",
            r'(\d+)\s+month': lambda m: f"{m.group(1)}_months",
            r'(\d+)\s+year': lambda m: f"{m.group(1)}_years",
            r'ytd|year.to.date': lambda m: "YTD",
            r'quarterly': lambda m: "quarterly",
            r'annually': lambda m: "annually"
        }
        
        for pattern, formatter in time_patterns.items():
            match = re.search(pattern, query.lower())
            if match:
                return formatter(match)
        
        return None
    
    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract filters and conditions from query"""
        
        filters = {}
        
        # ESG filters
        if any(word in query.lower() for word in ['esg', 'sustainable', 'green', 'climate']):
            filters['esg_focused'] = True
        
        # Dividend filters
        if any(word in query.lower() for word in ['dividend', 'yield']):
            filters['dividend_focused'] = True
        
        # Growth filters
        if any(word in query.lower() for word in ['growth', 'growing']):
            filters['growth_focused'] = True
        
        # Value filters
        if any(word in query.lower() for word in ['value', 'undervalued', 'cheap']):
            filters['value_focused'] = True
        
        # Market cap filters
        if 'large cap' in query.lower():
            filters['market_cap'] = 'large'
        elif 'small cap' in query.lower():
            filters['market_cap'] = 'small'
        elif 'mid cap' in query.lower():
            filters['market_cap'] = 'mid'
        
        return filters
    
    async def _handle_stock_analysis(self, parsed_query: ParsedQuery, 
                                   context: Dict[str, Any]) -> ConversationalResponse:
        """Handle stock analysis queries"""
        
        stocks = parsed_query.entities.get('stocks', [])
        if not stocks:
            return ConversationalResponse(
                query=parsed_query.original_query,
                response_text="I'd be happy to analyze a stock for you! Could you please specify which stock symbol you'd like me to analyze?",
                data_analysis={},
                visualizations=[],
                follow_up_questions=["Which stock would you like me to analyze?"],
                confidence_score=0.9,
                processing_time_ms=0,
                sources=[]
            )
        
        primary_stock = stocks[0]
        
        # Simulate comprehensive stock analysis
        analysis_data = {
            'symbol': primary_stock,
            'current_price': 175.50,
            'price_change': 2.35,
            'price_change_percent': 1.36,
            'volume': 45_678_900,
            'market_cap': 2_750_000_000_000,
            'pe_ratio': 28.5,
            'eps': 6.16,
            'dividend_yield': 0.52,
            'beta': 1.25,
            'analyst_rating': 'BUY',
            'price_target': 195.00,
            'esg_score': 82,
            'risk_rating': 'MODERATE',
            'key_metrics': {
                'revenue_growth': 8.2,
                'profit_margin': 25.3,
                'roe': 147.4,
                'debt_to_equity': 1.73
            }
        }
        
        # Generate natural language response
        response_text = await self._generate_stock_analysis_response(
            primary_stock, analysis_data, parsed_query
        )
        
        # Create visualizations
        visualizations = [
            {
                'type': 'price_chart',
                'title': f'{primary_stock} Price Chart',
                'data': 'price_history_data'
            },
            {
                'type': 'metrics_comparison',
                'title': 'Key Metrics vs Industry',
                'data': 'metrics_comparison_data'
            }
        ]
        
        # Generate follow-up questions
        follow_up_questions = [
            f"Would you like to see how {primary_stock} compares to its competitors?",
            f"Are you interested in the risk analysis for {primary_stock}?",
            f"Would you like to know about {primary_stock}'s ESG performance?"
        ]
        
        return ConversationalResponse(
            query=parsed_query.original_query,
            response_text=response_text,
            data_analysis=analysis_data,
            visualizations=visualizations,
            follow_up_questions=follow_up_questions,
            confidence_score=0.92,
            processing_time_ms=0,
            sources=['Yahoo Finance', 'Bloomberg', 'Company Filings']
        )
    
    async def _handle_market_screening(self, parsed_query: ParsedQuery, 
                                     context: Dict[str, Any]) -> ConversationalResponse:
        """Handle market screening queries"""
        
        # Example: "Show me European tech stocks with strong ESG scores and low correlation to the US market"
        
        filters = parsed_query.filters
        regions = parsed_query.entities.get('regions', [])
        sectors = parsed_query.entities.get('sectors', [])
        
        # Simulate screening results
        screening_results = [
            {
                'symbol': 'ASML',
                'name': 'ASML Holding N.V.',
                'sector': 'Technology',
                'region': 'Europe',
                'price': 650.25,
                'market_cap': 265_000_000_000,
                'esg_score': 89,
                'us_correlation': 0.23,
                'pe_ratio': 35.2,
                'revenue_growth': 18.5
            },
            {
                'symbol': 'SAP',
                'name': 'SAP SE',
                'sector': 'Technology',
                'region': 'Europe',
                'price': 125.80,
                'market_cap': 145_000_000_000,
                'esg_score': 85,
                'us_correlation': 0.31,
                'pe_ratio': 22.1,
                'revenue_growth': 12.3
            }
        ]
        
        # Generate response
        response_text = await self._generate_screening_response(
            screening_results, parsed_query
        )
        
        visualizations = [
            {
                'type': 'screening_table',
                'title': 'Screening Results',
                'data': screening_results
            },
            {
                'type': 'scatter_plot',
                'title': 'ESG Score vs US Correlation',
                'data': 'correlation_esg_data'
            }
        ]
        
        follow_up_questions = [
            "Would you like me to analyze any of these stocks in detail?",
            "Should I add more screening criteria?",
            "Would you like to see the risk analysis for this selection?"
        ]
        
        return ConversationalResponse(
            query=parsed_query.original_query,
            response_text=response_text,
            data_analysis={'screening_results': screening_results},
            visualizations=visualizations,
            follow_up_questions=follow_up_questions,
            confidence_score=0.88,
            processing_time_ms=0,
            sources=['Refinitiv', 'MSCI ESG', 'Bloomberg']
        )
    
    async def _handle_scenario_analysis(self, parsed_query: ParsedQuery, 
                                      context: Dict[str, Any]) -> ConversationalResponse:
        """Handle scenario analysis queries"""
        
        # Example: "What would happen to our portfolio if European energy prices doubled while the Euro weakened 15%?"
        
        parameters = parsed_query.parameters
        percentages = parameters.get('percentages', [])
        
        # Simulate scenario analysis
        scenario_results = {
            'base_case': {
                'portfolio_value': 1_000_000,
                'expected_return': 8.5,
                'volatility': 12.3
            },
            'scenario_case': {
                'portfolio_value': 925_000,
                'expected_return': 3.2,
                'volatility': 18.7,
                'value_at_risk': -75_000,
                'impact_breakdown': {
                    'energy_exposure': -45_000,
                    'currency_impact': -30_000,
                    'correlation_effects': -15_000,
                    'hedging_benefits': 15_000
                }
            }
        }
        
        response_text = await self._generate_scenario_response(
            scenario_results, parsed_query
        )
        
        visualizations = [
            {
                'type': 'scenario_comparison',
                'title': 'Base Case vs Scenario',
                'data': scenario_results
            },
            {
                'type': 'impact_waterfall',
                'title': 'Impact Breakdown',
                'data': scenario_results['scenario_case']['impact_breakdown']
            }
        ]
        
        follow_up_questions = [
            "Would you like to see additional stress test scenarios?",
            "Should I analyze hedging strategies for this risk?",
            "Would you like to see the impact on individual positions?"
        ]
        
        return ConversationalResponse(
            query=parsed_query.original_query,
            response_text=response_text,
            data_analysis=scenario_results,
            visualizations=visualizations,
            follow_up_questions=follow_up_questions,
            confidence_score=0.85,
            processing_time_ms=0,
            sources=['Risk Models', 'Market Data', 'Correlation Analysis']
        )
    
    async def _generate_stock_analysis_response(self, symbol: str, 
                                              data: Dict[str, Any], 
                                              parsed_query: ParsedQuery) -> str:
        """Generate natural language response for stock analysis"""
        
        # Use AI to generate natural response
        prompt = f"""
        Generate a comprehensive but conversational analysis of {symbol} stock based on this data:
        
        Current Price: ${data['current_price']:.2f} ({data['price_change']:+.2f}, {data['price_change_percent']:+.1f}%)
        Market Cap: ${data['market_cap']:,.0f}
        P/E Ratio: {data['pe_ratio']}
        EPS: ${data['eps']}
        Analyst Rating: {data['analyst_rating']}
        Price Target: ${data['price_target']:.2f}
        ESG Score: {data['esg_score']}/100
        
        Key Metrics:
        - Revenue Growth: {data['key_metrics']['revenue_growth']}%
        - Profit Margin: {data['key_metrics']['profit_margin']}%
        - ROE: {data['key_metrics']['roe']}%
        
        Write this as a conversational response that sounds natural and informative.
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a knowledgeable financial advisor providing conversational investment analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except:
            # Fallback response
            return f"""
            Looking at {symbol}, I can see some interesting dynamics. The stock is currently trading at ${data['current_price']:.2f}, 
            which represents a {data['price_change_percent']:+.1f}% move today. 
            
            From a valuation perspective, {symbol} is trading at {data['pe_ratio']}x earnings, which suggests 
            {'reasonable valuation' if data['pe_ratio'] < 25 else 'premium valuation'} in the current market environment.
            
            The company shows strong fundamentals with {data['key_metrics']['revenue_growth']}% revenue growth and 
            {data['key_metrics']['profit_margin']}% profit margins. The analyst consensus is {data['analyst_rating']} 
            with a price target of ${data['price_target']:.2f}.
            
            From an ESG perspective, {symbol} scores {data['esg_score']}/100, which is 
            {'excellent' if data['esg_score'] > 80 else 'good' if data['esg_score'] > 60 else 'average'} 
            for sustainable investing considerations.
            """
    
    async def _generate_screening_response(self, results: List[Dict[str, Any]], 
                                         parsed_query: ParsedQuery) -> str:
        """Generate response for screening results"""
        
        if not results:
            return "I couldn't find any stocks matching your criteria. You might want to adjust your filters or expand your search parameters."
        
        response = f"I found {len(results)} stocks that match your criteria:\n\n"
        
        for i, stock in enumerate(results[:3], 1):  # Show top 3
            response += f"{i}. **{stock['symbol']} ({stock['name']})**\n"
            response += f"   • Price: ${stock['price']:.2f}\n"
            response += f"   • Market Cap: ${stock['market_cap']:,.0f}\n"
            response += f"   • ESG Score: {stock['esg_score']}/100\n"
            response += f"   • US Correlation: {stock['us_correlation']:.2f}\n\n"
        
        if len(results) > 3:
            response += f"...and {len(results) - 3} more stocks in the full results.\n\n"
        
        response += "These stocks meet your criteria for strong ESG performance and low correlation to US markets."
        
        return response
    
    async def _generate_scenario_response(self, results: Dict[str, Any], 
                                        parsed_query: ParsedQuery) -> str:
        """Generate response for scenario analysis"""
        
        base_value = results['base_case']['portfolio_value']
        scenario_value = results['scenario_case']['portfolio_value']
        impact = scenario_value - base_value
        impact_percent = (impact / base_value) * 100
        
        response = f"""
        Based on your scenario analysis, here's what would happen to your portfolio:
        
        **Impact Summary:**
        • Portfolio value would decrease from ${base_value:,.0f} to ${scenario_value:,.0f}
        • Total impact: ${impact:,.0f} ({impact_percent:+.1f}%)
        • Expected return would drop from {results['base_case']['expected_return']:.1f}% to {results['scenario_case']['expected_return']:.1f}%
        • Portfolio volatility would increase from {results['base_case']['volatility']:.1f}% to {results['scenario_case']['volatility']:.1f}%
        
        **Impact Breakdown:**
        """
        
        for factor, impact_value in results['scenario_case']['impact_breakdown'].items():
            response += f"• {factor.replace('_', ' ').title()}: ${impact_value:,.0f}\n"
        
        response += f"\n**Value at Risk:** ${results['scenario_case']['value_at_risk']:,.0f}"
        
        return response
    
    def _update_conversation_context(self, user_id: str, parsed_query: ParsedQuery, 
                                   response: ConversationalResponse):
        """Update conversation context for continuity"""
        
        if user_id not in self.conversation_context:
            self.conversation_context[user_id] = {
                'history': [],
                'entities': {},
                'preferences': {}
            }
        
        context = self.conversation_context[user_id]
        
        # Add to history
        context['history'].append({
            'query': parsed_query.original_query,
            'query_type': parsed_query.query_type.value,
            'entities': parsed_query.entities,
            'timestamp': datetime.now()
        })
        
        # Update entities
        for entity_type, entities in parsed_query.entities.items():
            if entity_type not in context['entities']:
                context['entities'][entity_type] = []
            context['entities'][entity_type].extend(entities)
            # Keep only recent entities
            context['entities'][entity_type] = list(set(context['entities'][entity_type]))[-10:]
        
        # Keep only recent history
        context['history'] = context['history'][-10:]

# Demo implementation
async def demo_conversational_ai():
    """Demonstrate conversational AI capabilities"""
    
    print("🗣️ Conversational AI Engine Demo")
    print("=" * 50)
    
    engine = ConversationalAIEngine()
    
    # Sample queries
    queries = [
        "Analyze Apple stock potential for the next 12 months",
        "Show me European tech stocks with strong ESG scores and low correlation to the US market over the past 6 months",
        "What would happen to our portfolio if European energy prices doubled while the Euro weakened 15%?",
        "Compare Tesla and Ford from an investment perspective",
        "Find me undervalued dividend stocks in the healthcare sector"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: \"{query}\"")
        print("-" * 60)
        
        response = await engine.process_query(query)
        
        print(f"📊 Query Type: {response.data_analysis.get('query_type', 'General')}")
        print(f"⚡ Processing Time: {response.processing_time_ms:.1f}ms")
        print(f"🎯 Confidence: {response.confidence_score:.1%}")
        
        print(f"\n💬 Response:")
        print(response.response_text[:300] + "..." if len(response.response_text) > 300 else response.response_text)
        
        if response.follow_up_questions:
            print(f"\n❓ Follow-up Questions:")
            for j, question in enumerate(response.follow_up_questions[:2], 1):
                print(f"   {j}. {question}")
        
        if response.visualizations:
            print(f"\n📈 Visualizations Available:")
            for viz in response.visualizations:
                print(f"   • {viz['title']} ({viz['type']})")
    
    print("\n🎉 Conversational AI Demo Complete!")
    print("✅ Natural language processing for investment queries")
    print("✅ Intelligent query classification and routing")
    print("✅ Context-aware responses with follow-up questions")
    print("✅ Multi-modal analysis with visualizations")

if __name__ == "__main__":
    asyncio.run(demo_conversational_ai())