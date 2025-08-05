from http.server import BaseHTTPRequestHandler
import json
from datetime import datetime

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "insights": [
                {
                    "title": "Tech Sector Rotation Opportunity",
                    "description": "AI models detect rotation from growth to value tech stocks",
                    "confidence": 0.89,
                    "impact": "High",
                    "timeframe": "2-4 weeks"
                },
                {
                    "title": "Market Volatility Alert", 
                    "description": "Increased volatility expected due to earnings season",
                    "confidence": 0.76,
                    "impact": "Medium",
                    "timeframe": "1-2 weeks"
                },
                {
                    "title": "ESG Investment Momentum",
                    "description": "Strong institutional flows into ESG-focused investments",
                    "confidence": 0.92,
                    "impact": "High",
                    "timeframe": "3-6 months"
                }
            ],
            "model_performance": {
                "earnings_model": 0.942,
                "sentiment_model": 0.917,
                "risk_model": 0.961
            }
        }
        
        self.wfile.write(json.dumps(response).encode())