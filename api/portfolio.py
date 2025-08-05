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
            "portfolio": {
                "total_value": 2450000,
                "positions": [
                    {"symbol": "AAPL", "value": 500000, "weight": 20.4, "return": 12.5, "ai_score": 0.89},
                    {"symbol": "MSFT", "value": 450000, "weight": 18.4, "return": 15.2, "ai_score": 0.92},
                    {"symbol": "GOOGL", "value": 400000, "weight": 16.3, "return": 8.7, "ai_score": 0.85},
                    {"symbol": "TSLA", "value": 350000, "weight": 14.3, "return": -5.2, "ai_score": 0.73}
                ],
                "performance": {
                    "ytd_return": 12.3,
                    "sharpe_ratio": 1.85,
                    "max_drawdown": 8.7,
                    "volatility": 15.7
                }
            },
            "ai_analysis": {
                "recommendation": "REBALANCE",
                "confidence": 0.87,
                "key_insights": [
                    "Tech sector overweight detected",
                    "Consider defensive allocation", 
                    "Strong momentum in AI stocks"
                ]
            }
        }
        
        self.wfile.write(json.dumps(response).encode())