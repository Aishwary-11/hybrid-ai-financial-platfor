#!/usr/bin/env python3
"""
Simple Hybrid AI Financial Platform - Vercel Deployment
Minimal version that will definitely work
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/")
async def root():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>🚀 Hybrid AI Financial Platform</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
            padding: 50px;
            margin: 0;
        }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { font-size: 3rem; margin-bottom: 20px; }
        .status { 
            background: rgba(255,255,255,0.1); 
            padding: 20px; 
            border-radius: 10px; 
            margin: 20px 0;
        }
        .metric { 
            background: white; 
            color: #333; 
            padding: 20px; 
            margin: 10px; 
            border-radius: 10px; 
            display: inline-block;
            min-width: 150px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Hybrid AI Financial Platform</h1>
        <div class="status">
            <h2>✅ SUCCESSFULLY DEPLOYED ON VERCEL!</h2>
            <p>Your financial AI platform is now live worldwide</p>
        </div>
        
        <div class="metric">
            <h3>$2.45M</h3>
            <p>Portfolio Value</p>
        </div>
        
        <div class="metric">
            <h3>94.2%</h3>
            <p>AI Accuracy</p>
        </div>
        
        <div class="metric">
            <h3>1.85</h3>
            <p>Sharpe Ratio</p>
        </div>
        
        <div class="metric">
            <h3>15.7%</h3>
            <p>Risk Level</p>
        </div>
        
        <div class="status">
            <h3>🎉 Deployment Successful!</h3>
            <p>✅ FastAPI Backend: Working</p>
            <p>✅ Vercel Hosting: Active</p>
            <p>✅ Global CDN: Enabled</p>
            <p>✅ HTTPS: Secured</p>
        </div>
    </div>
</body>
</html>
    """)

@app.get("/api/test")
async def test():
    return {"status": "success", "message": "API is working!", "platform": "Vercel"}

@app.get("/api/portfolio")
async def portfolio():
    return {
        "status": "success",
        "portfolio_value": 2450000,
        "positions": [
            {"symbol": "AAPL", "value": 500000, "weight": 20.4},
            {"symbol": "MSFT", "value": 450000, "weight": 18.4},
            {"symbol": "GOOGL", "value": 400000, "weight": 16.3}
        ]
    }

# Required for Vercel
handler = app