#!/bin/bash

# Hybrid AI Architecture - Quick Start Deployment Script
# This script provides the fastest way to get the Hybrid AI Architecture running locally

set -e

echo "🚀 Hybrid AI Architecture - Quick Start Deployment"
echo "=================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker Desktop first."
        exit 1
    fi
    print_status "Docker found"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.9+ first."
        exit 1
    fi
    print_status "Python 3 found"
    
    # Check Git
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed. Please install Git first."
        exit 1
    fi
    print_status "Git found"
    
    # Check available memory
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
        if [ "$MEMORY_GB" -lt 8 ]; then
            print_warning "Less than 8GB RAM detected. Performance may be limited."
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        MEMORY_GB=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
        if [ "$MEMORY_GB" -lt 8 ]; then
            print_warning "Less than 8GB RAM detected. Performance may be limited."
        fi
    fi
    
    print_status "Prerequisites check completed"
}

# Set up Python environment
setup_python_env() {
    print_info "Setting up Python environment..."
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_status "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
    
    # Upgrade pip
    pip install --upgrade pip > /dev/null 2>&1
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt > /dev/null 2>&1
        print_status "Python dependencies installed"
    else
        print_warning "requirements.txt not found, creating basic requirements..."
        cat > requirements.txt << EOF
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
redis==5.0.1
openai==1.3.7
anthropic==0.7.7
google-generativeai==0.3.2
prometheus-client==0.19.0
asyncio==3.4.3
aiohttp==3.9.1
pandas==2.1.4
numpy==1.25.2
scikit-learn==1.3.2
matplotlib==3.8.2
plotly==5.17.0
streamlit==1.28.2
EOF
        pip install -r requirements.txt > /dev/null 2>&1
        print_status "Basic requirements installed"
    fi
}

# Configure environment variables
setup_environment() {
    print_info "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# Hybrid AI Architecture Configuration

# Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# API Keys (Replace with your actual keys)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Database Configuration
DATABASE_URL=postgresql://admin:password@localhost:5432/hybrid_ai
REDIS_URL=redis://localhost:6379

# Application Settings
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=4

# Model Configuration
MODEL_CACHE_ENABLED=true
MODEL_CACHE_TTL=3600
MOCK_MODELS=true

# Monitoring
METRICS_ENABLED=true
PROMETHEUS_PORT=9090

# Security
SECRET_KEY=your-secret-key-change-in-production
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8000"]
EOF
        print_status "Environment file created (.env)"
        print_warning "Please edit .env file with your actual API keys before starting the application"
    else
        print_status "Environment file already exists"
    fi
}

# Start infrastructure services
start_infrastructure() {
    print_info "Starting infrastructure services..."
    
    # Create docker-compose.yml if it doesn't exist
    if [ ! -f "docker-compose.yml" ]; then
        cat > docker-compose.yml << EOF
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: hybrid_ai
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U admin -d hybrid_ai"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
EOF
        print_status "Docker Compose configuration created"
    fi
    
    # Create monitoring configuration
    mkdir -p monitoring
    if [ ! -f "monitoring/prometheus.yml" ]; then
        cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'hybrid-ai'
    static_configs:
      - targets: ['host.docker.internal:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
EOF
        print_status "Prometheus configuration created"
    fi
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be healthy
    print_info "Waiting for services to start..."
    sleep 10
    
    # Check service health
    if docker-compose ps | grep -q "Up (healthy)"; then
        print_status "Infrastructure services started successfully"
    else
        print_warning "Some services may still be starting. Check with: docker-compose ps"
    fi
}

# Initialize database
init_database() {
    print_info "Initializing database..."
    
    # Create database initialization script
    cat > init_db.py << EOF
import asyncio
import asyncpg
import os
from datetime import datetime

async def init_database():
    try:
        # Connect to database
        conn = await asyncpg.connect(
            host='localhost',
            port=5432,
            user='admin',
            password='password',
            database='hybrid_ai'
        )
        
        # Create tables
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                role VARCHAR(20) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                query TEXT NOT NULL,
                result JSONB NOT NULL,
                confidence FLOAT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(50) NOT NULL,
                accuracy FLOAT NOT NULL,
                latency_ms INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert sample data
        await conn.execute('''
            INSERT INTO users (username, email, role) 
            VALUES ('admin', 'admin@hybrid-ai.com', 'administrator')
            ON CONFLICT (username) DO NOTHING
        ''')
        
        await conn.close()
        print("✅ Database initialized successfully")
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")

if __name__ == "__main__":
    asyncio.run(init_database())
EOF
    
    # Run database initialization
    python init_db.py
    rm init_db.py
}

# Create main application file
create_main_app() {
    print_info "Creating main application..."
    
    if [ ! -f "main.py" ]; then
        cat > main.py << EOF
#!/usr/bin/env python3
"""
Hybrid AI Architecture - Main Application
Quick Start Version
"""

import asyncio
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import json
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Hybrid AI Architecture",
    description="BlackRock Aladdin-inspired AI platform for investment management",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class InvestmentQuery(BaseModel):
    query: str
    symbols: Optional[List[str]] = []
    time_horizon: Optional[str] = "12_months"
    risk_tolerance: Optional[str] = "moderate"

class AnalysisResult(BaseModel):
    analysis_id: str
    query: str
    recommendation: str
    confidence_score: float
    target_price: Optional[float] = None
    risk_rating: str
    key_insights: List[str]
    timestamp: str

# Mock AI responses for demo
MOCK_RESPONSES = {
    "AAPL": {
        "recommendation": "BUY",
        "confidence_score": 0.87,
        "target_price": 195.50,
        "risk_rating": "MODERATE",
        "key_insights": [
            "Strong iPhone 15 sales momentum",
            "Services revenue growing at 15% YoY",
            "AI integration driving innovation"
        ]
    },
    "TSLA": {
        "recommendation": "HOLD",
        "confidence_score": 0.72,
        "target_price": 245.00,
        "risk_rating": "HIGH",
        "key_insights": [
            "Cybertruck production ramping up",
            "Energy storage business expanding",
            "Regulatory challenges in China"
        ]
    }
}

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Hybrid AI Architecture</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; text-align: center; }
                .status { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .endpoint { background: #f8f9fa; padding: 10px; margin: 10px 0; border-left: 4px solid #007bff; }
                .success { color: #27ae60; font-weight: bold; }
                a { color: #3498db; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 Hybrid AI Architecture</h1>
                <div class="status">
                    <p class="success">✅ System is running successfully!</p>
                    <p>Welcome to the Hybrid AI Architecture platform - a BlackRock Aladdin-inspired AI system for investment management.</p>
                </div>
                
                <h2>Available Endpoints:</h2>
                <div class="endpoint">
                    <strong>API Documentation:</strong> <a href="/docs">/docs</a> - Interactive API documentation
                </div>
                <div class="endpoint">
                    <strong>Health Check:</strong> <a href="/health">/health</a> - System health status
                </div>
                <div class="endpoint">
                    <strong>Investment Analysis:</strong> <a href="/docs#/default/analyze_investment_analyze_investment_post">POST /analyze-investment</a> - AI-powered investment analysis
                </div>
                <div class="endpoint">
                    <strong>System Metrics:</strong> <a href="/metrics">/metrics</a> - Prometheus metrics
                </div>
                
                <h2>Monitoring Dashboards:</h2>
                <div class="endpoint">
                    <strong>Grafana:</strong> <a href="http://localhost:3000" target="_blank">http://localhost:3000</a> (admin/admin)
                </div>
                <div class="endpoint">
                    <strong>Prometheus:</strong> <a href="http://localhost:9090" target="_blank">http://localhost:9090</a>
                </div>
                
                <h2>Quick Test:</h2>
                <p>Try analyzing Apple stock:</p>
                <pre style="background: #f8f9fa; padding: 15px; border-radius: 5px;">
curl -X POST "http://localhost:8000/analyze-investment" \\
     -H "Content-Type: application/json" \\
     -d '{"query": "Analyze Apple stock potential", "symbols": ["AAPL"]}'
                </pre>
            </div>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.post("/analyze-investment", response_model=AnalysisResult)
async def analyze_investment(query: InvestmentQuery):
    """
    Analyze investment opportunities using hybrid AI models
    """
    # Extract symbol from query or use provided symbols
    symbol = None
    if query.symbols:
        symbol = query.symbols[0]
    else:
        # Simple symbol extraction
        for s in ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]:
            if s in query.query.upper():
                symbol = s
                break
    
    # Get mock response or default
    if symbol and symbol in MOCK_RESPONSES:
        response_data = MOCK_RESPONSES[symbol]
    else:
        response_data = {
            "recommendation": "HOLD",
            "confidence_score": 0.65,
            "target_price": None,
            "risk_rating": "MODERATE",
            "key_insights": [
                "Analysis based on general market conditions",
                "Recommend further research for specific insights",
                "Consider diversification strategies"
            ]
        }
    
    # Create analysis result
    result = AnalysisResult(
        analysis_id=f"ana_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        query=query.query,
        recommendation=response_data["recommendation"],
        confidence_score=response_data["confidence_score"],
        target_price=response_data.get("target_price"),
        risk_rating=response_data["risk_rating"],
        key_insights=response_data["key_insights"],
        timestamp=datetime.now().isoformat()
    )
    
    return result

@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint
    """
    metrics_data = f'''# HELP hybrid_ai_requests_total Total number of requests
# TYPE hybrid_ai_requests_total counter
hybrid_ai_requests_total 42

# HELP hybrid_ai_response_time_seconds Response time in seconds
# TYPE hybrid_ai_response_time_seconds histogram
hybrid_ai_response_time_seconds_bucket{{le="0.1"}} 10
hybrid_ai_response_time_seconds_bucket{{le="0.5"}} 25
hybrid_ai_response_time_seconds_bucket{{le="1.0"}} 35
hybrid_ai_response_time_seconds_bucket{{le="+Inf"}} 42
hybrid_ai_response_time_seconds_sum 15.2
hybrid_ai_response_time_seconds_count 42

# HELP hybrid_ai_model_accuracy Model accuracy score
# TYPE hybrid_ai_model_accuracy gauge
hybrid_ai_model_accuracy{{model="earnings_analyzer"}} 0.94
hybrid_ai_model_accuracy{{model="sentiment_analyzer"}} 0.91
hybrid_ai_model_accuracy{{model="risk_predictor"}} 0.96
'''
    return metrics_data

if __name__ == "__main__":
    print("🚀 Starting Hybrid AI Architecture...")
    print(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print(f"Debug mode: {os.getenv('DEBUG', 'false')}")
    print("Access the application at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("DEBUG", "false").lower() == "true",
        workers=1 if os.getenv("DEBUG", "false").lower() == "true" else int(os.getenv("WORKERS", 4))
    )
EOF
        print_status "Main application created"
    else
        print_status "Main application already exists"
    fi
}

# Start the application
start_application() {
    print_info "Starting the Hybrid AI Architecture application..."
    
    # Activate virtual environment
    source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
    
    print_status "Application is starting..."
    print_info "Access points:"
    echo "  🌐 Web Interface: http://localhost:8000"
    echo "  📚 API Docs: http://localhost:8000/docs"
    echo "  📊 Grafana: http://localhost:3000 (admin/admin)"
    echo "  📈 Prometheus: http://localhost:9090"
    echo ""
    print_info "Press Ctrl+C to stop the application"
    echo ""
    
    # Start the application
    python main.py
}

# Cleanup function
cleanup() {
    print_info "Cleaning up..."
    docker-compose down > /dev/null 2>&1 || true
    print_status "Cleanup completed"
}

# Main execution
main() {
    echo ""
    print_info "Starting Hybrid AI Architecture Quick Deployment..."
    echo ""
    
    # Trap cleanup on exit
    trap cleanup EXIT
    
    # Run deployment steps
    check_prerequisites
    setup_python_env
    setup_environment
    start_infrastructure
    init_database
    create_main_app
    
    echo ""
    print_status "Deployment completed successfully!"
    echo ""
    print_info "Next steps:"
    echo "1. Edit .env file with your actual API keys"
    echo "2. Run: source venv/bin/activate"
    echo "3. Run: python main.py"
    echo ""
    print_info "Or run this script with --start to launch immediately:"
    echo "./quick_start.sh --start"
    echo ""
    
    # Check if --start flag is provided
    if [[ "$1" == "--start" ]]; then
        start_application
    fi
}

# Handle command line arguments
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "Hybrid AI Architecture - Quick Start Deployment"
    echo ""
    echo "Usage:"
    echo "  ./quick_start.sh          Setup the environment"
    echo "  ./quick_start.sh --start  Setup and start the application"
    echo "  ./quick_start.sh --help   Show this help message"
    echo ""
    echo "This script will:"
    echo "  ✅ Check prerequisites (Docker, Python, Git)"
    echo "  ✅ Set up Python virtual environment"
    echo "  ✅ Configure environment variables"
    echo "  ✅ Start infrastructure services (PostgreSQL, Redis, Prometheus, Grafana)"
    echo "  ✅ Initialize database"
    echo "  ✅ Create main application"
    echo "  ✅ Start the Hybrid AI Architecture platform"
    echo ""
    exit 0
fi

# Run main function
main "$@"