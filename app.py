#!/usr/bin/env python3
"""
Hybrid AI Financial Platform - Streamlit Web Application
Professional UI for the financial AI system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
import json
from typing import Dict, List, Any
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Hybrid AI Financial Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-metric {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .warning-metric {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Mock data generators
def generate_portfolio_data():
    """Generate mock portfolio data"""
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']
    data = []
    
    for symbol in symbols:
        data.append({
            'Symbol': symbol,
            'Position': np.random.uniform(1000, 50000),
            'Current Price': np.random.uniform(100, 3000),
            'Day Change %': np.random.uniform(-5, 5),
            'Market Value': np.random.uniform(100000, 500000),
            'Weight %': np.random.uniform(5, 20),
            'AI Score': np.random.uniform(0.6, 0.95),
            'Risk Level': np.random.choice(['Low', 'Medium', 'High'])
        })
    
    return pd.DataFrame(data)

def generate_market_data():
    """Generate mock market data"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    prices = []
    base_price = 100
    
    for _ in dates:
        change = np.random.normal(0, 2)
        base_price += change
        prices.append(base_price)
    
    return pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })

def generate_ai_insights():
    """Generate mock AI insights"""
    insights = [
        {
            'type': 'Market Opportunity',
            'title': 'Tech Sector Rotation Detected',
            'description': 'AI models detect rotation from growth to value tech stocks',
            'confidence': 0.87,
            'impact': 'High',
            'timeframe': '2-4 weeks'
        },
        {
            'type': 'Risk Alert',
            'title': 'Increased Volatility Expected',
            'description': 'Alternative data suggests increased market volatility ahead',
            'confidence': 0.73,
            'impact': 'Medium',
            'timeframe': '1-2 weeks'
        },
        {
            'type': 'ESG Opportunity',
            'title': 'Clean Energy Momentum',
            'description': 'ESG models show strong momentum in renewable energy sector',
            'confidence': 0.91,
            'impact': 'High',
            'timeframe': '3-6 months'
        }
    ]
    return insights

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Hybrid AI Financial Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">BlackRock Aladdin-inspired AI platform for investment management</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Dashboard", "Portfolio Analysis", "AI Insights", "Risk Management", "Market Data", "Settings"]
    )
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Portfolio Analysis":
        show_portfolio_analysis()
    elif page == "AI Insights":
        show_ai_insights()
    elif page == "Risk Management":
        show_risk_management()
    elif page == "Market Data":
        show_market_data()
    elif page == "Settings":
        show_settings()

def show_dashboard():
    """Main dashboard view"""
    st.header("📊 Investment Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card success-metric">
            <h3>$2.4M</h3>
            <p>Total Portfolio Value</p>
            <small>+12.3% YTD</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>94.2%</h3>
            <p>AI Model Accuracy</p>
            <small>Last 30 days</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card warning-metric">
            <h3>15.7%</h3>
            <p>Portfolio Risk</p>
            <small>Medium Risk Level</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>1.85</h3>
            <p>Sharpe Ratio</p>
            <small>Risk-adjusted returns</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Portfolio Performance")
        market_data = generate_market_data()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=market_data['Date'],
            y=market_data['Price'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Asset Allocation")
        
        # Generate allocation data
        labels = ['Stocks', 'Bonds', 'Crypto', 'Alternatives', 'Cash']
        values = [65, 20, 8, 5, 2]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker_colors=colors
        )])
        
        fig.update_layout(
            title="Current Asset Allocation",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent AI insights
    st.subheader("🤖 Latest AI Insights")
    insights = generate_ai_insights()
    
    for insight in insights[:2]:  # Show top 2 insights
        with st.expander(f"{insight['type']}: {insight['title']}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{insight['confidence']:.0%}")
            with col2:
                st.metric("Impact", insight['impact'])
            with col3:
                st.metric("Timeframe", insight['timeframe'])
            
            st.write(insight['description'])

def show_portfolio_analysis():
    """Portfolio analysis view"""
    st.header("📈 Portfolio Analysis")
    
    # Generate portfolio data
    portfolio_df = generate_portfolio_data()
    
    # Portfolio overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Holdings Overview")
        
        # Format the dataframe for display
        display_df = portfolio_df.copy()
        display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"${x:.2f}")
        display_df['Market Value'] = display_df['Market Value'].apply(lambda x: f"${x:,.0f}")
        display_df['Day Change %'] = display_df['Day Change %'].apply(lambda x: f"{x:+.2f}%")
        display_df['Weight %'] = display_df['Weight %'].apply(lambda x: f"{x:.1f}%")
        display_df['AI Score'] = display_df['AI Score'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(display_df, use_container_width=True)
    
    with col2:
        st.subheader("Portfolio Metrics")
        
        total_value = portfolio_df['Market Value'].sum()
        top_holding = portfolio_df.loc[portfolio_df['Market Value'].idxmax(), 'Symbol']
        avg_ai_score = portfolio_df['AI Score'].mean()
        
        st.metric("Total Value", f"${total_value:,.0f}")
        st.metric("Top Holding", top_holding)
        st.metric("Avg AI Score", f"{avg_ai_score:.2f}")
        st.metric("Positions", len(portfolio_df))
    
    # AI Analysis
    st.subheader("🤖 AI Portfolio Analysis")
    
    if st.button("Run AI Analysis", type="primary"):
        with st.spinner("Analyzing portfolio with AI models..."):
            time.sleep(2)  # Simulate processing
            
            st.success("Analysis Complete!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="info-box">
                    <h4>🎯 Optimization Recommendations</h4>
                    <ul>
                        <li>Reduce TSLA position by 15% (high volatility)</li>
                        <li>Increase MSFT allocation by 8% (strong fundamentals)</li>
                        <li>Consider adding defensive positions</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="info-box">
                    <h4>⚠️ Risk Alerts</h4>
                    <ul>
                        <li>Portfolio concentration risk in tech sector</li>
                        <li>Correlation risk during market stress</li>
                        <li>Consider ESG diversification</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

def show_ai_insights():
    """AI insights view"""
    st.header("🤖 AI Insights & Recommendations")
    
    # AI model status
    st.subheader("Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Earnings Model", "94.2%", "2.1%")
    with col2:
        st.metric("Sentiment Model", "91.7%", "1.8%")
    with col3:
        st.metric("Risk Model", "96.1%", "0.9%")
    
    # Insights
    st.subheader("Current Insights")
    
    insights = generate_ai_insights()
    
    for i, insight in enumerate(insights):
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.write(f"**{insight['title']}**")
                st.write(insight['description'])
            
            with col2:
                confidence_color = "green" if insight['confidence'] > 0.8 else "orange" if insight['confidence'] > 0.6 else "red"
                st.markdown(f"<span style='color: {confidence_color}; font-weight: bold;'>{insight['confidence']:.0%}</span>", unsafe_allow_html=True)
            
            with col3:
                impact_color = "red" if insight['impact'] == 'High' else "orange" if insight['impact'] == 'Medium' else "green"
                st.markdown(f"<span style='color: {impact_color}; font-weight: bold;'>{insight['impact']}</span>", unsafe_allow_html=True)
            
            with col4:
                st.write(insight['timeframe'])
            
            st.divider()
    
    # Real-time analysis
    st.subheader("🔄 Real-Time Analysis")
    
    if st.button("Refresh AI Analysis", type="primary"):
        with st.spinner("Running real-time AI analysis..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            st.success("Real-time analysis complete!")
            st.balloons()

def show_risk_management():
    """Risk management view"""
    st.header("⚠️ Risk Management")
    
    # Risk metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Portfolio VaR", "5.2%", "-0.3%")
    with col2:
        st.metric("Beta", "1.15", "+0.05")
    with col3:
        st.metric("Max Drawdown", "8.7%", "+1.2%")
    with col4:
        st.metric("Correlation", "0.73", "-0.02")
    
    # Risk visualization
    st.subheader("Risk Analysis")
    
    # Generate risk data
    risk_categories = ['Market Risk', 'Credit Risk', 'Liquidity Risk', 'Operational Risk', 'Model Risk']
    risk_scores = [0.65, 0.23, 0.15, 0.08, 0.12]
    
    fig = go.Figure(data=[
        go.Bar(x=risk_categories, y=risk_scores, marker_color='rgba(255, 99, 132, 0.8)')
    ])
    
    fig.update_layout(
        title="Risk Breakdown by Category",
        xaxis_title="Risk Category",
        yaxis_title="Risk Score",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Stress testing
    st.subheader("🧪 Stress Testing")
    
    scenarios = ['Market Crash (-20%)', 'Interest Rate Spike (+200bp)', 'Currency Crisis', 'Sector Rotation']
    
    for scenario in scenarios:
        with st.expander(f"Scenario: {scenario}"):
            impact = np.random.uniform(-15, -5)
            recovery_time = np.random.randint(3, 12)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Portfolio Impact", f"{impact:.1f}%")
            with col2:
                st.metric("Recovery Time", f"{recovery_time} months")

def show_market_data():
    """Market data view"""
    st.header("📊 Market Data & Analytics")
    
    # Market overview
    st.subheader("Market Overview")
    
    # Generate market data
    indices = ['S&P 500', 'NASDAQ', 'DOW', 'Russell 2000']
    values = [4500, 14200, 35000, 2100]
    changes = [0.8, 1.2, 0.5, -0.3]
    
    cols = st.columns(len(indices))
    
    for i, (index, value, change) in enumerate(zip(indices, values, changes)):
        with cols[i]:
            st.metric(index, f"{value:,}", f"{change:+.1f}%")
    
    # Market chart
    st.subheader("Market Trends")
    
    market_data = generate_market_data()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=market_data['Date'],
        y=market_data['Price'],
        mode='lines',
        name='Market Index',
        line=dict(color='#2ca02c', width=2)
    ))
    
    fig.update_layout(
        title="Market Performance (YTD)",
        xaxis_title="Date",
        yaxis_title="Index Value",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sector performance
    st.subheader("Sector Performance")
    
    sectors = ['Technology', 'Healthcare', 'Financials', 'Energy', 'Consumer', 'Industrials']
    performance = [12.5, 8.3, 15.2, -2.1, 6.7, 9.8]
    
    fig = go.Figure(data=[
        go.Bar(x=sectors, y=performance, 
               marker_color=['green' if p > 0 else 'red' for p in performance])
    ])
    
    fig.update_layout(
        title="Sector Performance (YTD %)",
        xaxis_title="Sector",
        yaxis_title="Performance (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_settings():
    """Settings view"""
    st.header("⚙️ Settings & Configuration")
    
    # API Configuration
    st.subheader("API Configuration")
    
    with st.expander("Data Sources"):
        st.text_input("Bloomberg API Key", type="password", placeholder="Enter API key")
        st.text_input("Alpha Vantage API Key", type="password", placeholder="Enter API key")
        st.text_input("IEX Cloud API Key", type="password", placeholder="Enter API key")
    
    # Model Settings
    st.subheader("AI Model Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.slider("Risk Tolerance", 1, 10, 5)
        st.slider("Model Confidence Threshold", 0.5, 1.0, 0.8)
    
    with col2:
        st.selectbox("Default Time Horizon", ["1 Month", "3 Months", "6 Months", "1 Year"])
        st.selectbox("Rebalancing Frequency", ["Daily", "Weekly", "Monthly", "Quarterly"])
    
    # Notifications
    st.subheader("Notifications")
    
    st.checkbox("Email Alerts", value=True)
    st.checkbox("Risk Alerts", value=True)
    st.checkbox("Performance Reports", value=True)
    st.checkbox("AI Insights", value=True)
    
    if st.button("Save Settings", type="primary"):
        st.success("Settings saved successfully!")

# Footer
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🚀 Hybrid AI Financial Platform | Built with Streamlit | Powered by Advanced AI Models</p>
        <p>© 2024 - Professional Investment Management Technology</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    show_footer()