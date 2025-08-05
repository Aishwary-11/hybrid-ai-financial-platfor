#!/usr/bin/env python3
"""
Vercel Deployment Script for Hybrid AI Financial Platform
"""

import subprocess
import sys
import os
import json

def check_vercel_cli():
    """Check if Vercel CLI is installed"""
    try:
        result = subprocess.run(['vercel', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Vercel CLI found: {result.stdout.strip()}")
            return True
        else:
            print("❌ Vercel CLI not found")
            return False
    except FileNotFoundError:
        print("❌ Vercel CLI not installed")
        return False

def install_vercel_cli():
    """Install Vercel CLI"""
    print("📦 Installing Vercel CLI...")
    try:
        subprocess.run(['npm', 'install', '-g', 'vercel'], check=True)
        print("✅ Vercel CLI installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install Vercel CLI")
        print("Please install Node.js first: https://nodejs.org/")
        return False
    except FileNotFoundError:
        print("❌ npm not found. Please install Node.js first: https://nodejs.org/")
        return False

def deploy_to_vercel():
    """Deploy to Vercel"""
    print("🚀 Deploying to Vercel...")
    
    try:
        # Deploy to production
        result = subprocess.run(['vercel', '--prod'], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Deployment successful!")
            print(f"🌐 Your app is live at: {result.stdout.strip()}")
            return result.stdout.strip()
        else:
            print("❌ Deployment failed")
            print(f"Error: {result.stderr}")
            return None
            
    except FileNotFoundError:
        print("❌ Vercel CLI not found")
        return None

def main():
    """Main deployment function"""
    print("🚀 Hybrid AI Financial Platform - Vercel Deployment")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('vercel.json'):
        print("❌ vercel.json not found. Make sure you're in the project directory.")
        return
    
    if not os.path.exists('api/index.py'):
        print("❌ api/index.py not found. Make sure the API file exists.")
        return
    
    print("✅ Project files found")
    
    # Check Vercel CLI
    if not check_vercel_cli():
        print("\n📦 Installing Vercel CLI...")
        if not install_vercel_cli():
            print("\n❌ Cannot proceed without Vercel CLI")
            print("Please install manually:")
            print("npm install -g vercel")
            return
    
    # Deploy
    print("\n🚀 Starting deployment...")
    url = deploy_to_vercel()
    
    if url:
        print("\n" + "=" * 60)
        print("🎉 DEPLOYMENT SUCCESSFUL!")
        print("=" * 60)
        print(f"🌐 Live URL: {url}")
        print("📊 Dashboard: Available at the URL above")
        print("🔗 APIs: /api/portfolio, /api/insights, /api/risk, /api/market")
        print("📚 Health Check: /health")
        print("\n✅ Your Hybrid AI Financial Platform is now live on Vercel!")
        print("🚀 Ready for production use and client demonstrations")
        
        # Save URL to file
        with open('deployment_url.txt', 'w') as f:
            f.write(url)
        print(f"💾 URL saved to deployment_url.txt")
        
    else:
        print("\n❌ Deployment failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged into Vercel: vercel login")
        print("2. Check your internet connection")
        print("3. Verify all files are present")

if __name__ == "__main__":
    main()