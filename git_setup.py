#!/usr/bin/env python3
"""
Git Setup Script for Hybrid AI Financial Platform
This script will help you upload all files to GitHub
"""

import subprocess
import os
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Success!")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} - Failed!")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} - Exception: {e}")
        return False

def check_git_installed():
    """Check if Git is installed"""
    return run_command("git --version", "Checking Git installation")

def initialize_git_repo():
    """Initialize Git repository"""
    if not os.path.exists('.git'):
        return run_command("git init", "Initializing Git repository")
    else:
        print("✅ Git repository already initialized")
        return True

def add_all_files():
    """Add all files to Git"""
    return run_command("git add .", "Adding all files to Git")

def commit_files():
    """Commit files to Git"""
    return run_command('git commit -m "Initial commit: Hybrid AI Financial Platform"', "Committing files")

def main():
    """Main Git setup function"""
    print("🚀 HYBRID AI FINANCIAL PLATFORM - GIT SETUP")
    print("=" * 60)
    print("This script will prepare your project for GitHub upload")
    print("=" * 60)
    
    # Check if Git is installed
    if not check_git_installed():
        print("\n❌ Git is not installed!")
        print("Please install Git first:")
        print("Windows: Download from https://git-scm.com/download/win")
        print("Mac: brew install git")
        print("Linux: sudo apt install git")
        return
    
    # Initialize repository
    if not initialize_git_repo():
        print("❌ Failed to initialize Git repository")
        return
    
    # Configure Git (if not already configured)
    print("\n🔧 Configuring Git...")
    run_command('git config user.name "Hybrid AI Platform"', "Setting Git username")
    run_command('git config user.email "admin@hybrid-ai-platform.com"', "Setting Git email")
    
    # Add all files
    if not add_all_files():
        print("❌ Failed to add files to Git")
        return
    
    # Commit files
    if not commit_files():
        print("❌ Failed to commit files")
        return
    
    print("\n" + "=" * 60)
    print("🎉 GIT SETUP COMPLETE!")
    print("=" * 60)
    print("Your project is now ready for GitHub!")
    print("\nNext steps:")
    print("1. Go to GitHub.com")
    print("2. Create a new repository: 'hybrid-ai-financial-platform'")
    print("3. Copy the repository URL")
    print("4. Run these commands:")
    print("   git remote add origin <your-repo-url>")
    print("   git branch -M main")
    print("   git push -u origin main")
    print("\nOr use the GitHub Desktop app for easier upload!")
    print("=" * 60)

if __name__ == "__main__":
    main()