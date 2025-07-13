#!/usr/bin/env python3
"""
🎨 UI Enhancement Setup Script
Installs all dependencies for the modern interface
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command with error handling"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed: {str(e)}")
        return False

def main():
    """Main setup function"""
    print("🎨 MODERN UI ENHANCEMENT SETUP")
    print("=" * 50)
    print("Installing dependencies for the enhanced interface...")
    print()
    
    # Enhanced requirements
    enhanced_requirements = [
        "streamlit>=1.28.0",
        "plotly>=5.15.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0"
    ]
    
    print("📦 Installing enhanced UI dependencies...")
    for requirement in enhanced_requirements:
        success = run_command(f"pip install {requirement}", f"Installing {requirement}")
        if not success:
            print(f"⚠️  Warning: Failed to install {requirement}")
    
    # Optional AI dependencies
    print("\n🤖 Installing optional AI dependencies...")
    ai_requirements = [
        "torch",
        "transformers>=4.25.0",
        "accelerate",
        "scipy"
    ]
    
    for requirement in ai_requirements:
        success = run_command(f"pip install {requirement}", f"Installing {requirement}")
        if not success:
            print(f"⚠️  Warning: Failed to install {requirement} (optional)")
    
    print("\n🧪 Testing enhanced interface...")
    
    # Test imports
    try:
        import streamlit
        import plotly.graph_objects as go
        import plotly.express as px
        import pandas as pd
        import numpy as np
        print("✅ Enhanced UI dependencies working")
    except ImportError as e:
        print(f"❌ Enhanced UI test failed: {e}")
    
    # Create launch script
    launch_script = '''#!/bin/bash
echo "🚀 GameTheorySimAI - Modern UI Launcher"
echo "======================================"
echo ""
echo "Choose your experience:"
echo "1. 🤖 Simple AI (Zero dependencies)"
echo "2. 🆓 Free AI (Open-source models)"
echo "3. 🎯 Rule-based (Classic strategies)"
echo "4. 📊 Original (OpenAI GPT-4)"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "🤖 Launching Simple AI..."
        streamlit run SimpleAIPrisonersDilemma.py --server.port 8501
        ;;
    2)
        echo "🆓 Launching Free AI..."
        streamlit run FreeAIModern.py --server.port 8502
        ;;
    3)
        echo "🎯 Launching Rule-based..."
        streamlit run RuleBasedStreamlit.py --server.port 8503
        ;;
    4)
        echo "📊 Launching Original..."
        streamlit run VanilaPrisonersDilemma.py --server.port 8504
        ;;
    *)
        echo "❌ Invalid choice. Please run again."
        ;;
esac
'''
    
    with open("launch.sh", "w") as f:
        f.write(launch_script)
    
    # Make executable
    os.chmod("launch.sh", 0o755)
    
    print("\n🎉 Setup Complete!")
    print("=" * 50)
    print("📱 Enhanced Applications Available:")
    print()
    print("1. 🤖 Simple AI (Best for beginners):")
    print("   streamlit run SimpleAIPrisonersDilemma.py")
    print()
    print("2. 🆓 Free AI (Advanced open-source models):")
    print("   streamlit run FreeAIModern.py")
    print()
    print("3. 🎯 Rule-Based (Classic game theory):")
    print("   streamlit run RuleBasedStreamlit.py")
    print()
    print("4. 📊 Original (OpenAI GPT-4):")
    print("   streamlit run VanilaPrisonersDilemma.py")
    print()
    print("🚀 Quick Launch:")
    print("   ./launch.sh  (interactive menu)")
    print()
    print("🎨 Features:")
    print("   ✅ Modern dark theme")
    print("   ✅ Interactive visualizations")
    print("   ✅ Enhanced UX/UI")
    print("   ✅ Real-time progress tracking")
    print("   ✅ Advanced analytics")
    print("   ✅ Multiple export formats")
    print()
    print("🌟 Ready to explore AI decision-making with style!")

if __name__ == "__main__":
    main()
