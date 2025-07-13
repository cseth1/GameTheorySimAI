#!/usr/bin/env python3
"""
Quick Start Script for Game Theory AI Lab
Choose your preferred interface and experience level
"""

import os
import sys
import subprocess

def print_banner():
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    🎮 Game Theory AI Lab                      ║
    ║                   Quick Start Launcher                       ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("📦 Install them with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All required packages are installed!")
    return True

def main():
    print_banner()
    
    # Check requirements
    if not check_requirements():
        return
    
    print("\n🎯 Choose your Game Theory experience:\n")
    
    options = {
        "1": {
            "name": "⚪ Clean UI (Recommended)",
            "file": "CleanUI_PrisonersDilemma.py",
            "description": "Black & white interface, easy to read with colorful charts"
        },
        "2": {
            "name": "🚀 Modern UI (Dark Theme)",
            "file": "ModernUI_PrisonersDilemma.py",
            "description": "Clean, modern interface with enhanced AI agents"
        },
        "3": {
            "name": "✨ Premium UI (Advanced)",
            "file": "PremiumUI_PrisonersDilemma.py", 
            "description": "Glassmorphism design with real-time analytics and emotional AI"
        },
        "4": {
            "name": "🎲 Simple AI (Zero Setup)",
            "file": "SimpleAIPrisonersDilemma.py",
            "description": "Quick start with rule-based intelligent agents"
        },
        "5": {
            "name": "📊 Original Version",
            "file": "VanilaPrisonersDilemma.py",
            "description": "Classic version with OpenAI GPT-4 (requires API key)"
        }
    }
    
    # Display options
    for key, option in options.items():
        print(f"{key}. {option['name']}")
        print(f"   {option['description']}")
        print()
    
    # Get user choice
    while True:
        choice = input("👉 Enter your choice (1-5): ").strip()
        
        if choice in options:
            selected_option = options[choice]
            file_path = selected_option["file"]
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"❌ File {file_path} not found!")
                print("📝 Make sure all application files are in the current directory.")
                return
            
            print(f"\n🎮 Starting {selected_option['name']}...")
            print(f"📂 Running: {file_path}")
            print("🌐 Open your browser to: http://localhost:8501")
            print("⏹️  Press Ctrl+C to stop the application")
            print("\n" + "="*60 + "\n")
            
            # Run the selected application
            try:
                subprocess.run([sys.executable, "-m", "streamlit", "run", file_path], 
                             check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ Error running application: {e}")
            except KeyboardInterrupt:
                print("\n👋 Application stopped by user.")
            
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, 4, or 5.")

if __name__ == "__main__":
    main()
