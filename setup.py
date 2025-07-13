#!/usr/bin/env python3
"""
🚀 GameTheorySimAI Setup Script
Sets up the environment for running free AI prisoner's dilemma simulations
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command with error handling"""
    print(f"📦 {description}...")
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
    print("🎮 GameTheorySimAI Setup")
    print("=" * 50)
    print("Setting up your FREE AI prisoner's dilemma environment...")
    print()
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Please upgrade Python.")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install basic requirements
    print("\n📦 Installing basic requirements...")
    basic_success = run_command("pip install -r requirements.txt", "Installing requirements")
    
    if not basic_success:
        print("❌ Failed to install requirements. Please check your internet connection.")
        sys.exit(1)
    
    # Try to install PyTorch with appropriate backend
    print("\n🔥 Setting up PyTorch...")
    
    # Check if CUDA is available
    cuda_available = run_command("nvidia-smi", "Checking CUDA availability")
    
    if cuda_available:
        print("🚀 CUDA detected! Installing PyTorch with GPU support...")
        torch_success = run_command(
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "Installing PyTorch with CUDA"
        )
    else:
        print("💻 No CUDA detected. Installing CPU-only PyTorch...")
        torch_success = run_command(
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
            "Installing PyTorch (CPU)"
        )
    
    if not torch_success:
        print("⚠️  PyTorch installation failed. You can still use rule-based agents.")
    
    # Test installations
    print("\n🧪 Testing installations...")
    
    # Test basic imports
    try:
        import streamlit
        import pandas
        import numpy
        import matplotlib
        print("✅ Basic packages working")
    except ImportError as e:
        print(f"❌ Basic package import failed: {e}")
    
    # Test PyTorch
    try:
        import torch
        print(f"✅ PyTorch working (version: {torch.__version__})")
        if torch.cuda.is_available():
            print(f"🚀 CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("💻 CPU mode (no CUDA)")
    except ImportError:
        print("⚠️  PyTorch not available - Free AI features will be limited")
    
    # Test Transformers
    try:
        import transformers
        print(f"✅ Transformers working (version: {transformers.__version__})")
    except ImportError:
        print("⚠️  Transformers not available - Free AI features will be limited")
    
    # Create demo files
    print("\n📁 Creating demo files...")
    
    # Create a simple test script
    test_script = '''#!/usr/bin/env python3
"""Test script for GameTheorySimAI"""

def test_rule_based():
    """Test rule-based agents"""
    try:
        from RuleBasedAgents import run_rule_based_simulation
        print("🤖 Testing rule-based agents...")
        df = run_rule_based_simulation("tit_for_tat", "always_cooperate", 10)
        print(f"✅ Rule-based test passed! Final scores: {df['A_Cumulative'].iloc[-1]}, {df['B_Cumulative'].iloc[-1]}")
        return True
    except Exception as e:
        print(f"❌ Rule-based test failed: {e}")
        return False

def test_free_ai():
    """Test free AI models"""
    try:
        import torch
        from transformers import pipeline
        print("🆓 Testing free AI models...")
        
        # Try a simple text generation
        generator = pipeline("text-generation", model="gpt2", max_length=50)
        result = generator("The prisoner's dilemma is")
        print("✅ Free AI test passed!")
        return True
    except Exception as e:
        print(f"⚠️  Free AI test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 GameTheorySimAI Test Suite")
    print("=" * 40)
    
    success = 0
    total = 2
    
    if test_rule_based():
        success += 1
    
    if test_free_ai():
        success += 1
    
    print(f"\\n📊 Results: {success}/{total} tests passed")
    if success == total:
        print("🎉 All systems ready!")
    else:
        print("⚠️  Some features may be limited")
'''
    
    with open("test_setup.py", "w") as f:
        f.write(test_script)
    
    print("✅ Created test_setup.py")
    
    # Print usage instructions
    print("\n🚀 Setup Complete!")
    print("=" * 50)
    print("📋 Available Applications:")
    print()
    print("1. 🆓 Free AI Simulation:")
    print("   streamlit run FreeAIPrisonersDilemma.py")
    print()
    print("2. 🤖 Rule-Based Agents:")
    print("   streamlit run RuleBasedStreamlit.py")
    print()
    print("3. 📊 Original OpenAI Version:")
    print("   streamlit run VanilaPrisonersDilemma.py")
    print()
    print("4. 📓 Jupyter Notebook:")
    print("   jupyter notebook IterrativePrisonersDilemma.ipynb")
    print()
    print("5. 🧪 Test Everything:")
    print("   python test_setup.py")
    print()
    print("💡 Pro Tips:")
    print("- Start with rule-based agents (zero cost)")
    print("- Try free AI models for more realistic behavior")
    print("- Use Jupyter notebook for advanced experiments")
    print("- Check test_setup.py if you have issues")
    print()
    print("🆓 Remember: Rule-based agents are COMPLETELY FREE!")
    print("📚 Check COMPREHENSIVE_GUIDE.md for full documentation")

if __name__ == "__main__":
    main()
