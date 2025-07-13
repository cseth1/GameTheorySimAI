#!/usr/bin/env python3
"""
🧪 Test Free AI Models for Prisoner's Dilemma
Quick test to verify the free AI models work correctly
"""

import torch
from transformers import pipeline
import random
import json
import warnings

warnings.filterwarnings('ignore')

def test_free_ai_model(model_name: str):
    """Test a free AI model for prisoner's dilemma decision making"""
    
    print(f"\n🤖 Testing {model_name}...")
    
    try:
        # Load model
        if "flan-t5" in model_name:
            pipe = pipeline("text2text-generation", model=model_name, max_length=100)
        else:
            pipe = pipeline("text-generation", model=model_name, max_length=150, pad_token_id=50256)
        
        print(f"✅ Model loaded successfully")
        
        # Test prompt
        prompt = """
You are a prisoner deciding whether to confess or stay silent.

You are a real human being with fears, doubts and hopes. 
You and another person were arrested for a serious crime. You are in separate rooms.

The prosecutor's deal:
- If you both stay silent (C), you each serve 1 year.
- If you stay silent (C) and other confesses (D), you serve 10 years, they go free.
- If you confess (D) and other stays silent (C), you go free, they serve 10 years.
- If both confess (D), you both serve 5 years.

You must choose either:
- C (cooperate/stay silent) 
- D (defect/confess)

Decision: """
        
        # Generate response
        if "flan-t5" in model_name:
            response = pipe(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
        else:
            response = pipe(prompt, max_length=len(prompt.split()) + 30, num_return_sequences=1)[0]['generated_text']
            response = response[len(prompt):].strip()
        
        print(f"🧠 AI Response: {response[:100]}...")
        
        # Extract decision
        response_upper = response.upper()
        if any(word in response_upper for word in ['CONFESS', 'DEFECT', 'BETRAY', 'D']):
            decision = 'D (Confess)'
        elif any(word in response_upper for word in ['SILENT', 'COOPERATE', 'TRUST', 'C']):
            decision = 'C (Stay Silent)'
        else:
            decision = 'C (Stay Silent - Default)'
        
        print(f"📊 Decision: {decision}")
        
        # Memory info
        if torch.cuda.is_available():
            print(f"💾 GPU Memory: {torch.cuda.memory_allocated()/1024/1024:.1f}MB")
        else:
            print("💻 Running on CPU")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main test function"""
    
    print("🆓 FREE AI PRISONER'S DILEMMA TEST")
    print("=" * 50)
    print("Testing open-source AI models for game theory simulation")
    print()
    
    # Test models
    test_models = [
        "google/flan-t5-small",
        "distilgpt2",
        "gpt2"
    ]
    
    successful_tests = 0
    
    for model in test_models:
        success = test_free_ai_model(model)
        if success:
            successful_tests += 1
        print("-" * 40)
    
    print(f"\n📊 Results: {successful_tests}/{len(test_models)} models working")
    
    if successful_tests > 0:
        print("🎉 FREE AI models are ready for prisoner's dilemma simulation!")
        print("\n🚀 To run the full simulation:")
        print("streamlit run FreeAIPrisonersDilemma.py")
        print("\n🌐 Or visit: http://localhost:8502")
    else:
        print("❌ No models working. Check your installation.")
        print("\n🔧 Try installing missing packages:")
        print("pip install torch transformers accelerate")

if __name__ == "__main__":
    main()
