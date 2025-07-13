#!/bin/bash
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
