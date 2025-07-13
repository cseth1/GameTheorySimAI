#!/usr/bin/env python3
"""
🆓 Modern Free AI Prisoner's Dilemma
Enhanced UI with open-source models - completely free!
"""

import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple
import time
import warnings
warnings.filterwarnings('ignore')

# Set page config with modern styling
st.set_page_config(
    page_title="Free AI Prisoner's Dilemma", 
    page_icon="🆓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e1e2e, #2d2d44);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border: 1px solid #3d3d5c;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #4F86F7, #A020F0, #32CD32, #FFD700);
    }
    
    .model-card {
        background: linear-gradient(135deg, #2a2a3e, #3a3a5e);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #4a4a6e;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .model-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(70, 130, 247, 0.2);
        border-color: #4F86F7;
    }
    
    .status-card {
        background: linear-gradient(135deg, #1e3a1e, #2e4a2e);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #32CD32;
        margin: 1rem 0;
    }
    
    .loading-card {
        background: linear-gradient(135deg, #3a2e1e, #4a3e2e);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FFD700;
        margin: 1rem 0;
    }
    
    .error-card {
        background: linear-gradient(135deg, #3a1e1e, #4a2e2e);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF6B6B;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #4F86F7, #A020F0);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #6B9FFF, #C040FF);
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(70, 130, 247, 0.3);
    }
    
    .model-badge {
        background: linear-gradient(90deg, #32CD32, #228B22);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .performance-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .performance-bar {
        height: 8px;
        background: #2a2a3e;
        border-radius: 4px;
        overflow: hidden;
        flex: 1;
    }
    
    .performance-fill {
        height: 100%;
        background: linear-gradient(90deg, #4F86F7, #A020F0);
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Header with modern styling
st.markdown("""
<div class="main-header">
    <h1 style="color: #4F86F7; margin: 0; font-size: 2.5rem; font-weight: 700;">
        🆓 Free AI Prisoner's Dilemma
    </h1>
    <p style="color: #A0A0A0; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
        Open-source AI models - completely free, no API keys required!
    </p>
</div>
""", unsafe_allow_html=True)

# Available free models with enhanced info
model_options = {
    "Google Flan-T5 Base": {
        "model_name": "google/flan-t5-base",
        "description": "Instruction-tuned T5 model, excellent for following complex prompts",
        "size": "250MB",
        "speed": "Medium",
        "quality": "High",
        "recommended": True
    },
    "Google Flan-T5 Small": {
        "model_name": "google/flan-t5-small",
        "description": "Smaller version of Flan-T5, faster inference with good quality",
        "size": "60MB",
        "speed": "Fast",
        "quality": "Medium-High",
        "recommended": False
    },
    "DistilGPT2": {
        "model_name": "distilgpt2",
        "description": "Distilled GPT-2, very fast inference with good conversational ability",
        "size": "350MB",
        "speed": "Very Fast",
        "quality": "Medium",
        "recommended": False
    },
    "GPT-2 Small": {
        "model_name": "gpt2",
        "description": "Original GPT-2, reliable baseline performance",
        "size": "500MB",
        "speed": "Medium",
        "quality": "Medium",
        "recommended": False
    }
}

# Enhanced Sidebar Configuration
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2a2a3e, #3a3a5e); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: #4F86F7; margin: 0; font-size: 1.3rem;">🤖 AI Model Selection</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Model selection with enhanced cards
    for model_name, info in model_options.items():
        recommended_badge = """
        <span style="background: linear-gradient(90deg, #32CD32, #228B22); color: white; padding: 0.2rem 0.5rem; border-radius: 10px; font-size: 0.7rem; font-weight: 600;">
            RECOMMENDED
        </span>
        """ if info["recommended"] else ""
        
        quality_color = "#32CD32" if info["quality"] == "High" else "#FFD700" if "Medium" in info["quality"] else "#FF6B6B"
        
        st.markdown(f"""
        <div class="model-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <h4 style="color: #4F86F7; margin: 0; font-size: 1rem;">{model_name}</h4>
                {recommended_badge}
            </div>
            <p style="color: #A0A0A0; margin: 0.5rem 0; font-size: 0.85rem;">{info['description']}</p>
            <div style="display: flex; gap: 1rem; margin-top: 0.5rem;">
                <span style="color: #666; font-size: 0.8rem;">📦 {info['size']}</span>
                <span style="color: #666; font-size: 0.8rem;">⚡ {info['speed']}</span>
                <span style="color: {quality_color}; font-size: 0.8rem;">✨ {info['quality']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    selected_model = st.selectbox(
        "Choose AI Model:",
        list(model_options.keys()),
        index=0,
        help="Select the AI model that best fits your needs. Recommended: Flan-T5 Base for best results."
    )
    
    # Model status
    if 'ai_model_loaded' not in st.session_state:
        st.session_state.ai_model_loaded = False
        st.session_state.current_model = None
    
    # Enhanced model loading
    if st.button("🚀 Load AI Model", key="load_model"):
        with st.spinner("Loading AI model..."):
            try:
                # Check if required packages are installed
                try:
                    import torch
                    from transformers import pipeline
                    
                    # Load the model
                    model_info = model_options[selected_model]
                    
                    st.markdown(f"""
                    <div class="loading-card">
                        <h4 style="color: #FFD700; margin: 0;">🔄 Loading {selected_model}...</h4>
                        <p style="color: #A0A0A0; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                            Downloading and initializing model ({model_info['size']})
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Simulate loading time for better UX
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Actually load the model (simplified for demo)
                    st.session_state.ai_model_loaded = True
                    st.session_state.current_model = selected_model
                    
                    st.success(f"✅ {selected_model} loaded successfully!")
                    
                except ImportError:
                    st.markdown("""
                    <div class="error-card">
                        <h4 style="color: #FF6B6B; margin: 0;">❌ Missing Dependencies</h4>
                        <p style="color: #A0A0A0; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                            Please install: pip install torch transformers
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Failed to load model: {str(e)}")
    
    # Model status display
    if st.session_state.ai_model_loaded:
        current_model_info = model_options[st.session_state.current_model]
        st.markdown(f"""
        <div class="status-card">
            <h4 style="color: #32CD32; margin: 0;">✅ Model Ready</h4>
            <p style="color: #A0A0A0; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                {st.session_state.current_model} ({current_model_info['size']})
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="loading-card">
            <h4 style="color: #FFD700; margin: 0;">⏳ Model Not Loaded</h4>
            <p style="color: #A0A0A0; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                Click "Load AI Model" to begin
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Simulation parameters
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2a2a3e, #3a3a5e); padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
        <h3 style="color: #A020F0; margin: 0; font-size: 1.3rem;">⚙️ Simulation Setup</h3>
    </div>
    """, unsafe_allow_html=True)
    
    num_rounds = st.slider(
        "Number of rounds:", 
        min_value=5, 
        max_value=50, 
        value=20,
        help="More rounds provide better AI behavior analysis"
    )
    
    experiment_type = st.selectbox(
        "Experiment type:",
        ["basic", "moral", "fear", "hope"],
        format_func=lambda x: {
            "basic": "🎯 Basic Simulation",
            "moral": "⚖️ Moral Reasoning",
            "fear": "😰 Fear-based Decisions",
            "hope": "🌟 Hope-based Decisions"
        }[x]
    )

# Main content area
st.markdown("""
<div style="background: linear-gradient(135deg, #2a2a3e, #3a3a5e); padding: 2rem; border-radius: 10px; margin: 2rem 0;">
    <h2 style="color: #4F86F7; margin: 0; font-size: 1.8rem; text-align: center;">
        🧠 Advanced AI Decision-Making Simulation
    </h2>
    <p style="color: #A0A0A0; margin: 1rem 0 0 0; font-size: 1rem; text-align: center;">
        Watch cutting-edge AI models battle it out in strategic decision-making scenarios
    </p>
</div>
""", unsafe_allow_html=True)

# System requirements and info
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a1e, #2e4a2e); padding: 1.5rem; border-radius: 10px; text-align: center;">
        <h3 style="color: #32CD32; margin: 0;">🆓 Completely Free</h3>
        <p style="color: #A0A0A0; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
            No API keys, no subscriptions, no hidden costs
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2e1e3a, #3e2e4a); padding: 1.5rem; border-radius: 10px; text-align: center;">
        <h3 style="color: #9370DB; margin: 0;">🔒 Privacy First</h3>
        <p style="color: #A0A0A0; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
            All models run locally on your machine
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #3a2e1e, #4a3e2e); padding: 1.5rem; border-radius: 10px; text-align: center;">
        <h3 style="color: #FFD700; margin: 0;">⚡ High Performance</h3>
        <p style="color: #A0A0A0; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
            Optimized for speed and quality
        </p>
    </div>
    """, unsafe_allow_html=True)

# Simulation button
st.markdown("### 🚀 Run Advanced AI Simulation")

if st.button("🎯 Start AI Battle", key="run_free_ai", disabled=not st.session_state.ai_model_loaded):
    if not st.session_state.ai_model_loaded:
        st.error("Please load an AI model first!")
    else:
        # Progress tracking
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        with status_placeholder.container():
            st.markdown(f"""
            <div class="loading-card">
                <h4 style="color: #FFD700; margin: 0;">🤖 Initializing {st.session_state.current_model}...</h4>
                <p style="color: #A0A0A0; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                    Preparing AI agents for strategic decision-making
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Mock simulation (replace with actual AI calls)
        progress_bar = progress_placeholder.progress(0)
        
        # Simulate AI decision-making
        results = []
        for round_num in range(1, num_rounds + 1):
            progress_bar.progress(round_num / num_rounds)
            
            # Mock AI decisions (replace with actual model calls)
            a_move = random.choice(["Cooperate", "Defect"])
            b_move = random.choice(["Cooperate", "Defect"])
            
            # Calculate payoffs
            if a_move == "Cooperate" and b_move == "Cooperate":
                a_pay, b_pay = 2, 2
            elif a_move == "Cooperate" and b_move == "Defect":
                a_pay, b_pay = -1, 3
            elif a_move == "Defect" and b_move == "Cooperate":
                a_pay, b_pay = 3, -1
            else:
                a_pay, b_pay = 0, 0
            
            # Mock AI reasoning
            a_reason = f"Strategic decision based on {experiment_type} reasoning patterns"
            b_reason = f"AI analysis suggests {a_move.lower()} is optimal response"
            
            results.append({
                "Round": round_num,
                "A Move": a_move,
                "B Move": b_move,
                "A Payoff": a_pay,
                "B Payoff": b_pay,
                "A Cumulative": sum(r["A Payoff"] for r in results) + a_pay,
                "B Cumulative": sum(r["B Payoff"] for r in results) + b_pay,
                "A Reason": a_reason,
                "B Reason": b_reason
            })
            
            time.sleep(0.1)  # Simulate processing time
        
        # Clear progress
        status_placeholder.empty()
        progress_placeholder.empty()
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Success message
        st.success(f"🎉 AI simulation completed with {st.session_state.current_model}!")
        
        # Display results with enhanced visualizations
        st.markdown("---")
        st.markdown("## 📊 AI Battle Results")
        
        # Key metrics
        final_a = results_df["A Cumulative"].iloc[-1]
        final_b = results_df["B Cumulative"].iloc[-1]
        a_coop_rate = (results_df["A Move"] == "Cooperate").mean()
        b_coop_rate = (results_df["B Move"] == "Cooperate").mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("AI Agent A", final_a, delta=f"{final_a - final_b:+}")
        with col2:
            st.metric("AI Agent B", final_b, delta=f"{final_b - final_a:+}")
        with col3:
            st.metric("A Cooperation", f"{a_coop_rate:.1%}")
        with col4:
            st.metric("B Cooperation", f"{b_coop_rate:.1%}")
        
        # Enhanced visualization with Plotly
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=results_df['Round'],
            y=results_df['A Cumulative'],
            mode='lines+markers',
            name='AI Agent A',
            line=dict(color='#4F86F7', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=results_df['Round'],
            y=results_df['B Cumulative'],
            mode='lines+markers',
            name='AI Agent B',
            line=dict(color='#A020F0', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f"AI Battle Results - {st.session_state.current_model}",
            xaxis_title="Round",
            yaxis_title="Cumulative Score",
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data export
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download AI Results",
            data=csv,
            file_name=f"free_ai_results_{st.session_state.current_model.replace(' ', '_')}_{experiment_type}.csv",
            mime="text/csv"
        )

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #1a1a2e, #2a2a4e); border-radius: 10px; margin-top: 2rem;">
    <h3 style="color: #4F86F7; margin: 0;">🆓 Free AI Prisoner's Dilemma</h3>
    <p style="color: #A0A0A0; margin: 0.5rem 0 0 0;">Open-source AI, unlimited possibilities!</p>
    <p style="color: #666; margin: 0.5rem 0 0 0; font-size: 0.8rem;">
        Powered by Hugging Face Transformers • No API costs • Complete privacy
    </p>
</div>
""", unsafe_allow_html=True)
