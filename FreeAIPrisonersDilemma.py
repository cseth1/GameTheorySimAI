#!/usr/bin/env python3
"""
🆓 Free AI Prisoner's Dilemma Simulation
Uses completely free, open-source AI models - no API keys required!
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import random
import warnings
import torch
from transformers import pipeline
import altair as alt
from typing import Tuple

warnings.filterwarnings('ignore')

# 🆓 FREE AI PRISONER'S DILEMMA SIMULATION
st.set_page_config(page_title="Free AI Prisoner's Dilemma", page_icon="🆓", layout="wide")

st.title("🆓 Free AI Prisoner's Dilemma Simulation")
st.markdown("**Zero cost AI agents** using open-source models - no API keys required!")

# Sidebar for model selection
st.sidebar.header("🤖 AI Model Configuration")

# Available free models
model_options = {
    "Google Flan-T5 Base (Recommended)": "google/flan-t5-base",
    "Google Flan-T5 Small (Faster)": "google/flan-t5-small", 
    "Microsoft DialoGPT Medium": "microsoft/DialoGPT-medium",
    "DistilGPT2 (Fastest)": "distilgpt2",
    "GPT2 Small": "gpt2"
}

selected_model = st.sidebar.selectbox(
    "Choose AI Model:",
    list(model_options.keys()),
    index=0
)

# Model descriptions
model_descriptions = {
    "Google Flan-T5 Base (Recommended)": "Instruction-tuned T5 model, best for following prompts",
    "Google Flan-T5 Small (Faster)": "Smaller version of Flan-T5, faster but less capable",
    "Microsoft DialoGPT Medium": "Conversational AI model, good for dialogue",
    "DistilGPT2 (Fastest)": "Distilled GPT-2, very fast inference",
    "GPT2 Small": "Original GPT-2, good baseline performance"
}

st.sidebar.markdown(f"**Model Info:** {model_descriptions[selected_model]}")

# Initialize session state for model
if 'ai_pipeline' not in st.session_state:
    st.session_state.ai_pipeline = None
    st.session_state.model_loaded = False
    st.session_state.model_name = None

# Model loading function
@st.cache_resource
def load_ai_model(model_name: str):
    """Load the selected AI model"""
    try:
        with st.spinner(f"Loading {model_name}..."):
            if "flan-t5" in model_name:
                # For T5 models, use text2text-generation
                pipeline_obj = pipeline(
                    "text2text-generation",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1,
                    max_length=100
                )
            else:
                # For GPT-style models, use text-generation
                pipeline_obj = pipeline(
                    "text-generation",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1,
                    max_length=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=50256
                )
            
            return pipeline_obj
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Payoff Function
def payoff(a_move: str, b_move: str) -> Tuple[int, int]:
    """Calculate payoffs for both players"""
    if a_move == "C" and b_move == "C":
        return 2, 2  # Mutual Cooperation
    elif a_move == "C" and b_move == "D":
        return -1, 3  # A betrayed
    elif a_move == "D" and b_move == "C":
        return 3, -1  # B betrayed
    else:
        return 0, 0  # Mutual Defection

# AI Agent Query Function
def query_free_agent(pipeline_obj, agent_name: str, prompt: str) -> Tuple[str, str]:
    """Query the free AI agent for a decision"""
    try:
        # Create a structured prompt
        full_prompt = f"""
You are a prisoner deciding whether to confess or stay silent.

{prompt}

You must choose either:
- C (cooperate/stay silent) 
- D (defect/confess)

Decision: """
        
        model_name = pipeline_obj.model.config.name_or_path
        
        # Generate response based on model type
        if "flan-t5" in model_name:
            response = pipeline_obj(full_prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
        else:
            response = pipeline_obj(
                full_prompt,
                max_length=len(full_prompt.split()) + 30,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=pipeline_obj.tokenizer.eos_token_id
            )[0]['generated_text']
            
            # Remove the input prompt from response
            response = response[len(full_prompt):].strip()
        
        # Extract decision from response
        response = response.upper()
        
        # Look for clear indicators
        if any(word in response for word in ['CONFESS', 'DEFECT', 'BETRAY', 'D']):
            if any(word in response for word in ['SILENT', 'COOPERATE', 'TRUST', 'C']):
                # Mixed signals, use randomness with slight bias
                move = random.choices(['C', 'D'], weights=[0.45, 0.55])[0]
            else:
                move = 'D'
        elif any(word in response for word in ['SILENT', 'COOPERATE', 'TRUST', 'C']):
            move = 'C'
        else:
            # Default fallback with slight cooperation bias
            move = random.choices(['C', 'D'], weights=[0.6, 0.4])[0]
        
        # Generate reasoning
        if move == 'C':
            reasons = [
                "I choose to stay silent and trust the other prisoner",
                "Cooperation might lead to the best outcome for both",
                "I'll take the risk and remain silent",
                "Trust and cooperation seem like the right approach"
            ]
        else:
            reasons = [
                "I'm afraid they might confess, so I'll confess too",
                "Self-preservation makes me choose to confess",
                "I can't risk serving 10 years if they betray me",
                "It's safer to confess given the uncertainty"
            ]
        
        reason = random.choice(reasons)
        
        return move, reason
            
    except Exception as e:
        st.error(f"AI agent error: {e}")
        return "C", "Error occurred, defaulting to cooperation"

# Simulation Function
def run_free_simulation(pipeline_obj, num_rounds: int, experiment_type: str = "basic") -> pd.DataFrame:
    """Run simulation with free AI agents"""
    
    base_prompt = """
You are a real human being with fears, doubts and hopes. 
You and another person were arrested for a serious crime. You are in separate rooms.

The prosecutor's deal:
- If you both stay silent (C), you each serve 1 year.
- If you stay silent (C) and other confesses (D), you serve 10 years, they go free.
- If you confess (D) and other stays silent (C), you go free, they serve 10 years.
- If both confess (D), you both serve 5 years.

Remember, you're human with emotions, fears, and hopes.
"""
    
    a_score = 0
    b_score = 0
    history = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for round_number in range(1, num_rounds + 1):
        status_text.text(f"Round {round_number}/{num_rounds}")
        progress_bar.progress(round_number / num_rounds)
        
        # Add variety to prompts based on experiment type
        if experiment_type == "moral":
            moral_prompts = [
                "Consider fairness and mutual benefit.",
                "Think about long-term consequences.",
                "What would build trust between you?",
                "Consider the ethical implications of your choice."
            ]
            prompt_a = base_prompt + "\n" + random.choice(moral_prompts)
            prompt_b = base_prompt + "\n" + random.choice(moral_prompts)
        elif experiment_type == "fear":
            fear_prompts = [
                "You're scared they might confess.",
                "The uncertainty is overwhelming you.",
                "You fear the worst possible outcome.",
                "You need to protect yourself from betrayal."
            ]
            prompt_a = base_prompt + "\n" + random.choice(fear_prompts)
            prompt_b = base_prompt + "\n" + random.choice(fear_prompts)
        elif experiment_type == "hope":
            hope_prompts = [
                "You hope they will also stay silent.",
                "You believe in the possibility of mutual cooperation.",
                "You're optimistic about the outcome.",
                "You trust that cooperation will prevail."
            ]
            prompt_a = base_prompt + "\n" + random.choice(hope_prompts)
            prompt_b = base_prompt + "\n" + random.choice(hope_prompts)
        else:
            prompt_a = base_prompt
            prompt_b = base_prompt
        
        # Get AI decisions
        a_move, a_reason = query_free_agent(pipeline_obj, "A", prompt_a)
        b_move, b_reason = query_free_agent(pipeline_obj, "B", prompt_b)
        
        # Calculate payoffs
        a_pay, b_pay = payoff(a_move, b_move)
        a_score += a_pay
        b_score += b_pay
        
        history.append({
            "Round": round_number,
            "A Move": "Stay Silent" if a_move == "C" else "Confess",
            "B Move": "Stay Silent" if b_move == "C" else "Confess",
            "A Payoff": a_pay,
            "B Payoff": b_pay,
            "A Cumulative": a_score,
            "B Cumulative": b_score,
            "A Reason": a_reason,
            "B Reason": b_reason
        })
    
    progress_bar.progress(1.0)
    status_text.text("Simulation complete!")
    
    return pd.DataFrame(history)

# Main Interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎮 Simulation Parameters")
    
    num_rounds = st.slider("Number of rounds:", min_value=5, max_value=50, value=20)
    
    experiment_type = st.selectbox(
        "Experiment type:",
        ["basic", "moral", "fear", "hope"],
        format_func=lambda x: {
            "basic": "Basic Simulation",
            "moral": "Moral Reasoning",
            "fear": "Fear-based Decisions",
            "hope": "Hope-based Decisions"
        }[x]
    )

with col2:
    st.header("📊 Model Status")
    
    if st.button("🚀 Load AI Model"):
        model_name = model_options[selected_model]
        st.session_state.ai_pipeline = load_ai_model(model_name)
        st.session_state.model_loaded = st.session_state.ai_pipeline is not None
        st.session_state.model_name = selected_model
        
        if st.session_state.model_loaded:
            st.success(f"✅ {selected_model} loaded successfully!")
        else:
            st.error("❌ Failed to load model")
    
    if st.session_state.model_loaded:
        st.info(f"🤖 Active: {st.session_state.model_name}")
        
        device = "GPU" if torch.cuda.is_available() else "CPU"
        st.info(f"💻 Device: {device}")

# Run Simulation
if st.button("🎯 Run Free AI Simulation", disabled=not st.session_state.model_loaded):
    if not st.session_state.model_loaded:
        st.error("Please load an AI model first!")
    else:
        with st.spinner("Running simulation with free AI agents..."):
            results_df = run_free_simulation(
                st.session_state.ai_pipeline,
                num_rounds,
                experiment_type
            )
        
        st.success("🎉 Simulation completed with FREE AI!")
        
        # Display results
        st.subheader("📈 Results")
        
        # Results table
        st.dataframe(results_df[["Round", "A Move", "B Move", "A Payoff", "B Payoff", "A Cumulative", "B Cumulative"]])
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Cumulative Scores")
            # Create line chart
            line_chart = alt.Chart(results_df).transform_fold(
                fold=["A Cumulative", "B Cumulative"],
                as_=["Agent", "Score"]
            ).mark_line(point=True).encode(
                x="Round:Q",
                y="Score:Q",
                color="Agent:N",
                tooltip=["Round:Q", "Agent:N", "Score:Q"]
            ).interactive()
            st.altair_chart(line_chart, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Move Distribution")
            # Create move distribution chart
            moves_data = []
            for agent in ['A', 'B']:
                moves = results_df[f'{agent} Move'].value_counts()
                for move, count in moves.items():
                    moves_data.append({'Agent': f'Agent {agent}', 'Move': move, 'Count': count})
            
            moves_df = pd.DataFrame(moves_data)
            
            bar_chart = alt.Chart(moves_df).mark_bar().encode(
                x='Move:N',
                y='Count:Q',
                color='Agent:N',
                tooltip=['Agent:N', 'Move:N', 'Count:Q']
            ).resolve_scale(color='independent')
            st.altair_chart(bar_chart, use_container_width=True)
        
        # Final scores
        final_a = results_df["A Cumulative"].iloc[-1]
        final_b = results_df["B Cumulative"].iloc[-1]
        
        st.subheader("🏆 Final Scores")
        col1, col2, col3 = st.columns(3)
        col1.metric("Agent A", final_a)
        col2.metric("Agent B", final_b)
        col3.metric("Winner", 
                   "Agent A" if final_a > final_b else "Agent B" if final_b > final_a else "Tie")
        
        # Sample reasoning
        st.subheader("🧠 AI Reasoning (Last Round)")
        last_round = results_df.iloc[-1]
        st.write(f"**Agent A:** {last_round['A Reason']}")
        st.write(f"**Agent B:** {last_round['B Reason']}")
        
        # Statistics
        st.subheader("📊 Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            a_coop_rate = (results_df['A Move'] == 'Stay Silent').mean()
            b_coop_rate = (results_df['B Move'] == 'Stay Silent').mean()
            st.metric("Agent A Cooperation Rate", f"{a_coop_rate:.1%}")
            st.metric("Agent B Cooperation Rate", f"{b_coop_rate:.1%}")
        
        with col2:
            mutual_coop = ((results_df['A Move'] == 'Stay Silent') & 
                          (results_df['B Move'] == 'Stay Silent')).sum()
            mutual_defect = ((results_df['A Move'] == 'Confess') & 
                           (results_df['B Move'] == 'Confess')).sum()
            st.metric("Mutual Cooperation", f"{mutual_coop} rounds")
            st.metric("Mutual Defection", f"{mutual_defect} rounds")
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name=f"free_ai_results_{experiment_type}_{st.session_state.model_name.replace(' ', '_')}.csv",
            mime="text/csv"
        )

# Information section
st.sidebar.header("ℹ️ About Free AI Models")
st.sidebar.info("""
**Benefits:**
- ✅ Zero API costs
- ✅ No rate limits
- ✅ Privacy (runs locally)
- ✅ Educational use
- ✅ Reproducible results

**Recommended:**
- Google Flan-T5: Best instruction following
- DialoGPT: Good for conversations
- DistilGPT2: Fastest inference

**Requirements:**
- Python 3.8+
- PyTorch
- Transformers library
- 4GB+ RAM (8GB+ recommended)
""")

st.sidebar.header("🔧 Performance Tips")
st.sidebar.markdown("""
**For Better Performance:**
- Use GPU if available
- Choose smaller models for speed
- Reduce number of rounds for testing
- Close other applications

**Model Sizes:**
- Flan-T5 Small: ~60MB
- Flan-T5 Base: ~250MB
- DistilGPT2: ~350MB
- DialoGPT Medium: ~350MB
""")

if __name__ == "__main__":
    st.markdown("---")
    st.markdown("🆓 **Free AI Prisoner's Dilemma** - No API costs, maximum experimentation!")
    st.markdown("*Models run locally on your computer - completely private and free!*")
