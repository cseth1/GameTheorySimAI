#!/usr/bin/env python3
"""
🆓 Simple Free AI Prisoner's Dilemma (No PyTorch Required)
Uses rule-based AI that simulates different reasoning patterns
"""

import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple
import time

# Set page config with custom styling
st.set_page_config(
    page_title="Simple Free AI Prisoner's Dilemma", 
    page_icon="🤖", 
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
    }
    
    .metric-card {
        background: linear-gradient(135deg, #2a2a3e, #3a3a5e);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #4a4a6e;
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(70, 130, 247, 0.2);
    }
    
    .agent-card {
        background: linear-gradient(135deg, #1a1a2e, #2a2a4e);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #4a4a6e;
        margin: 1rem 0;
        border-left: 4px solid #4F86F7;
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
    
    .sidebar .stSelectbox > div > div {
        background: linear-gradient(135deg, #2a2a3e, #3a3a5e);
        border: 1px solid #4a4a6e;
        border-radius: 8px;
    }
    
    .simulation-status {
        background: linear-gradient(135deg, #1e3a3a, #2e4a4a);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #32CD32;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #3a2e1e, #4a3e2e);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FFD700;
        margin: 1rem 0;
    }
    
    .error-box {
        background: linear-gradient(135deg, #3a1e1e, #4a2e2e);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF6B6B;
        margin: 1rem 0;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .personality-badge {
        background: linear-gradient(90deg, #4F86F7, #A020F0);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header with modern styling
st.markdown("""
<div class="main-header">
    <h1 style="color: #4F86F7; margin: 0; font-size: 2.5rem; font-weight: 700;">
        🤖 Simple Free AI Prisoner's Dilemma
    </h1>
    <p style="color: #A0A0A0; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
        Zero cost AI agents using rule-based reasoning patterns - no external dependencies!
    </p>
</div>
""", unsafe_allow_html=True)

# AI personality types
ai_personalities = {
    "rational": {
        "name": "Rational Agent",
        "description": "Makes logical decisions based on expected outcomes",
        "cooperation_bias": 0.3,
        "fear_factor": 0.7,
        "trust_factor": 0.4
    },
    "cooperative": {
        "name": "Cooperative Agent", 
        "description": "Tends to trust and cooperate with others",
        "cooperation_bias": 0.8,
        "fear_factor": 0.2,
        "trust_factor": 0.9
    },
    "suspicious": {
        "name": "Suspicious Agent",
        "description": "Distrusts others and tends to defect",
        "cooperation_bias": 0.2,
        "fear_factor": 0.9,
        "trust_factor": 0.1
    },
    "emotional": {
        "name": "Emotional Agent",
        "description": "Decisions influenced by emotions and uncertainty",
        "cooperation_bias": 0.5,
        "fear_factor": 0.6,
        "trust_factor": 0.5
    },
    "adaptive": {
        "name": "Adaptive Agent",
        "description": "Learns from opponent's behavior",
        "cooperation_bias": 0.5,
        "fear_factor": 0.4,
        "trust_factor": 0.6
    }
}

# Enhanced Sidebar Configuration
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2a2a3e, #3a3a5e); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: #4F86F7; margin: 0; font-size: 1.3rem;">🎯 Agent Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🤖 Agent A")
    agent_a_type = st.selectbox(
        "Personality Type:",
        list(ai_personalities.keys()),
        format_func=lambda x: ai_personalities[x]["name"],
        key="agent_a"
    )
    
    st.markdown(f"""
    <div class="agent-card">
        <span class="personality-badge">{ai_personalities[agent_a_type]['name']}</span>
        <p style="color: #A0A0A0; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
            {ai_personalities[agent_a_type]['description']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🤖 Agent B")
    agent_b_type = st.selectbox(
        "Personality Type:",
        list(ai_personalities.keys()),
        index=1,
        format_func=lambda x: ai_personalities[x]["name"],
        key="agent_b"
    )
    
    st.markdown(f"""
    <div class="agent-card">
        <span class="personality-badge">{ai_personalities[agent_b_type]['name']}</span>
        <p style="color: #A0A0A0; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
            {ai_personalities[agent_b_type]['description']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Simulation Parameters
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2a2a3e, #3a3a5e); padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
        <h3 style="color: #4F86F7; margin: 0; font-size: 1.3rem;">⚙️ Simulation Parameters</h3>
    </div>
    """, unsafe_allow_html=True)
    
    num_rounds = st.slider(
        "Number of rounds:", 
        min_value=5, 
        max_value=100, 
        value=30,
        help="More rounds provide better pattern analysis"
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
    
    # Status indicators
    st.markdown("### 📊 System Status")
    st.success("✅ AI Agents Ready")
    st.success("✅ Zero Dependencies")
    st.success("✅ Completely Free")

# Payoff function
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

# AI Agent class
class SimpleAIAgent:
    def __init__(self, personality_type: str, agent_name: str):
        self.name = agent_name
        self.personality = ai_personalities[personality_type]
        self.history = []
        self.opponent_history = []
        self.cooperation_rate = self.personality["cooperation_bias"]
        
    def decide(self, round_num: int, experiment_type: str = "basic") -> Tuple[str, str]:
        """Make a decision based on personality and history"""
        
        # Base cooperation probability
        coop_prob = self.personality["cooperation_bias"]
        
        # Adjust based on opponent history
        if self.opponent_history:
            recent_coop = self.opponent_history[-3:].count("C") / min(3, len(self.opponent_history))
            overall_coop = self.opponent_history.count("C") / len(self.opponent_history)
            
            # Adaptive agents learn from opponent
            if self.personality["name"] == "Adaptive Agent":
                coop_prob = 0.3 + 0.6 * overall_coop
            else:
                # Trust factor influences decision
                trust_adjustment = self.personality["trust_factor"] * recent_coop
                coop_prob = coop_prob * 0.7 + trust_adjustment * 0.3
        
        # Experiment type adjustments
        if experiment_type == "moral":
            moral_prompts = [
                "fairness", "mutual benefit", "long-term thinking", "ethical choice"
            ]
            moral_boost = random.choice([0.1, 0.2, 0.3])
            coop_prob += moral_boost
            reasoning_context = f"Considering {random.choice(moral_prompts)}"
            
        elif experiment_type == "fear":
            fear_prompts = [
                "worry about betrayal", "fear of worst outcome", "uncertainty", "self-protection"
            ]
            fear_penalty = self.personality["fear_factor"] * random.uniform(0.1, 0.3)
            coop_prob -= fear_penalty
            reasoning_context = f"Driven by {random.choice(fear_prompts)}"
            
        elif experiment_type == "hope":
            hope_prompts = [
                "optimism", "trust in cooperation", "hope for mutual benefit", "belief in others"
            ]
            hope_boost = random.uniform(0.1, 0.2)
            coop_prob += hope_boost
            reasoning_context = f"Motivated by {random.choice(hope_prompts)}"
        else:
            reasoning_context = "basic analysis"
        
        # Add some randomness
        coop_prob += random.uniform(-0.1, 0.1)
        coop_prob = max(0, min(1, coop_prob))  # Clamp between 0 and 1
        
        # Make decision
        if random.random() < coop_prob:
            move = "C"
            if experiment_type == "moral":
                reasons = [
                    "I believe cooperation is the right thing to do",
                    "Fairness suggests we should both stay silent",
                    "Mutual benefit is more important than individual gain",
                    "This is an ethical choice for both of us"
                ]
            elif experiment_type == "fear":
                reasons = [
                    "Despite my fears, I'll trust them to cooperate",
                    "I'm scared but I'll take the risk of staying silent",
                    "My anxiety makes me want to trust this time",
                    "I hope they won't betray me"
                ]
            elif experiment_type == "hope":
                reasons = [
                    "I'm optimistic we can both benefit from cooperation",
                    "I trust they will also choose to stay silent",
                    "Hope for mutual benefit guides my decision",
                    "I believe in the power of working together"
                ]
            else:
                reasons = [
                    "I choose to trust and cooperate",
                    "Staying silent seems like the better strategy",
                    "I'll take the risk of cooperation",
                    "Trust might lead to the best outcome"
                ]
        else:
            move = "D"
            if experiment_type == "moral":
                reasons = [
                    "I must protect myself despite moral concerns",
                    "Self-preservation overrides ethical considerations",
                    "I can't risk being exploited, even if it's unfair",
                    "Sometimes pragmatism must override ethics"
                ]
            elif experiment_type == "fear":
                reasons = [
                    "I'm too afraid they'll confess first",
                    "Fear of the worst outcome makes me confess",
                    "I can't risk 10 years in prison",
                    "My fear of betrayal forces this choice"
                ]
            elif experiment_type == "hope":
                reasons = [
                    "I hope this protects me from being exploited",
                    "I'm hopeful this is the safer choice",
                    "Despite my optimism, I must be practical",
                    "I hope they'll understand my caution"
                ]
            else:
                reasons = [
                    "I don't trust them to cooperate",
                    "Confessing seems safer given the stakes",
                    "I need to protect myself from betrayal",
                    "The risk of staying silent is too high"
                ]
        
        reason = random.choice(reasons)
        return move, reason
    
    def update_history(self, my_move: str, opponent_move: str):
        """Update agent's history"""
        self.history.append(my_move)
        self.opponent_history.append(opponent_move)

# Simulation function
def run_simple_ai_simulation(agent_a_type: str, agent_b_type: str, num_rounds: int, experiment_type: str = "basic") -> pd.DataFrame:
    """Run simulation with simple AI agents"""
    
    agent_a = SimpleAIAgent(agent_a_type, "A")
    agent_b = SimpleAIAgent(agent_b_type, "B")
    
    a_score = 0
    b_score = 0
    history = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for round_num in range(1, num_rounds + 1):
        status_text.text(f"Round {round_num}/{num_rounds}")
        progress_bar.progress(round_num / num_rounds)
        
        # Agents make decisions
        a_move, a_reason = agent_a.decide(round_num, experiment_type)
        b_move, b_reason = agent_b.decide(round_num, experiment_type)
        
        # Calculate payoffs
        a_pay, b_pay = payoff(a_move, b_move)
        a_score += a_pay
        b_score += b_pay
        
        # Update histories
        agent_a.update_history(a_move, b_move)
        agent_b.update_history(b_move, a_move)
        
        # Record round
        history.append({
            "Round": round_num,
            "A Move": "Stay Silent" if a_move == "C" else "Confess",
            "B Move": "Stay Silent" if b_move == "C" else "Confess",
            "A Payoff": a_pay,
            "B Payoff": b_pay,
            "A Cumulative": a_score,
            "B Cumulative": b_score,
            "A Reason": a_reason,
            "B Reason": b_reason
        })
        
        # Small delay for visual effect
        time.sleep(0.1)
    
    progress_bar.progress(1.0)
    status_text.text("Simulation complete!")
    
    return pd.DataFrame(history)

# Main Content Area
st.markdown("""
<div style="background: linear-gradient(135deg, #2a2a3e, #3a3a5e); padding: 2rem; border-radius: 10px; margin: 2rem 0;">
    <h2 style="color: #4F86F7; margin: 0; font-size: 1.8rem; text-align: center;">
        🎮 Ready to Simulate AI Decision-Making?
    </h2>
    <p style="color: #A0A0A0; margin: 1rem 0 0 0; font-size: 1rem; text-align: center;">
        Watch as AI agents with different personalities battle it out in the classic prisoner's dilemma!
    </p>
</div>
""", unsafe_allow_html=True)

# Payoff Matrix Display
st.markdown("### 💰 Payoff Matrix")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a1a2e, #2a2a4e); padding: 1.5rem; border-radius: 10px; border: 1px solid #4a4a6e;">
        <table style="width: 100%; color: white; text-align: center; font-size: 1.1rem;">
            <tr style="background: linear-gradient(90deg, #4F86F7, #A020F0); color: white;">
                <th style="padding: 1rem; border-radius: 5px;">Player A \\ Player B</th>
                <th style="padding: 1rem;">Cooperate (C)</th>
                <th style="padding: 1rem;">Defect (D)</th>
            </tr>
            <tr>
                <td style="padding: 1rem; font-weight: bold; background: rgba(79, 134, 247, 0.3);">Cooperate (C)</td>
                <td style="padding: 1rem; background: rgba(50, 205, 50, 0.3);">+2, +2</td>
                <td style="padding: 1rem; background: rgba(255, 107, 107, 0.3);">-1, +3</td>
            </tr>
            <tr>
                <td style="padding: 1rem; font-weight: bold; background: rgba(79, 134, 247, 0.3);">Defect (D)</td>
                <td style="padding: 1rem; background: rgba(255, 107, 107, 0.3);">+3, -1</td>
                <td style="padding: 1rem; background: rgba(128, 128, 128, 0.3);">0, 0</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

# Simulation Control
st.markdown("### 🚀 Run Simulation")

# Create enhanced button
if st.button("🎯 Start AI Battle", key="run_simulation"):
    # Create placeholders for dynamic updates
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    with status_placeholder.container():
        st.markdown("""
        <div class="simulation-status">
            <h4 style="color: #32CD32; margin: 0;">🤖 Initializing AI Agents...</h4>
            <p style="color: #A0A0A0; margin: 0.5rem 0 0 0;">Setting up personalities and decision-making systems</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Progress bar
    progress_bar = progress_placeholder.progress(0)
    
    # Run simulation with enhanced feedback
    results_df = run_simple_ai_simulation(
        agent_a_type,
        agent_b_type,
        num_rounds,
        experiment_type
    )
    
    # Clear status
    status_placeholder.empty()
    progress_placeholder.empty()
    
    # Success message
    st.success("🎉 Simulation completed successfully!")
    
    # Results Section
    st.markdown("---")
    st.markdown("## 📊 Simulation Results")
    
    # Key metrics in cards
    final_a = results_df["A Cumulative"].iloc[-1]
    final_b = results_df["B Cumulative"].iloc[-1]
    a_coop_rate = (results_df["A Move"] == "Cooperate").mean()
    b_coop_rate = (results_df["B Move"] == "Cooperate").mean()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #4F86F7; margin: 0; font-size: 1.2rem;">Agent A Score</h3>
            <p style="color: white; font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{final_a}</p>
            <p style="color: #A0A0A0; font-size: 0.9rem; margin: 0;">{ai_personalities[agent_a_type]['name']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #A020F0; margin: 0; font-size: 1.2rem;">Agent B Score</h3>
            <p style="color: white; font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{final_b}</p>
            <p style="color: #A0A0A0; font-size: 0.9rem; margin: 0;">{ai_personalities[agent_b_type]['name']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #32CD32; margin: 0; font-size: 1.2rem;">A Cooperation</h3>
            <p style="color: white; font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{a_coop_rate:.1%}</p>
            <p style="color: #A0A0A0; font-size: 0.9rem; margin: 0;">Cooperation Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #FFD700; margin: 0; font-size: 1.2rem;">B Cooperation</h3>
            <p style="color: white; font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{b_coop_rate:.1%}</p>
            <p style="color: #A0A0A0; font-size: 0.9rem; margin: 0;">Cooperation Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Winner announcement
    winner = "Agent A" if final_a > final_b else "Agent B" if final_b > final_a else "Tie"
    winner_color = "#4F86F7" if winner == "Agent A" else "#A020F0" if winner == "Agent B" else "#FFD700"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {winner_color}20, {winner_color}40); padding: 2rem; border-radius: 10px; margin: 2rem 0; text-align: center; border: 2px solid {winner_color};">
        <h2 style="color: {winner_color}; margin: 0; font-size: 2rem;">🏆 {winner} Wins!</h2>
        <p style="color: #A0A0A0; margin: 1rem 0 0 0; font-size: 1.1rem;">
            {"Perfect strategic balance!" if winner == "Tie" else f"Superior strategy and decision-making by {winner}"}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.success("🎉 Simulation completed!")
    
    # Display results
    st.subheader("📈 Results")
    
    # Results table
    st.dataframe(results_df[["Round", "A Move", "B Move", "A Payoff", "B Payoff", "A Cumulative", "B Cumulative"]])
    
    # Enhanced Visualizations
    st.markdown("### 📈 Interactive Visualizations")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["📊 Cumulative Scores", "🎯 Move Distribution", "🧠 Decision Patterns"])
    
    with tab1:
        st.markdown("#### Cumulative Scores Over Time")
        
        fig_line = go.Figure()
        
        fig_line.add_trace(go.Scatter(
            x=results_df['Round'],
            y=results_df['A Cumulative'],
            mode='lines+markers',
            name=f'Agent A ({ai_personalities[agent_a_type]["name"]})',
            line=dict(color='#4F86F7', width=3),
            marker=dict(size=8, color='#4F86F7')
        ))
        
        fig_line.add_trace(go.Scatter(
            x=results_df['Round'],
            y=results_df['B Cumulative'],
            mode='lines+markers',
            name=f'Agent B ({ai_personalities[agent_b_type]["name"]})',
            line=dict(color='#A020F0', width=3),
            marker=dict(size=8, color='#A020F0')
        ))
        
        fig_line.update_layout(
            title="Score Progression Throughout the Game",
            xaxis_title="Round",
            yaxis_title="Cumulative Score",
            template="plotly_dark",
            height=400,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_line, use_container_width=True)
    
    with tab2:
        st.markdown("#### Move Distribution Analysis")
        
        # Create move distribution data
        agent_a_moves = results_df['A Move'].value_counts()
        agent_b_moves = results_df['B Move'].value_counts()
        
        fig_bar = go.Figure()
        
        fig_bar.add_trace(go.Bar(
            x=agent_a_moves.index,
            y=agent_a_moves.values,
            name=f'Agent A ({ai_personalities[agent_a_type]["name"]})',
            marker_color='#4F86F7',
            text=agent_a_moves.values,
            textposition='auto',
        ))
        
        fig_bar.add_trace(go.Bar(
            x=agent_b_moves.index,
            y=agent_b_moves.values,
            name=f'Agent B ({ai_personalities[agent_b_type]["name"]})',
            marker_color='#A020F0',
            text=agent_b_moves.values,
            textposition='auto',
        ))
        
        fig_bar.update_layout(
            title="Decision Frequency by Agent",
            xaxis_title="Decision Type",
            yaxis_title="Number of Times Chosen",
            template="plotly_dark",
            height=400,
            barmode='group'
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab3:
        st.markdown("#### Decision Patterns Over Time")
        
        # Create decision timeline
        fig_timeline = go.Figure()
        
        # Convert moves to numeric for plotting
        a_moves_numeric = [1 if move == "Cooperate" else 0 for move in results_df['A Move']]
        b_moves_numeric = [1 if move == "Cooperate" else 0 for move in results_df['B Move']]
        
        fig_timeline.add_trace(go.Scatter(
            x=results_df['Round'],
            y=a_moves_numeric,
            mode='lines+markers',
            name=f'Agent A ({ai_personalities[agent_a_type]["name"]})',
            line=dict(color='#4F86F7', width=2),
            marker=dict(size=6, color='#4F86F7'),
            yaxis='y1'
        ))
        
        fig_timeline.add_trace(go.Scatter(
            x=results_df['Round'],
            y=b_moves_numeric,
            mode='lines+markers',
            name=f'Agent B ({ai_personalities[agent_b_type]["name"]})',
            line=dict(color='#A020F0', width=2),
            marker=dict(size=6, color='#A020F0'),
            yaxis='y1'
        ))
        
        fig_timeline.update_layout(
            title="Cooperation vs Defection Timeline",
            xaxis_title="Round",
            yaxis=dict(
                title="Decision",
                tickvals=[0, 1],
                ticktext=["Defect", "Cooperate"]
            ),
            template="plotly_dark",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    
    # AI Reasoning Section
    st.markdown("### 🧠 AI Decision-Making Insights")
    
    # Create expandable sections for each agent's reasoning
    with st.expander("🤖 Agent A Reasoning Pattern", expanded=True):
        last_round = results_df.iloc[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2a2a3e, #3a3a5e); padding: 1rem; border-radius: 8px; border-left: 4px solid #4F86F7;">
                <h5 style="color: #4F86F7; margin: 0;">Final Decision</h5>
                <p style="color: white; font-size: 1.1rem; margin: 0.5rem 0;">{last_round['A Move']}</p>
                <p style="color: #A0A0A0; font-size: 0.9rem; margin: 0;">{last_round['A Reason']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            a_coop_rate = (results_df['A Move'] == 'Cooperate').mean()
            trend = "📈 Increasing" if results_df['A Move'].iloc[-5:].value_counts().get('Cooperate', 0) > 2 else "📉 Decreasing"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2a2a3e, #3a3a5e); padding: 1rem; border-radius: 8px;">
                <h5 style="color: #4F86F7; margin: 0;">Behavioral Pattern</h5>
                <p style="color: white; margin: 0.5rem 0;">Cooperation Rate: {a_coop_rate:.1%}</p>
                <p style="color: #A0A0A0; font-size: 0.9rem; margin: 0;">Trend: {trend}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with st.expander("� Agent B Reasoning Pattern", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2a2a3e, #3a3a5e); padding: 1rem; border-radius: 8px; border-left: 4px solid #A020F0;">
                <h5 style="color: #A020F0; margin: 0;">Final Decision</h5>
                <p style="color: white; font-size: 1.1rem; margin: 0.5rem 0;">{last_round['B Move']}</p>
                <p style="color: #A0A0A0; font-size: 0.9rem; margin: 0;">{last_round['B Reason']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            b_coop_rate = (results_df['B Move'] == 'Cooperate').mean()
            trend = "📈 Increasing" if results_df['B Move'].iloc[-5:].value_counts().get('Cooperate', 0) > 2 else "📉 Decreasing"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2a2a3e, #3a3a5e); padding: 1rem; border-radius: 8px;">
                <h5 style="color: #A020F0; margin: 0;">Behavioral Pattern</h5>
                <p style="color: white; margin: 0.5rem 0;">Cooperation Rate: {b_coop_rate:.1%}</p>
                <p style="color: #A0A0A0; font-size: 0.9rem; margin: 0;">Trend: {trend}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Advanced Statistics
    st.markdown("### 📊 Advanced Game Statistics")
    
    # Calculate advanced metrics
    mutual_coop = ((results_df['A Move'] == 'Cooperate') & 
                  (results_df['B Move'] == 'Cooperate')).sum()
    mutual_defect = ((results_df['A Move'] == 'Defect') & 
                   (results_df['B Move'] == 'Defect')).sum()
    a_betrayed = ((results_df['A Move'] == 'Cooperate') & 
                 (results_df['B Move'] == 'Defect')).sum()
    b_betrayed = ((results_df['A Move'] == 'Defect') & 
                 (results_df['B Move'] == 'Cooperate')).sum()
    
    # Create statistics grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e3a1e, #2e4a2e); padding: 1rem; border-radius: 8px; text-align: center;">
            <h4 style="color: #32CD32; margin: 0;">🤝 Mutual Cooperation</h4>
            <p style="color: white; font-size: 1.5rem; margin: 0.5rem 0;">{mutual_coop}</p>
            <p style="color: #A0A0A0; font-size: 0.8rem; margin: 0;">{mutual_coop/num_rounds:.1%} of rounds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #3a1e1e, #4a2e2e); padding: 1rem; border-radius: 8px; text-align: center;">
            <h4 style="color: #FF6B6B; margin: 0;">⚔️ Mutual Defection</h4>
            <p style="color: white; font-size: 1.5rem; margin: 0.5rem 0;">{mutual_defect}</p>
            <p style="color: #A0A0A0; font-size: 0.8rem; margin: 0;">{mutual_defect/num_rounds:.1%} of rounds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #3a2e1e, #4a3e2e); padding: 1rem; border-radius: 8px; text-align: center;">
            <h4 style="color: #FFD700; margin: 0;">😞 A Betrayed</h4>
            <p style="color: white; font-size: 1.5rem; margin: 0.5rem 0;">{a_betrayed}</p>
            <p style="color: #A0A0A0; font-size: 0.8rem; margin: 0;">{a_betrayed/num_rounds:.1%} of rounds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #2e1e3a, #3e2e4a); padding: 1rem; border-radius: 8px; text-align: center;">
            <h4 style="color: #9370DB; margin: 0;">😞 B Betrayed</h4>
            <p style="color: white; font-size: 1.5rem; margin: 0.5rem 0;">{b_betrayed}</p>
            <p style="color: #A0A0A0; font-size: 0.8rem; margin: 0;">{b_betrayed/num_rounds:.1%} of rounds</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Data table with enhanced styling
    st.markdown("### 📋 Detailed Round-by-Round Results")
    
    # Style the dataframe
    styled_df = results_df.style.background_gradient(
        subset=['A Cumulative', 'B Cumulative'], 
        cmap='RdYlBu_r'
    ).format({
        'A Cumulative': '{:.0f}',
        'B Cumulative': '{:.0f}',
        'A Payoff': '{:.0f}',
        'B Payoff': '{:.0f}'
    })
    
    st.dataframe(styled_df, use_container_width=True, height=300)
    
    # Export options
    st.markdown("### 📥 Export Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="� Download CSV",
            data=csv,
            file_name=f"ai_simulation_{agent_a_type}_vs_{agent_b_type}_{experiment_type}.csv",
            mime="text/csv"
        )
    
    with col2:
        json_data = results_df.to_json(orient='records', indent=2)
        st.download_button(
            label="📄 Download JSON",
            data=json_data,
            file_name=f"ai_simulation_{agent_a_type}_vs_{agent_b_type}_{experiment_type}.json",
            mime="application/json"
        )
    
    with col3:
        # Create summary report
        summary = f"""
# AI Prisoner's Dilemma Simulation Report

## Configuration
- Agent A: {ai_personalities[agent_a_type]['name']}
- Agent B: {ai_personalities[agent_b_type]['name']}
- Experiment Type: {experiment_type}
- Number of Rounds: {num_rounds}

## Results
- Agent A Final Score: {final_a}
- Agent B Final Score: {final_b}
- Winner: {winner}

## Statistics
- Agent A Cooperation Rate: {a_coop_rate:.1%}
- Agent B Cooperation Rate: {b_coop_rate:.1%}
- Mutual Cooperation: {mutual_coop} rounds ({mutual_coop/num_rounds:.1%})
- Mutual Defection: {mutual_defect} rounds ({mutual_defect/num_rounds:.1%})
"""
        st.download_button(
            label="📋 Download Report",
            data=summary,
            file_name=f"ai_simulation_report_{agent_a_type}_vs_{agent_b_type}.md",
            mime="text/markdown"
        )

# Enhanced Sidebar Information
with st.sidebar:
    st.markdown("---")
    
    # About section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a1e, #2e4a2e); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: #32CD32; margin: 0; font-size: 1.3rem;">💡 About Simple AI</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2a2a3e, #3a3a5e); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h4 style="color: #4F86F7; margin: 0 0 0.5rem 0;">✨ Key Features</h4>
        <ul style="color: #A0A0A0; margin: 0; padding-left: 1rem;">
            <li>✅ Zero external dependencies</li>
            <li>✅ No API costs or keys required</li>
            <li>✅ Instant setup and execution</li>
            <li>✅ Intelligent behavior patterns</li>
            <li>✅ Educational and research ready</li>
            <li>✅ Lightning-fast simulations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Personality types
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2a2a3e, #3a3a5e); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h4 style="color: #A020F0; margin: 0 0 0.5rem 0;">🧠 AI Personalities</h4>
        <div style="color: #A0A0A0; font-size: 0.9rem;">
            <p style="margin: 0.3rem 0;"><strong>Rational:</strong> Logic-based decisions</p>
            <p style="margin: 0.3rem 0;"><strong>Cooperative:</strong> Trusting and helpful</p>
            <p style="margin: 0.3rem 0;"><strong>Suspicious:</strong> Defensive and cautious</p>
            <p style="margin: 0.3rem 0;"><strong>Emotional:</strong> Influenced by feelings</p>
            <p style="margin: 0.3rem 0;"><strong>Adaptive:</strong> Learns from opponent</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # How it works
    st.markdown("""
    <div style="background: linear-gradient(135deg, #3a2e1e, #4a3e2e); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: #FFD700; margin: 0; font-size: 1.3rem;">🔧 How It Works</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2a2a3e, #3a3a5e); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h4 style="color: #FFD700; margin: 0 0 0.5rem 0;">⚡ Intelligence Engine</h4>
        <ul style="color: #A0A0A0; margin: 0; padding-left: 1rem; font-size: 0.9rem;">
            <li>🧮 Personality-based decision making</li>
            <li>📊 Historical behavior analysis</li>
            <li>🎲 Probabilistic reasoning</li>
            <li>🎯 Contextual prompt adjustment</li>
            <li>😊 Emotional state simulation</li>
            <li>🔄 Adaptive learning patterns</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.success("🚀 **No external AI required!** All intelligence is built into the rule-based system.")
    
    # Performance info
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2e1e3a, #3e2e4a); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
        <h4 style="color: #9370DB; margin: 0 0 0.5rem 0;">⚡ Performance</h4>
        <p style="color: #A0A0A0; margin: 0; font-size: 0.9rem;">
            Simulations run instantly on your local machine with zero latency and unlimited usage.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #1a1a2e, #2a2a4e); border-radius: 10px; margin-top: 2rem;">
    <h3 style="color: #4F86F7; margin: 0;">🤖 Simple AI Prisoner's Dilemma</h3>
    <p style="color: #A0A0A0; margin: 0.5rem 0 0 0;">Zero dependencies, maximum intelligence!</p>
    <p style="color: #666; margin: 0.5rem 0 0 0; font-size: 0.8rem;">
        Experience sophisticated AI behavior without external requirements
    </p>
</div>
""", unsafe_allow_html=True)
