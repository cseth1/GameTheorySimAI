import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import json
import random
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure page
st.set_page_config(
    page_title="Game Theory AI Lab",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --accent-color: #06b6d4;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --background-dark: #0f1419;
        --surface-color: #1a202c;
        --text-primary: #ffffff;
        --text-secondary: #94a3b8;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .main-container {
        background: linear-gradient(135deg, #0f1419 0%, #1a202c 100%);
        padding: 0;
        margin: 0;
    }
    
    /* Title styling */
    .main-title {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Card styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3);
        border-color: rgba(99, 102, 241, 0.5);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(26, 32, 44, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Progress bar */
    .progress-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Agent card styling */
    .agent-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid var(--primary-color);
        transition: all 0.3s ease;
    }
    
    .agent-card:hover {
        background: rgba(255, 255, 255, 0.08);
        transform: translateX(5px);
    }
    
    /* Results table styling */
    .results-table {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Animation for cards */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-in {
        animation: slideIn 0.6s ease-out;
    }
    
    /* Status indicators */
    .status-success {
        color: #10b981;
        background: rgba(16, 185, 129, 0.1);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .status-warning {
        color: #f59e0b;
        background: rgba(245, 158, 11, 0.1);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    
    /* Metric styling */
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# AI Agent Classes
class GameAgent:
    def __init__(self, name, personality_type):
        self.name = name
        self.personality = personality_type
        self.history = []
        self.opponent_history = []
        self.score = 0
        self.reasoning_history = []
    
    def make_decision(self, round_num, opponent_last_move=None):
        """Make a decision based on personality and game state"""
        if self.personality == "Rational":
            return self._rational_decision(round_num)
        elif self.personality == "Cooperative":
            return self._cooperative_decision(round_num)
        elif self.personality == "Suspicious":
            return self._suspicious_decision(round_num)
        elif self.personality == "Emotional":
            return self._emotional_decision(round_num)
        elif self.personality == "Adaptive":
            return self._adaptive_decision(round_num)
        else:
            return self._random_decision()
    
    def _rational_decision(self, round_num):
        """Nash equilibrium strategy with some cooperation"""
        if len(self.opponent_history) == 0:
            move = "C"
            reason = "Starting with cooperation to test opponent's strategy"
        else:
            cooperation_rate = self.opponent_history.count("C") / len(self.opponent_history)
            if cooperation_rate > 0.6:
                move = "C"
                reason = f"Opponent cooperates {cooperation_rate:.1%} of the time, reciprocating"
            else:
                move = "D"
                reason = f"Opponent defects frequently ({1-cooperation_rate:.1%}), protecting myself"
        
        return move, reason
    
    def _cooperative_decision(self, round_num):
        """Tends to cooperate unless severely betrayed"""
        if len(self.opponent_history) == 0:
            move = "C"
            reason = "Starting with trust and cooperation"
        else:
            recent_betrayals = self.opponent_history[-3:].count("D")
            if recent_betrayals >= 2:
                move = "D"
                reason = f"Recent betrayals detected ({recent_betrayals}/3), reluctantly defecting"
            else:
                move = "C"
                reason = "Maintaining cooperative stance for mutual benefit"
        
        return move, reason
    
    def _suspicious_decision(self, round_num):
        """Defensive strategy, quick to defect"""
        if len(self.opponent_history) == 0:
            move = "D"
            reason = "Starting defensively to avoid exploitation"
        else:
            if self.opponent_history[-1] == "D":
                move = "D"
                reason = "Opponent defected last round, maintaining defense"
            else:
                cooperation_rate = self.opponent_history.count("C") / len(self.opponent_history)
                if cooperation_rate > 0.8:
                    move = "C"
                    reason = f"Opponent shows strong cooperation ({cooperation_rate:.1%}), cautiously trusting"
                else:
                    move = "D"
                    reason = "Remaining defensive due to insufficient trust"
        
        return move, reason
    
    def _emotional_decision(self, round_num):
        """Decisions influenced by emotional states"""
        if len(self.opponent_history) == 0:
            move = "C"
            reason = "Feeling hopeful about cooperation"
        else:
            recent_score = sum(self.get_payoff(self.history[i], self.opponent_history[i])[0] 
                             for i in range(max(0, len(self.history)-3), len(self.history)))
            
            if recent_score < 0:
                move = "D"
                reason = f"Feeling frustrated by recent losses ({recent_score}), acting defensively"
            elif recent_score > 3:
                move = "C"
                reason = f"Feeling good about recent success ({recent_score}), maintaining cooperation"
            else:
                if random.random() < 0.3:
                    move = "D"
                    reason = "Feeling anxious about being exploited"
                else:
                    move = "C"
                    reason = "Hoping for mutual cooperation despite uncertainty"
        
        return move, reason
    
    def _adaptive_decision(self, round_num):
        """Learns and adapts to opponent's strategy"""
        if len(self.opponent_history) < 2:
            move = "C"
            reason = "Learning opponent's pattern, starting cooperatively"
        else:
            # Simple pattern recognition
            if len(self.opponent_history) >= 3:
                last_three = self.opponent_history[-3:]
                if last_three == ["C", "C", "C"]:
                    move = "C"
                    reason = "Detected consistent cooperation pattern, reciprocating"
                elif last_three == ["D", "D", "D"]:
                    move = "D"
                    reason = "Detected consistent defection pattern, defending"
                else:
                    # Tit-for-tat strategy
                    move = self.opponent_history[-1]
                    reason = f"Mirroring opponent's last move ({move})"
            else:
                move = self.opponent_history[-1]
                reason = f"Adapting to opponent's last move ({move})"
        
        return move, reason
    
    def _random_decision(self):
        """Random strategy as baseline"""
        move = random.choice(["C", "D"])
        reason = f"Random decision: {'cooperating' if move == 'C' else 'defecting'}"
        return move, reason
    
    def get_payoff(self, my_move, opponent_move):
        """Calculate payoffs for prisoner's dilemma"""
        if my_move == "C" and opponent_move == "C":
            return 2, 2
        elif my_move == "C" and opponent_move == "D":
            return -1, 3
        elif my_move == "D" and opponent_move == "C":
            return 3, -1
        else:
            return 0, 0
    
    def update_history(self, my_move, opponent_move, reasoning):
        """Update game history"""
        self.history.append(my_move)
        self.opponent_history.append(opponent_move)
        self.reasoning_history.append(reasoning)
        
        my_payoff, _ = self.get_payoff(my_move, opponent_move)
        self.score += my_payoff

# Game simulation function
def run_modern_simulation(agent_a_type, agent_b_type, num_rounds, experiment_type="Basic"):
    """Run a modern game simulation with enhanced AI agents"""
    
    # Initialize agents
    agent_a = GameAgent("Agent A", agent_a_type)
    agent_b = GameAgent("Agent B", agent_b_type)
    
    # Game history
    game_history = []
    
    # Progress tracking
    progress_container = st.container()
    
    for round_num in range(1, num_rounds + 1):
        # Update progress
        with progress_container:
            progress = round_num / num_rounds
            st.progress(progress, text=f"Round {round_num}/{num_rounds}")
        
        # Agents make decisions
        a_move, a_reason = agent_a.make_decision(round_num)
        b_move, b_reason = agent_b.make_decision(round_num)
        
        # Calculate payoffs
        a_payoff, b_payoff = agent_a.get_payoff(a_move, b_move)
        
        # Update agent histories
        agent_a.update_history(a_move, b_move, a_reason)
        agent_b.update_history(b_move, a_move, b_reason)
        
        # Record game state
        game_history.append({
            "Round": round_num,
            "A_Move": "Cooperate" if a_move == "C" else "Defect",
            "B_Move": "Cooperate" if b_move == "C" else "Defect",
            "A_Payoff": a_payoff,
            "B_Payoff": b_payoff,
            "A_Cumulative": agent_a.score,
            "B_Cumulative": agent_b.score,
            "A_Reasoning": a_reason,
            "B_Reasoning": b_reason
        })
    
    # Clear progress
    progress_container.empty()
    
    return pd.DataFrame(game_history), agent_a, agent_b

# Modern visualization functions
def create_modern_charts(df):
    """Create modern, interactive charts"""
    
    # Cumulative scores chart
    fig_cumulative = go.Figure()
    
    fig_cumulative.add_trace(go.Scatter(
        x=df['Round'],
        y=df['A_Cumulative'],
        mode='lines+markers',
        name='Agent A',
        line=dict(color='#6366f1', width=3),
        marker=dict(size=6, color='#6366f1')
    ))
    
    fig_cumulative.add_trace(go.Scatter(
        x=df['Round'],
        y=df['B_Cumulative'],
        mode='lines+markers',
        name='Agent B',
        line=dict(color='#8b5cf6', width=3),
        marker=dict(size=6, color='#8b5cf6')
    ))
    
    fig_cumulative.update_layout(
        title="Cumulative Scores Over Time",
        xaxis_title="Round",
        yaxis_title="Cumulative Score",
        template="plotly_dark",
        height=400,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        hovermode='x unified'
    )
    
    # Action distribution chart
    a_actions = df['A_Move'].value_counts()
    b_actions = df['B_Move'].value_counts()
    
    fig_actions = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Agent A Actions', 'Agent B Actions'),
        specs=[[{"type": "pie"}, {"type": "pie"}]]
    )
    
    fig_actions.add_trace(go.Pie(
        labels=a_actions.index,
        values=a_actions.values,
        name="Agent A",
        marker_colors=['#10b981', '#ef4444']
    ), row=1, col=1)
    
    fig_actions.add_trace(go.Pie(
        labels=b_actions.index,
        values=b_actions.values,
        name="Agent B",
        marker_colors=['#10b981', '#ef4444']
    ), row=1, col=2)
    
    fig_actions.update_layout(
        title="Action Distribution",
        template="plotly_dark",
        height=400,
        showlegend=True
    )
    
    return fig_cumulative, fig_actions

def create_payoff_heatmap(df):
    """Create a heatmap of payoffs over time"""
    
    # Create payoff matrix over time
    rounds = df['Round'].values
    a_payoffs = df['A_Payoff'].values
    b_payoffs = df['B_Payoff'].values
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=[a_payoffs, b_payoffs],
        x=rounds,
        y=['Agent A', 'Agent B'],
        colorscale='RdYlBu',
        colorbar=dict(title="Payoff"),
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Payoff Heatmap Over Time",
        xaxis_title="Round",
        yaxis_title="Agent",
        template="plotly_dark",
        height=300
    )
    
    return fig

# Main UI
def main():
    # Title section
    st.markdown('<h1 class="main-title">🎮 Game Theory AI Lab</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Explore strategic decision-making with intelligent AI agents</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Agent selection
        st.markdown("### 🤖 Agent Selection")
        agent_types = ["Rational", "Cooperative", "Suspicious", "Emotional", "Adaptive"]
        
        agent_a_type = st.selectbox(
            "Agent A Personality:",
            agent_types,
            index=0,
            help="Choose the decision-making style for Agent A"
        )
        
        agent_b_type = st.selectbox(
            "Agent B Personality:",
            agent_types,
            index=1,
            help="Choose the decision-making style for Agent B"
        )
        
        # Game parameters
        st.markdown("### 🎯 Game Parameters")
        num_rounds = st.slider(
            "Number of Rounds:",
            min_value=5,
            max_value=100,
            value=20,
            help="Number of rounds to simulate"
        )
        
        experiment_type = st.selectbox(
            "Experiment Type:",
            ["Basic", "Moral", "Fear", "Hope"],
            help="Choose the experimental context"
        )
        
        # Agent personality explanations
        st.markdown("### 📚 Agent Personalities")
        personality_info = {
            "Rational": "Logic-based decisions, adapts to opponent patterns",
            "Cooperative": "Trusting and helpful, slow to retaliate",
            "Suspicious": "Defensive and cautious, quick to defect",
            "Emotional": "Influenced by recent outcomes and feelings",
            "Adaptive": "Learns opponent patterns, uses tit-for-tat"
        }
        
        for personality, description in personality_info.items():
            st.markdown(f"**{personality}:** {description}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🎲 Simulation Control")
        
        # Run simulation button
        if st.button("🚀 Run Simulation", use_container_width=True):
            with st.spinner("Running simulation..."):
                # Run the simulation
                df, agent_a, agent_b = run_modern_simulation(
                    agent_a_type, agent_b_type, num_rounds, experiment_type
                )
                
                # Store results in session state
                st.session_state.df = df
                st.session_state.agent_a = agent_a
                st.session_state.agent_b = agent_b
                
                st.success("Simulation completed!")
    
    with col2:
        st.markdown("## 📊 Quick Stats")
        
        if 'df' in st.session_state:
            df = st.session_state.df
            
            # Quick metrics
            total_rounds = len(df)
            a_final_score = df['A_Cumulative'].iloc[-1]
            b_final_score = df['B_Cumulative'].iloc[-1]
            cooperation_rate = (df['A_Move'].value_counts().get('Cooperate', 0) + 
                              df['B_Move'].value_counts().get('Cooperate', 0)) / (total_rounds * 2)
            
            st.metric("Total Rounds", total_rounds)
            st.metric("Agent A Score", a_final_score)
            st.metric("Agent B Score", b_final_score)
            st.metric("Cooperation Rate", f"{cooperation_rate:.1%}")
    
    # Results section
    if 'df' in st.session_state:
        st.markdown("## 📈 Results Dashboard")
        
        df = st.session_state.df
        
        # Create visualizations
        fig_cumulative, fig_actions = create_modern_charts(df)
        fig_heatmap = create_payoff_heatmap(df)
        
        # Display charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(fig_cumulative, use_container_width=True)
        
        with col2:
            st.plotly_chart(fig_actions, use_container_width=True)
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Game history table
        st.markdown("## 📋 Game History")
        
        # Display detailed results
        display_df = df[['Round', 'A_Move', 'B_Move', 'A_Payoff', 'B_Payoff', 'A_Cumulative', 'B_Cumulative']].copy()
        
        # Style the dataframe
        styled_df = display_df.style.format({
            'A_Payoff': '{:+d}',
            'B_Payoff': '{:+d}',
            'A_Cumulative': '{:+d}',
            'B_Cumulative': '{:+d}'
        }).apply(lambda x: ['background-color: rgba(16, 185, 129, 0.1)' if v == 'Cooperate' 
                           else 'background-color: rgba(239, 68, 68, 0.1)' for v in x], 
                 subset=['A_Move', 'B_Move'])
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Agent reasoning section
        st.markdown("## 🧠 Agent Reasoning")
        
        # Display recent reasoning
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Agent A Reasoning")
            for i, reasoning in enumerate(st.session_state.agent_a.reasoning_history[-5:]):
                round_num = len(st.session_state.agent_a.reasoning_history) - 5 + i + 1
                st.markdown(f"**Round {round_num}:** {reasoning}")
        
        with col2:
            st.markdown("### Agent B Reasoning")
            for i, reasoning in enumerate(st.session_state.agent_b.reasoning_history[-5:]):
                round_num = len(st.session_state.agent_b.reasoning_history) - 5 + i + 1
                st.markdown(f"**Round {round_num}:** {reasoning}")
        
        # Download results
        st.markdown("## 💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"game_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"game_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
