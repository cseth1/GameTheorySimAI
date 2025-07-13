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

# Clean monochrome CSS theme
st.markdown("""
<style>
    /* Import clean font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main app styling */
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }
    
    /* Main content background */
    .main .block-container {
        background-color: #ffffff;
        padding: 2rem 1rem;
    }
    
    /* Title styling */
    .main-title {
        color: #000000;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
    }
    
    .subtitle {
        text-align: center;
        color: #666666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Card styling */
    .metric-card {
        background-color: #ffffff;
        border: 2px solid #000000;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        background-color: #f5f5f5;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #000000;
        color: #ffffff;
        border: 2px solid #000000;
        border-radius: 6px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #333333;
        border-color: #333333;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
        border-right: 2px solid #e9ecef;
    }
    
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: #000000;
    }
    
    .css-1d391kg p, .css-1d391kg span, .css-1d391kg div {
        color: #333333;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #ffffff;
        border: 2px solid #000000;
        border-radius: 6px;
    }
    
    .stSelectbox > div > div > div {
        color: #000000;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background-color: #000000;
    }
    
    .stSlider > div > div > div {
        background-color: #e9ecef;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: #000000;
    }
    
    .stProgress > div > div {
        background-color: #e9ecef;
    }
    
    /* Agent card styling */
    .agent-card {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #000000;
    }
    
    .agent-card h4 {
        color: #000000;
        margin-bottom: 0.5rem;
    }
    
    .agent-card p {
        color: #666666;
        margin: 0;
    }
    
    /* Results section styling */
    .results-section {
        background-color: #ffffff;
        border: 2px solid #000000;
        border-radius: 8px;
        padding: 2rem;
        margin: 2rem 0;
    }
    
    .results-section h2, .results-section h3 {
        color: #000000;
        border-bottom: 2px solid #000000;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Table styling */
    .dataframe {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 6px;
    }
    
    .dataframe th {
        background-color: #f8f9fa;
        color: #000000;
        font-weight: 600;
        border-bottom: 2px solid #dee2e6;
    }
    
    .dataframe td {
        color: #333333;
        border-bottom: 1px solid #dee2e6;
    }
    
    /* Metric styling */
    .metric-container {
        background-color: #ffffff;
        border: 2px solid #000000;
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #000000;
        margin: 0;
    }
    
    .metric-label {
        color: #666666;
        font-size: 0.9rem;
        margin-top: 0.25rem;
        font-weight: 500;
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    
    p, span, div {
        color: #333333;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .info-box h4 {
        color: #000000;
        margin-bottom: 0.5rem;
    }
    
    .info-box p {
        color: #666666;
        margin: 0;
    }
    
    /* Status indicators */
    .status-success {
        color: #000000;
        background-color: #ffffff;
        border: 2px solid #000000;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    
    .status-neutral {
        color: #666666;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
        display: inline-block;
    }
    
    /* Download button styling */
    .download-section {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 1.5rem;
        margin: 2rem 0;
    }
    
    .download-section h3 {
        color: #000000;
        margin-bottom: 1rem;
    }
    
    /* Chart container */
    .chart-container {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Clean input styling */
    .stTextInput > div > div > input {
        background-color: #ffffff;
        border: 2px solid #000000;
        border-radius: 6px;
        color: #000000;
    }
    
    /* Checkbox styling */
    .stCheckbox > label > div {
        background-color: #ffffff;
        border: 2px solid #000000;
    }
    
    .stCheckbox > label > div[data-checked="true"] {
        background-color: #000000;
    }
    
    /* Radio button styling */
    .stRadio > label > div {
        background-color: #ffffff;
        border: 2px solid #000000;
    }
    
    .stRadio > label > div[data-checked="true"] {
        background-color: #000000;
    }
    
    /* Tabs styling */
    .stTabs > div > div > div {
        background-color: #ffffff;
        border: 2px solid #000000;
        border-radius: 6px;
    }
    
    .stTabs > div > div > div > div {
        color: #000000;
        font-weight: 600;
    }
    
    /* Success/Error messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        background-color: #ffffff;
        border: 2px solid #000000;
        border-radius: 6px;
        color: #000000;
    }
    
    /* Clean spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        
        .subtitle {
            font-size: 1rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
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
def run_simulation(agent_a_type, agent_b_type, num_rounds, experiment_type="Basic"):
    """Run a game simulation with AI agents"""
    
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

# Clean chart creation functions
def create_clean_charts(df):
    """Create clean, colorful charts with white backgrounds"""
    
    # Cumulative scores chart
    fig_cumulative = go.Figure()
    
    fig_cumulative.add_trace(go.Scatter(
        x=df['Round'],
        y=df['A_Cumulative'],
        mode='lines+markers',
        name='Agent A',
        line=dict(color='#2E8B57', width=3),  # Sea Green
        marker=dict(size=6, color='#2E8B57')
    ))
    
    fig_cumulative.add_trace(go.Scatter(
        x=df['Round'],
        y=df['B_Cumulative'],
        mode='lines+markers',
        name='Agent B',
        line=dict(color='#4169E1', width=3),  # Royal Blue
        marker=dict(size=6, color='#4169E1')
    ))
    
    fig_cumulative.update_layout(
        title=dict(text="Cumulative Scores Over Time", font=dict(color='#000000', size=18)),
        xaxis_title="Round",
        yaxis_title="Cumulative Score",
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        hovermode='x unified',
        font=dict(color='#000000'),
        xaxis=dict(
            gridcolor='#E5E5E5',
            zerolinecolor='#E5E5E5',
            color='#000000'
        ),
        yaxis=dict(
            gridcolor='#E5E5E5',
            zerolinecolor='#E5E5E5',
            color='#000000'
        )
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
        marker_colors=['#32CD32', '#DC143C']  # Lime Green, Crimson
    ), row=1, col=1)
    
    fig_actions.add_trace(go.Pie(
        labels=b_actions.index,
        values=b_actions.values,
        name="Agent B",
        marker_colors=['#32CD32', '#DC143C']  # Lime Green, Crimson
    ), row=1, col=2)
    
    fig_actions.update_layout(
        title=dict(text="Action Distribution", font=dict(color='#000000', size=18)),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        showlegend=True,
        font=dict(color='#000000')
    )
    
    return fig_cumulative, fig_actions

def create_payoff_heatmap(df):
    """Create a clean heatmap of payoffs over time"""
    
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
        colorbar=dict(title="Payoff", titlefont=dict(color='#000000')),
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=dict(text="Payoff Heatmap Over Time", font=dict(color='#000000', size=18)),
        xaxis_title="Round",
        yaxis_title="Agent",
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=300,
        font=dict(color='#000000'),
        xaxis=dict(color='#000000'),
        yaxis=dict(color='#000000')
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
            st.markdown(f'<div class="info-box"><h4>{personality}</h4><p>{description}</p></div>', 
                       unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.markdown("## 🎲 Simulation Control")
        
        # Run simulation button
        if st.button("🚀 Run Simulation", use_container_width=True):
            with st.spinner("Running simulation..."):
                # Run the simulation
                df, agent_a, agent_b = run_simulation(
                    agent_a_type, agent_b_type, num_rounds, experiment_type
                )
                
                # Store results in session state
                st.session_state.df = df
                st.session_state.agent_a = agent_a
                st.session_state.agent_b = agent_b
                
                st.success("✅ Simulation completed successfully!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.markdown("## 📊 Quick Stats")
        
        if 'df' in st.session_state:
            df = st.session_state.df
            
            # Quick metrics
            total_rounds = len(df)
            a_final_score = df['A_Cumulative'].iloc[-1]
            b_final_score = df['B_Cumulative'].iloc[-1]
            cooperation_rate = (df['A_Move'].value_counts().get('Cooperate', 0) + 
                              df['B_Move'].value_counts().get('Cooperate', 0)) / (total_rounds * 2)
            
            # Display metrics with clean styling
            st.markdown(f'<div class="metric-container"><div class="metric-value">{total_rounds}</div><div class="metric-label">Total Rounds</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-container"><div class="metric-value">{a_final_score:+}</div><div class="metric-label">Agent A Score</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-container"><div class="metric-value">{b_final_score:+}</div><div class="metric-label">Agent B Score</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-container"><div class="metric-value">{cooperation_rate:.1%}</div><div class="metric-label">Cooperation Rate</div></div>', unsafe_allow_html=True)
            
            # Winner determination
            if a_final_score > b_final_score:
                st.markdown('<div class="status-success">🏆 Agent A Wins!</div>', unsafe_allow_html=True)
            elif b_final_score > a_final_score:
                st.markdown('<div class="status-success">🏆 Agent B Wins!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-neutral">🤝 Draw!</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Results section
    if 'df' in st.session_state:
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.markdown("## 📈 Results Dashboard")
        
        df = st.session_state.df
        
        # Create visualizations
        fig_cumulative, fig_actions = create_clean_charts(df)
        fig_heatmap = create_payoff_heatmap(df)
        
        # Display charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig_cumulative, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig_actions, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Game history table
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.markdown("## 📋 Game History")
        
        # Display detailed results
        display_df = df[['Round', 'A_Move', 'B_Move', 'A_Payoff', 'B_Payoff', 'A_Cumulative', 'B_Cumulative']].copy()
        
        st.dataframe(display_df, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Agent reasoning section
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.markdown("## 🧠 Agent Reasoning")
        
        # Display recent reasoning
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Agent A Recent Reasoning")
            for i, reasoning in enumerate(st.session_state.agent_a.reasoning_history[-5:]):
                round_num = len(st.session_state.agent_a.reasoning_history) - 5 + i + 1
                st.markdown(f'<div class="agent-card"><h4>Round {round_num}</h4><p>{reasoning}</p></div>', 
                           unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Agent B Recent Reasoning")
            for i, reasoning in enumerate(st.session_state.agent_b.reasoning_history[-5:]):
                round_num = len(st.session_state.agent_b.reasoning_history) - 5 + i + 1
                st.markdown(f'<div class="agent-card"><h4>Round {round_num}</h4><p>{reasoning}</p></div>', 
                           unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Download results
        st.markdown('<div class="download-section">', unsafe_allow_html=True)
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
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
