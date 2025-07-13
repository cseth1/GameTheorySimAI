import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import random
import time
from datetime import datetime
import base64
from io import BytesIO

# Configure page with modern settings
st.set_page_config(
    page_title="🎮 Game Theory AI Lab | Premium",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced modern CSS with animations and glassmorphism
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    :root {
        --primary: #6366f1;
        --primary-light: #8b5cf6;
        --secondary: #06b6d4;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --dark: #0f172a;
        --surface: #1e293b;
        --surface-light: #334155;
        --text: #f8fafc;
        --text-muted: #94a3b8;
        --border: rgba(255, 255, 255, 0.1);
        --shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }
    
    /* Hide default Streamlit styling */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        background-attachment: fixed;
    }
    
    /* Glassmorphism containers */
    .glass-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: var(--shadow);
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-container:hover {
        transform: translateY(-4px);
        box-shadow: 0 35px 60px -12px rgba(99, 102, 241, 0.4);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    /* Title with gradient text */
    .hero-title {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: -0.025em;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 20px rgba(99, 102, 241, 0.5)); }
        to { filter: drop-shadow(0 0 30px rgba(139, 92, 246, 0.8)); }
    }
    
    .hero-subtitle {
        text-align: center;
        color: var(--text-muted);
        font-size: 1.25rem;
        font-weight: 400;
        margin-bottom: 3rem;
        animation: fadeInUp 0.8s ease-out 0.2s both;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(45deg, var(--primary), var(--primary-light));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.4);
        position: relative;
        overflow: hidden;
        width: 100%;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(99, 102, 241, 0.6);
    }
    
    /* Sidebar enhancements */
    .css-1d391kg {
        background: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.1) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary);
        margin: 0;
        position: relative;
        z-index: 1;
    }
    
    .metric-label {
        color: var(--text-muted);
        font-size: 0.9rem;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        border-radius: 10px;
        height: 12px;
    }
    
    .stProgress > div > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        height: 12px;
    }
    
    /* Agent personality cards */
    .agent-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid var(--primary);
        transition: all 0.3s ease;
        position: relative;
    }
    
    .agent-card:hover {
        background: rgba(255, 255, 255, 0.08);
        transform: translateX(8px);
        box-shadow: 0 15px 30px rgba(99, 102, 241, 0.2);
    }
    
    /* Status badges */
    .status-badge {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        margin: 0.25rem;
    }
    
    .status-success {
        background: rgba(16, 185, 129, 0.2);
        color: var(--success);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.2);
        color: var(--warning);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .status-error {
        background: rgba(239, 68, 68, 0.2);
        color: var(--error);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Loading animation */
    .loading-spinner {
        border: 4px solid rgba(255, 255, 255, 0.1);
        border-top: 4px solid var(--primary);
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Table enhancements */
    .dataframe {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .hero-subtitle {
            font-size: 1rem;
        }
        
        .glass-container {
            padding: 1rem;
        }
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-light);
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Agent class with more sophisticated AI
class EnhancedGameAgent:
    def __init__(self, name, personality_type, difficulty="Normal"):
        self.name = name
        self.personality = personality_type
        self.difficulty = difficulty
        self.history = []
        self.opponent_history = []
        self.score = 0
        self.reasoning_history = []
        self.emotional_state = 0.5  # 0 = very negative, 1 = very positive
        self.trust_level = 0.5
        self.adaptation_rate = 0.1
        self.memory_window = 5
        
    def update_emotional_state(self, recent_payoffs):
        """Update emotional state based on recent performance"""
        if len(recent_payoffs) > 0:
            avg_payoff = sum(recent_payoffs) / len(recent_payoffs)
            if avg_payoff > 1:
                self.emotional_state = min(1.0, self.emotional_state + 0.1)
            elif avg_payoff < 0:
                self.emotional_state = max(0.0, self.emotional_state - 0.15)
    
    def update_trust_level(self, opponent_cooperations):
        """Update trust level based on opponent's cooperation rate"""
        if len(self.opponent_history) > 0:
            coop_rate = opponent_cooperations / len(self.opponent_history)
            target_trust = coop_rate
            self.trust_level += self.adaptation_rate * (target_trust - self.trust_level)
    
    def make_decision(self, round_num):
        """Enhanced decision making with emotional and trust factors"""
        
        # Update emotional state and trust
        recent_payoffs = []
        if len(self.history) > 0:
            for i in range(max(0, len(self.history) - self.memory_window), len(self.history)):
                my_payoff, _ = self.get_payoff(self.history[i], self.opponent_history[i])
                recent_payoffs.append(my_payoff)
        
        self.update_emotional_state(recent_payoffs)
        
        opponent_cooperations = self.opponent_history.count("C")
        self.update_trust_level(opponent_cooperations)
        
        # Decision logic based on personality
        if self.personality == "Rational":
            return self._enhanced_rational_decision(round_num)
        elif self.personality == "Cooperative":
            return self._enhanced_cooperative_decision(round_num)
        elif self.personality == "Suspicious":
            return self._enhanced_suspicious_decision(round_num)
        elif self.personality == "Emotional":
            return self._enhanced_emotional_decision(round_num)
        elif self.personality == "Adaptive":
            return self._enhanced_adaptive_decision(round_num)
        elif self.personality == "Aggressive":
            return self._aggressive_decision(round_num)
        elif self.personality == "Pacifist":
            return self._pacifist_decision(round_num)
        else:
            return self._random_decision()
    
    def _enhanced_rational_decision(self, round_num):
        """Enhanced rational strategy with game theory concepts"""
        if len(self.opponent_history) == 0:
            move = "C"
            reason = "🎯 Starting with cooperation to establish baseline"
        else:
            # Calculate expected values
            coop_rate = self.opponent_history.count("C") / len(self.opponent_history)
            
            # Expected payoff for cooperation
            expected_coop = coop_rate * 2 + (1 - coop_rate) * (-1)
            # Expected payoff for defection
            expected_defect = coop_rate * 3 + (1 - coop_rate) * 0
            
            if expected_coop > expected_defect:
                move = "C"
                reason = f"📊 Expected cooperation payoff ({expected_coop:.2f}) > defection ({expected_defect:.2f})"
            else:
                move = "D"
                reason = f"📊 Expected defection payoff ({expected_defect:.2f}) > cooperation ({expected_coop:.2f})"
        
        return move, reason
    
    def _enhanced_cooperative_decision(self, round_num):
        """Enhanced cooperative strategy with forgiveness"""
        if len(self.opponent_history) == 0:
            move = "C"
            reason = "🤝 Starting with trust and cooperation"
        else:
            # Forgiveness mechanism
            recent_betrayals = self.opponent_history[-3:].count("D")
            total_betrayals = self.opponent_history.count("D")
            
            if self.trust_level > 0.3 and recent_betrayals < 2:
                move = "C"
                reason = f"💚 Maintaining cooperation (Trust: {self.trust_level:.2f})"
            elif total_betrayals < len(self.opponent_history) * 0.7:
                move = "C"
                reason = f"🕊️ Forgiving occasional defection ({total_betrayals}/{len(self.opponent_history)} betrayals)"
            else:
                move = "D"
                reason = f"😔 Reluctantly defecting due to repeated betrayals"
        
        return move, reason
    
    def _enhanced_suspicious_decision(self, round_num):
        """Enhanced suspicious strategy with gradual trust building"""
        if len(self.opponent_history) == 0:
            move = "D"
            reason = "🛡️ Starting defensively to test opponent"
        else:
            consecutive_cooperations = 0
            for move in reversed(self.opponent_history):
                if move == "C":
                    consecutive_cooperations += 1
                else:
                    break
            
            if consecutive_cooperations >= 3 and self.trust_level > 0.6:
                move = "C"
                reason = f"🤔 Cautiously trusting after {consecutive_cooperations} consecutive cooperations"
            elif self.opponent_history[-1] == "D":
                move = "D"
                reason = "⚠️ Immediate retaliation for defection"
            else:
                move = "D"
                reason = f"🔒 Remaining defensive (Trust: {self.trust_level:.2f})"
        
        return move, reason
    
    def _enhanced_emotional_decision(self, round_num):
        """Enhanced emotional strategy with mood swings"""
        if len(self.opponent_history) == 0:
            move = "C"
            reason = "😊 Feeling optimistic about cooperation"
        else:
            # Emotional decision based on current state
            if self.emotional_state > 0.7:
                move = "C"
                reason = f"😄 Feeling great (Mood: {self.emotional_state:.2f}), spreading positivity"
            elif self.emotional_state < 0.3:
                move = "D"
                reason = f"😠 Feeling frustrated (Mood: {self.emotional_state:.2f}), acting defensively"
            else:
                # Neutral state - influenced by recent opponent behavior
                if self.opponent_history[-1] == "C":
                    move = "C"
                    reason = f"😐 Neutral mood, reciprocating cooperation"
                else:
                    move = "D"
                    reason = f"😤 Neutral mood, but hurt by defection"
        
        return move, reason
    
    def _enhanced_adaptive_decision(self, round_num):
        """Enhanced adaptive strategy with pattern recognition"""
        if len(self.opponent_history) < 3:
            move = "C"
            reason = "🔄 Learning phase: starting cooperatively"
        else:
            # Pattern recognition
            recent_pattern = self.opponent_history[-3:]
            
            # Look for patterns
            if recent_pattern == ["C", "C", "C"]:
                move = "C"
                reason = "🎯 Detected cooperation pattern, reciprocating"
            elif recent_pattern == ["D", "D", "D"]:
                move = "D"
                reason = "🎯 Detected defection pattern, defending"
            elif recent_pattern == ["C", "D", "C"]:
                move = "C"
                reason = "🎯 Detected alternating pattern, testing cooperation"
            else:
                # Tit-for-tat with generous forgiveness
                if self.opponent_history[-1] == "C" or random.random() < 0.1:
                    move = "C"
                    reason = f"🤖 Generous tit-for-tat: {'reciprocating' if self.opponent_history[-1] == 'C' else 'forgiving'}"
                else:
                    move = "D"
                    reason = "🤖 Tit-for-tat: retaliating for defection"
        
        return move, reason
    
    def _aggressive_decision(self, round_num):
        """Aggressive strategy focused on winning"""
        if len(self.opponent_history) == 0:
            move = "D"
            reason = "⚡ Aggressive start to establish dominance"
        else:
            # Mostly defect, but cooperate if opponent seems consistently cooperative
            coop_rate = self.opponent_history.count("C") / len(self.opponent_history)
            
            if coop_rate > 0.8 and len(self.opponent_history) > 5:
                move = "C"
                reason = f"⚡ Exploiting highly cooperative opponent ({coop_rate:.1%} cooperation)"
            else:
                move = "D"
                reason = f"⚡ Maintaining aggressive stance"
        
        return move, reason
    
    def _pacifist_decision(self, round_num):
        """Pacifist strategy - almost always cooperate"""
        if len(self.opponent_history) == 0:
            move = "C"
            reason = "🕊️ Peaceful approach to build harmony"
        else:
            # Almost always cooperate, only defect in extreme cases
            defection_rate = self.opponent_history.count("D") / len(self.opponent_history)
            
            if defection_rate > 0.9 and len(self.opponent_history) > 10:
                move = "D"
                reason = f"🕊️ Reluctantly defending against consistent aggression ({defection_rate:.1%} defection)"
            else:
                move = "C"
                reason = f"🕊️ Maintaining peaceful stance despite challenges"
        
        return move, reason
    
    def _random_decision(self):
        """Random strategy with reasoning"""
        move = random.choice(["C", "D"])
        reason = f"🎲 Random decision: {'cooperating' if move == 'C' else 'defecting'}"
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

# Enhanced simulation function with real-time updates
def run_enhanced_simulation(agent_a_type, agent_b_type, num_rounds, experiment_type="Basic", speed="Normal"):
    """Run enhanced simulation with real-time updates"""
    
    # Initialize agents
    agent_a = EnhancedGameAgent("Agent A", agent_a_type)
    agent_b = EnhancedGameAgent("Agent B", agent_b_type)
    
    # Game history
    game_history = []
    
    # Real-time display containers
    progress_container = st.empty()
    metrics_container = st.empty()
    chart_container = st.empty()
    
    # Speed settings
    delay_map = {"Fast": 0.1, "Normal": 0.3, "Slow": 0.8}
    delay = delay_map.get(speed, 0.3)
    
    for round_num in range(1, num_rounds + 1):
        # Update progress
        progress = round_num / num_rounds
        progress_container.progress(progress, text=f"Round {round_num}/{num_rounds}")
        
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
            "B_Reasoning": b_reason,
            "A_Emotional_State": agent_a.emotional_state,
            "B_Emotional_State": agent_b.emotional_state,
            "A_Trust_Level": agent_a.trust_level,
            "B_Trust_Level": agent_b.trust_level
        })
        
        # Real-time metrics update
        if round_num % 5 == 0 or round_num == num_rounds:
            current_df = pd.DataFrame(game_history)
            
            # Update metrics
            col1, col2, col3, col4 = metrics_container.columns(4)
            with col1:
                st.metric("Agent A Score", agent_a.score)
            with col2:
                st.metric("Agent B Score", agent_b.score)
            with col3:
                cooperation_rate = (current_df['A_Move'].value_counts().get('Cooperate', 0) + 
                                  current_df['B_Move'].value_counts().get('Cooperate', 0)) / (len(current_df) * 2)
                st.metric("Cooperation Rate", f"{cooperation_rate:.1%}")
            with col4:
                st.metric("Rounds Completed", round_num)
            
            # Real-time chart update
            if len(game_history) > 1:
                fig = create_realtime_chart(current_df)
                chart_container.plotly_chart(fig, use_container_width=True)
        
        # Add delay for visualization
        if speed != "Instant":
            time.sleep(delay)
    
    # Clear progress
    progress_container.empty()
    
    return pd.DataFrame(game_history), agent_a, agent_b

def create_realtime_chart(df):
    """Create real-time updating chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Round'],
        y=df['A_Cumulative'],
        mode='lines+markers',
        name='Agent A',
        line=dict(color='#6366f1', width=2),
        marker=dict(size=4)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Round'],
        y=df['B_Cumulative'],
        mode='lines+markers',
        name='Agent B',
        line=dict(color='#8b5cf6', width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title="Real-time Score Tracking",
        xaxis_title="Round",
        yaxis_title="Score",
        template="plotly_dark",
        height=300,
        showlegend=True,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_advanced_charts(df):
    """Create advanced visualization charts"""
    
    # 1. Emotional state over time
    fig_emotions = go.Figure()
    
    fig_emotions.add_trace(go.Scatter(
        x=df['Round'],
        y=df['A_Emotional_State'],
        mode='lines+markers',
        name='Agent A Emotion',
        line=dict(color='#10b981', width=2),
        yaxis='y'
    ))
    
    fig_emotions.add_trace(go.Scatter(
        x=df['Round'],
        y=df['B_Emotional_State'],
        mode='lines+markers',
        name='Agent B Emotion',
        line=dict(color='#f59e0b', width=2),
        yaxis='y'
    ))
    
    fig_emotions.add_trace(go.Scatter(
        x=df['Round'],
        y=df['A_Trust_Level'],
        mode='lines+markers',
        name='Agent A Trust',
        line=dict(color='#6366f1', width=2, dash='dash'),
        yaxis='y'
    ))
    
    fig_emotions.add_trace(go.Scatter(
        x=df['Round'],
        y=df['B_Trust_Level'],
        mode='lines+markers',
        name='Agent B Trust',
        line=dict(color='#8b5cf6', width=2, dash='dash'),
        yaxis='y'
    ))
    
    fig_emotions.update_layout(
        title="Emotional State & Trust Levels Over Time",
        xaxis_title="Round",
        yaxis_title="Level (0-1)",
        template="plotly_dark",
        height=400,
        showlegend=True
    )
    
    # 2. Strategy effectiveness heatmap
    rounds = df['Round'].values
    a_payoffs = df['A_Payoff'].values
    b_payoffs = df['B_Payoff'].values
    
    fig_heatmap = go.Figure()
    
    fig_heatmap.add_trace(go.Heatmap(
        z=[a_payoffs, b_payoffs],
        x=rounds,
        y=['Agent A', 'Agent B'],
        colorscale='RdYlBu',
        colorbar=dict(title="Payoff"),
        hoverongaps=False
    ))
    
    fig_heatmap.update_layout(
        title="Payoff Heatmap",
        xaxis_title="Round",
        yaxis_title="Agent",
        template="plotly_dark",
        height=250
    )
    
    return fig_emotions, fig_heatmap

def main():
    # Hero section
    st.markdown('<h1 class="hero-title">🎮 Game Theory AI Lab</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Advanced strategic decision-making with intelligent AI agents</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Agent selection with expanded options
        st.markdown("### 🤖 Agent Selection")
        agent_types = ["Rational", "Cooperative", "Suspicious", "Emotional", "Adaptive", "Aggressive", "Pacifist"]
        
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
            min_value=10,
            max_value=100,
            value=30,
            help="Number of rounds to simulate"
        )
        
        experiment_type = st.selectbox(
            "Experiment Type:",
            ["Basic", "Moral", "Fear", "Hope", "Competitive"],
            help="Choose the experimental context"
        )
        
        speed = st.selectbox(
            "Simulation Speed:",
            ["Fast", "Normal", "Slow", "Instant"],
            index=1,
            help="Speed of simulation visualization"
        )
        
        # Advanced settings
        st.markdown("### 🔧 Advanced Settings")
        show_reasoning = st.checkbox("Show Agent Reasoning", value=True)
        show_emotions = st.checkbox("Show Emotional States", value=True)
        real_time_charts = st.checkbox("Real-time Charts", value=True)
        
        # Agent information
        st.markdown("### 📚 Agent Personalities")
        
        personality_descriptions = {
            "Rational": "🧠 Logic-based, calculates expected values",
            "Cooperative": "🤝 Trusting, forgiving, promotes mutual benefit",
            "Suspicious": "🛡️ Defensive, requires proof of trustworthiness",
            "Emotional": "😊 Mood-driven, affected by recent outcomes",
            "Adaptive": "🔄 Pattern recognition, learning strategy",
            "Aggressive": "⚡ Competitive, dominance-focused",
            "Pacifist": "🕊️ Peace-loving, avoids conflict"
        }
        
        for personality, description in personality_descriptions.items():
            st.markdown(f"**{personality}:** {description}")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown("## 🎲 Simulation Control")
        
        # Enhanced run button
        if st.button("🚀 Run Enhanced Simulation", use_container_width=True):
            with st.spinner("🔄 Initializing advanced AI agents..."):
                # Run the enhanced simulation
                df, agent_a, agent_b = run_enhanced_simulation(
                    agent_a_type, agent_b_type, num_rounds, experiment_type, speed
                )
                
                # Store results in session state
                st.session_state.df = df
                st.session_state.agent_a = agent_a
                st.session_state.agent_b = agent_b
                
                st.success("✅ Simulation completed successfully!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown("## 📊 Live Stats")
        
        if 'df' in st.session_state:
            df = st.session_state.df
            
            # Enhanced metrics with styling
            total_rounds = len(df)
            a_final_score = df['A_Cumulative'].iloc[-1]
            b_final_score = df['B_Cumulative'].iloc[-1]
            cooperation_rate = (df['A_Move'].value_counts().get('Cooperate', 0) + 
                              df['B_Move'].value_counts().get('Cooperate', 0)) / (total_rounds * 2)
            
            st.markdown(f'<div class="metric-card"><div class="metric-value">{total_rounds}</div><div class="metric-label">Total Rounds</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><div class="metric-value">{a_final_score:+}</div><div class="metric-label">Agent A Score</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><div class="metric-value">{b_final_score:+}</div><div class="metric-label">Agent B Score</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><div class="metric-value">{cooperation_rate:.1%}</div><div class="metric-label">Cooperation Rate</div></div>', unsafe_allow_html=True)
            
            # Winner determination
            if a_final_score > b_final_score:
                st.markdown('<div class="status-badge status-success">🏆 Agent A Wins!</div>', unsafe_allow_html=True)
            elif b_final_score > a_final_score:
                st.markdown('<div class="status-badge status-success">🏆 Agent B Wins!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-badge status-warning">🤝 Draw!</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced results section
    if 'df' in st.session_state:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown("## 📈 Advanced Analytics Dashboard")
        
        df = st.session_state.df
        
        # Create advanced visualizations
        fig_emotions, fig_heatmap = create_advanced_charts(df)
        
        # Display charts in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Scores", "🧠 Psychology", "🔥 Heatmap", "📋 Data"])
        
        with tab1:
            # Score visualization
            fig_scores = go.Figure()
            
            fig_scores.add_trace(go.Scatter(
                x=df['Round'],
                y=df['A_Cumulative'],
                mode='lines+markers',
                name='Agent A',
                line=dict(color='#6366f1', width=3),
                marker=dict(size=8, color='#6366f1')
            ))
            
            fig_scores.add_trace(go.Scatter(
                x=df['Round'],
                y=df['B_Cumulative'],
                mode='lines+markers',
                name='Agent B',
                line=dict(color='#8b5cf6', width=3),
                marker=dict(size=8, color='#8b5cf6')
            ))
            
            fig_scores.update_layout(
                title="Cumulative Score Progression",
                xaxis_title="Round",
                yaxis_title="Cumulative Score",
                template="plotly_dark",
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig_scores, use_container_width=True)
        
        with tab2:
            if show_emotions:
                st.plotly_chart(fig_emotions, use_container_width=True)
            else:
                st.info("Enable 'Show Emotional States' in sidebar to view psychological analysis")
        
        with tab3:
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with tab4:
            # Enhanced data table
            display_df = df[['Round', 'A_Move', 'B_Move', 'A_Payoff', 'B_Payoff', 'A_Cumulative', 'B_Cumulative']].copy()
            st.dataframe(display_df, use_container_width=True, height=400)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Agent reasoning section
        if show_reasoning:
            st.markdown('<div class="glass-container">', unsafe_allow_html=True)
            st.markdown("## 🧠 Agent Decision Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🤖 Agent A Latest Reasoning")
                for i, reasoning in enumerate(st.session_state.agent_a.reasoning_history[-5:]):
                    round_num = len(st.session_state.agent_a.reasoning_history) - 5 + i + 1
                    st.markdown(f'<div class="agent-card"><strong>Round {round_num}:</strong> {reasoning}</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 🤖 Agent B Latest Reasoning")
                for i, reasoning in enumerate(st.session_state.agent_b.reasoning_history[-5:]):
                    round_num = len(st.session_state.agent_b.reasoning_history) - 5 + i + 1
                    st.markdown(f'<div class="agent-card"><strong>Round {round_num}:</strong> {reasoning}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Export section
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown("## 💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
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
        
        with col3:
            # Generate summary report
            summary_report = f"""
# Game Theory Simulation Report

## Configuration
- Agent A: {agent_a_type}
- Agent B: {agent_b_type}
- Rounds: {len(df)}
- Experiment: {experiment_type}

## Results
- Agent A Final Score: {df['A_Cumulative'].iloc[-1]}
- Agent B Final Score: {df['B_Cumulative'].iloc[-1]}
- Cooperation Rate: {cooperation_rate:.1%}

## Analysis
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            st.download_button(
                label="📄 Download Report",
                data=summary_report,
                file_name=f"simulation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
