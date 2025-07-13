#!/usr/bin/env python3
"""
🆓 Simple Free AI Prisoner's Dilemma (No PyTorch Required)
Uses rule-based AI that simulates different reasoning patterns
"""

import streamlit as st
import pandas as pd
import numpy as np
import random
import altair as alt
from typing import Tuple
import time

# Set page config
st.set_page_config(page_title="Simple Free AI Prisoner's Dilemma", page_icon="🤖", layout="wide")

st.title("🤖 Simple Free AI Prisoner's Dilemma")
st.markdown("**Zero cost AI agents** using rule-based reasoning patterns - no external dependencies!")

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

# Sidebar configuration
st.sidebar.header("🎯 Agent Configuration")

agent_a_type = st.sidebar.selectbox(
    "Agent A Personality:",
    list(ai_personalities.keys()),
    format_func=lambda x: ai_personalities[x]["name"]
)

agent_b_type = st.sidebar.selectbox(
    "Agent B Personality:",
    list(ai_personalities.keys()),
    index=1,
    format_func=lambda x: ai_personalities[x]["name"]
)

# Display personality descriptions
st.sidebar.markdown(f"**Agent A:** {ai_personalities[agent_a_type]['description']}")
st.sidebar.markdown(f"**Agent B:** {ai_personalities[agent_b_type]['description']}")

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

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎮 Simulation Parameters")
    
    num_rounds = st.slider("Number of rounds:", min_value=5, max_value=100, value=30)
    
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
    st.header("📊 Agent Status")
    
    st.info(f"🤖 Agent A: {ai_personalities[agent_a_type]['name']}")
    st.info(f"🤖 Agent B: {ai_personalities[agent_b_type]['name']}")
    
    st.success("✅ No external dependencies required!")
    st.success("✅ Completely free to use!")

# Run simulation
if st.button("🚀 Run Simple AI Simulation"):
    with st.spinner("Running simulation with simple AI agents..."):
        results_df = run_simple_ai_simulation(
            agent_a_type,
            agent_b_type,
            num_rounds,
            experiment_type
        )
    
    st.success("🎉 Simulation completed!")
    
    # Display results
    st.subheader("📈 Results")
    
    # Results table
    st.dataframe(results_df[["Round", "A Move", "B Move", "A Payoff", "B Payoff", "A Cumulative", "B Cumulative"]])
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Cumulative Scores")
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
        file_name=f"simple_ai_results_{agent_a_type}_vs_{agent_b_type}_{experiment_type}.csv",
        mime="text/csv"
    )

# Information section
st.sidebar.header("ℹ️ About Simple AI")
st.sidebar.info("""
**Benefits:**
- ✅ Zero external dependencies
- ✅ No API costs
- ✅ No complex setup
- ✅ Intelligent behavior patterns
- ✅ Educational value
- ✅ Fast execution

**Personality Types:**
- Rational: Logic-based decisions
- Cooperative: Trusting and helpful
- Suspicious: Distrustful and defensive
- Emotional: Influenced by feelings
- Adaptive: Learns from opponent
""")

st.sidebar.header("🔧 How It Works")
st.sidebar.markdown("""
**Simple AI uses:**
- Personality-based decision making
- Historical behavior analysis
- Probabilistic reasoning
- Contextual prompt adjustment
- Emotional state simulation

**No external AI required!**
All intelligence is built into the rule-based system.
""")

if __name__ == "__main__":
    st.markdown("---")
    st.markdown("🤖 **Simple AI Prisoner's Dilemma** - Zero dependencies, maximum intelligence!")
    st.markdown("*Experience sophisticated AI behavior without external requirements!*")
