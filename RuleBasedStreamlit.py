import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from typing import List, Tuple, Dict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import the RuleBasedAgent class
from RuleBasedAgents import RuleBasedAgent, payoff, run_rule_based_simulation, tournament_simulation

# Set page config
st.set_page_config(page_title="Rule-Based Prisoner's Dilemma", page_icon="🤖", layout="wide")

# Title and description
st.title("🤖 Rule-Based Prisoner's Dilemma Agents")
st.markdown("**Zero cost intelligent agents** - No AI API required, pure strategy-based decisions!")

# Sidebar configuration
st.sidebar.header("🎯 Agent Configuration")

# Available strategies
strategies = [
    "tit_for_tat",
    "generous_tit_for_tat", 
    "always_cooperate",
    "always_defect",
    "grudger",
    "pavlov",
    "adaptive",
    "fear_based",
    "reputation_based",
    "random"
]

# Strategy descriptions
strategy_descriptions = {
    "tit_for_tat": "Starts cooperative, then copies opponent's last move",
    "generous_tit_for_tat": "Like Tit-for-Tat but occasionally forgives betrayals",
    "always_cooperate": "Always cooperates regardless of opponent",
    "always_defect": "Always defects regardless of opponent",
    "grudger": "Cooperates until first betrayal, then always defects",
    "pavlov": "Win-stay, lose-shift strategy",
    "adaptive": "Adapts based on opponent's cooperation rate",
    "fear_based": "Decisions influenced by fear and uncertainty",
    "reputation_based": "Decisions based on opponent's reputation",
    "random": "Random choices between cooperate and defect"
}

# Agent selection
agent_a_strategy = st.sidebar.selectbox(
    "Agent A Strategy:",
    strategies,
    index=0,
    format_func=lambda x: x.replace('_', ' ').title()
)

agent_b_strategy = st.sidebar.selectbox(
    "Agent B Strategy:",
    strategies,
    index=1,
    format_func=lambda x: x.replace('_', ' ').title()
)

# Display strategy descriptions
st.sidebar.markdown(f"**Agent A ({agent_a_strategy}):** {strategy_descriptions[agent_a_strategy]}")
st.sidebar.markdown(f"**Agent B ({agent_b_strategy}):** {strategy_descriptions[agent_b_strategy]}")

# Simulation parameters
st.sidebar.header("⚙️ Simulation Parameters")
num_rounds = st.sidebar.slider("Number of rounds:", min_value=10, max_value=200, value=50)
show_reasoning = st.sidebar.checkbox("Show agent reasoning", value=True)

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    st.header("🎮 Single Match Simulation")
    
    if st.button("🚀 Run Single Match"):
        with st.spinner("Running simulation..."):
            # Run simulation
            results_df = run_rule_based_simulation(agent_a_strategy, agent_b_strategy, num_rounds)
            
            # Display results
            st.success(f"✅ Simulation completed! {num_rounds} rounds played.")
            
            # Final scores
            final_a = results_df['A_Cumulative'].iloc[-1]
            final_b = results_df['B_Cumulative'].iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Agent A Score", final_a)
            col2.metric("Agent B Score", final_b)
            col3.metric("Winner", 
                       "Agent A" if final_a > final_b else "Agent B" if final_b > final_a else "Tie")
            
            # Visualizations
            st.subheader("📊 Results Visualization")
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Cumulative Scores", "Move Distribution", 
                               "Payoff per Round", "Cooperation Rate"),
                specs=[[{"secondary_y": False}, {"type": "bar"}],
                       [{"type": "bar"}, {"secondary_y": False}]]
            )
            
            # 1. Cumulative scores
            fig.add_trace(
                go.Scatter(x=results_df['Round'], y=results_df['A_Cumulative'], 
                          name='Agent A', line=dict(color='#4F86F7')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=results_df['Round'], y=results_df['B_Cumulative'], 
                          name='Agent B', line=dict(color='#A020F0')),
                row=1, col=1
            )
            
            # 2. Move distribution
            a_moves = results_df['A_Move'].value_counts()
            b_moves = results_df['B_Move'].value_counts()
            
            fig.add_trace(
                go.Bar(x=a_moves.index, y=a_moves.values, name='Agent A', 
                       marker_color='#4F86F7'),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(x=b_moves.index, y=b_moves.values, name='Agent B', 
                       marker_color='#A020F0'),
                row=1, col=2
            )
            
            # 3. Payoff per round
            fig.add_trace(
                go.Bar(x=results_df['Round'], y=results_df['A_Payoff'], 
                       name='Agent A Payoff', marker_color='#4F86F7', opacity=0.7),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(x=results_df['Round'], y=results_df['B_Payoff'], 
                       name='Agent B Payoff', marker_color='#A020F0', opacity=0.7),
                row=2, col=1
            )
            
            # 4. Cooperation rate
            window = min(10, num_rounds // 5)
            a_coop = (results_df['A_Move'] == 'Cooperate').rolling(window=window).mean()
            b_coop = (results_df['B_Move'] == 'Cooperate').rolling(window=window).mean()
            
            fig.add_trace(
                go.Scatter(x=results_df['Round'], y=a_coop, 
                          name='Agent A Cooperation', line=dict(color='#4F86F7')),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=results_df['Round'], y=b_coop, 
                          name='Agent B Cooperation', line=dict(color='#A020F0')),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.subheader("📋 Detailed Results")
            display_columns = ["Round", "A_Move", "B_Move", "A_Payoff", "B_Payoff", "A_Cumulative", "B_Cumulative"]
            if show_reasoning:
                display_columns.extend(["A_Reason", "B_Reason"])
            
            st.dataframe(results_df[display_columns])
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name=f"rule_based_results_{agent_a_strategy}_vs_{agent_b_strategy}.csv",
                mime="text/csv"
            )

with col2:
    st.header("📈 Quick Stats")
    
    # Strategy effectiveness info
    st.subheader("🎯 Strategy Tips")
    st.info("""
    **Best Strategies:**
    - Tit-for-Tat: Classic reciprocal
    - Generous TfT: Forgiving variant
    - Adaptive: Learns opponent patterns
    
    **Exploitable:**
    - Always Cooperate: Too trusting
    - Always Defect: Predictable
    - Random: No strategy
    """)
    
    # Payoff matrix reminder
    st.subheader("💰 Payoff Matrix")
    st.markdown("""
    |   | C | D |
    |---|---|---|
    | **C** | 2,2 | -1,3 |
    | **D** | 3,-1 | 0,0 |
    """)

# Tournament section
st.header("🏆 Tournament Mode")
st.markdown("Run all strategies against each other to see which performs best overall.")

col1, col2 = st.columns([2, 1])

with col1:
    tournament_strategies = st.multiselect(
        "Select strategies for tournament:",
        strategies,
        default=["tit_for_tat", "generous_tit_for_tat", "always_cooperate", "always_defect"],
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    tournament_rounds = st.slider("Rounds per match:", min_value=20, max_value=100, value=50)

with col2:
    if st.button("🏁 Run Tournament"):
        if len(tournament_strategies) < 2:
            st.error("Please select at least 2 strategies for tournament!")
        else:
            with st.spinner("Running tournament..."):
                tournament_results = tournament_simulation(tournament_strategies, tournament_rounds)
                
                st.success("🎉 Tournament completed!")
                
                # Tournament results
                st.subheader("🏆 Tournament Results")
                st.dataframe(tournament_results)
                
                # Calculate overall rankings
                strategy_scores = {}
                for strategy in tournament_strategies:
                    total_score = 0
                    matches = 0
                    
                    for _, row in tournament_results.iterrows():
                        if row['Strategy_A'] == strategy:
                            total_score += row['Score_A']
                            matches += 1
                        elif row['Strategy_B'] == strategy:
                            total_score += row['Score_B']
                            matches += 1
                    
                    strategy_scores[strategy] = total_score / matches if matches > 0 else 0
                
                # Display rankings
                st.subheader("📊 Overall Rankings")
                ranking_df = pd.DataFrame(list(strategy_scores.items()), 
                                        columns=['Strategy', 'Average Score'])
                ranking_df = ranking_df.sort_values('Average Score', ascending=False)
                
                fig = px.bar(ranking_df, x='Strategy', y='Average Score', 
                           title='Tournament Strategy Rankings')
                st.plotly_chart(fig, use_container_width=True)
                
                # Download tournament results
                csv = tournament_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Tournament Results",
                    data=csv,
                    file_name="tournament_results.csv",
                    mime="text/csv"
                )

# Information section
st.sidebar.header("ℹ️ About Rule-Based Agents")
st.sidebar.success("""
**Benefits:**
- ✅ Zero API costs
- ✅ No rate limits
- ✅ Deterministic behavior
- ✅ Educational value
- ✅ Fast execution
- ✅ Complete transparency
""")

st.sidebar.header("🔬 Research Applications")
st.sidebar.markdown("""
**Use Cases:**
- Game theory education
- Strategy comparison
- Baseline for AI agents
- Behavioral modeling
- Tournament analysis
""")

# Footer
st.markdown("---")
st.markdown("🤖 **Rule-Based Prisoner's Dilemma** - Pure strategy, zero cost, maximum insight!")
