#!/usr/bin/env python3
"""
🤖 Rule-Based Prisoner's Dilemma Agents - Zero Cost Implementation

This script demonstrates how to create intelligent game theory agents
without any AI API costs using rule-based decision making.
"""

import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from typing import List, Tuple, Dict

class RuleBasedAgent:
    """Rule-based agent with different strategies"""
    
    def __init__(self, strategy: str = "tit_for_tat", personality: dict = None):
        self.strategy = strategy
        self.history = []
        self.opponent_history = []
        self.personality = personality or {}
        self.reputation = 0
        self.trust_level = 0.5
        self.fear_level = 0.3
        self.cooperation_tendency = 0.6
        
    def decide(self, round_number: int, context: dict = None) -> Tuple[str, str]:
        """Make a decision based on strategy and history"""
        
        if self.strategy == "always_cooperate":
            return "C", "I believe in cooperation and trust"
            
        elif self.strategy == "always_defect":
            return "D", "Self-interest is my priority"
            
        elif self.strategy == "tit_for_tat":
            if not self.opponent_history:
                return "C", "Starting with cooperation"
            last_opponent_move = self.opponent_history[-1]
            if last_opponent_move == "C":
                return "C", "Reciprocating cooperation"
            else:
                return "D", "Responding to betrayal"
                
        elif self.strategy == "generous_tit_for_tat":
            if not self.opponent_history:
                return "C", "Starting with cooperation"
            last_opponent_move = self.opponent_history[-1]
            if last_opponent_move == "C":
                return "C", "Reciprocating cooperation"
            else:
                # 20% chance to forgive
                if random.random() < 0.2:
                    return "C", "Forgiving betrayal"
                else:
                    return "D", "Responding to betrayal"
                    
        elif self.strategy == "grudger":
            if "D" in self.opponent_history:
                return "D", "Never forgetting betrayal"
            else:
                return "C", "Cooperating while trust remains"
                
        elif self.strategy == "pavlov":
            if not self.history:
                return "C", "Starting with cooperation"
            
            # Win-stay, lose-shift
            last_payoff = self.get_last_payoff()
            if last_payoff >= 2:  # Good outcome
                return self.history[-1], "Repeating successful strategy"
            else:  # Bad outcome
                return "D" if self.history[-1] == "C" else "C", "Switching strategy"
                
        elif self.strategy == "suspicious_tit_for_tat":
            if not self.opponent_history:
                return "D", "Starting with caution"
            last_opponent_move = self.opponent_history[-1]
            return last_opponent_move, "Mirroring opponent's last move"
            
        elif self.strategy == "adaptive":
            return self.adaptive_strategy()
            
        elif self.strategy == "random":
            move = random.choice(["C", "D"])
            return move, "Random decision"
            
        elif self.strategy == "fear_based":
            return self.fear_based_strategy()
            
        elif self.strategy == "reputation_based":
            return self.reputation_based_strategy()
            
        else:
            return "C", "Default cooperation"
    
    def adaptive_strategy(self) -> Tuple[str, str]:
        """Adaptive strategy based on opponent's behavior"""
        if len(self.opponent_history) < 3:
            return "C", "Learning opponent's pattern"
        
        recent_cooperation = self.opponent_history[-5:].count("C") / min(5, len(self.opponent_history))
        
        if recent_cooperation > 0.7:
            return "C", "Opponent seems cooperative"
        elif recent_cooperation < 0.3:
            return "D", "Opponent seems hostile"
        else:
            # Mixed strategy
            if random.random() < 0.6:
                return "C", "Taking calculated risk"
            else:
                return "D", "Protecting against exploitation"
    
    def fear_based_strategy(self) -> Tuple[str, str]:
        """Strategy influenced by fear and uncertainty"""
        if not self.opponent_history:
            return "D", "Fear of being betrayed first"
        
        recent_defections = self.opponent_history[-3:].count("D")
        
        if recent_defections >= 2:
            self.fear_level = min(1.0, self.fear_level + 0.2)
            return "D", "High fear due to recent betrayals"
        elif recent_defections == 0:
            self.fear_level = max(0.0, self.fear_level - 0.1)
            return "C", "Fear subsiding, attempting cooperation"
        else:
            if random.random() < self.fear_level:
                return "D", "Moderate fear influencing decision"
            else:
                return "C", "Overcoming fear to cooperate"
    
    def reputation_based_strategy(self) -> Tuple[str, str]:
        """Strategy based on reputation system"""
        if not self.opponent_history:
            return "C", "Starting with neutral reputation assumption"
        
        # Update trust based on opponent's history
        cooperation_rate = self.opponent_history.count("C") / len(self.opponent_history)
        
        if cooperation_rate > 0.7:
            return "C", "High trust due to good reputation"
        elif cooperation_rate < 0.3:
            return "D", "Low trust due to poor reputation"
        else:
            # Probabilistic decision based on reputation
            if random.random() < cooperation_rate:
                return "C", "Moderate trust, taking risk"
            else:
                return "D", "Moderate trust, playing safe"
    
    def get_last_payoff(self) -> int:
        """Get the payoff from the last round"""
        if not self.history or not self.opponent_history:
            return 0
        
        my_last = self.history[-1]
        opp_last = self.opponent_history[-1]
        
        if my_last == "C" and opp_last == "C":
            return 2
        elif my_last == "C" and opp_last == "D":
            return -1
        elif my_last == "D" and opp_last == "C":
            return 3
        else:
            return 0
    
    def update_history(self, my_move: str, opponent_move: str):
        """Update agent's history"""
        self.history.append(my_move)
        self.opponent_history.append(opponent_move)

def payoff(a_move: str, b_move: str) -> Tuple[int, int]:
    """Calculate payoffs for both agents"""
    if a_move == "C" and b_move == "C":
        return 2, 2  # Mutual cooperation
    elif a_move == "C" and b_move == "D":
        return -1, 3  # A betrayed
    elif a_move == "D" and b_move == "C":
        return 3, -1  # B betrayed
    else:
        return 0, 0  # Mutual defection

def run_rule_based_simulation(strategy_a: str, strategy_b: str, num_rounds: int = 100) -> pd.DataFrame:
    """Run simulation between two rule-based agents"""
    
    agent_a = RuleBasedAgent(strategy_a)
    agent_b = RuleBasedAgent(strategy_b)
    
    a_score = 0
    b_score = 0
    history = []
    
    for round_num in range(1, num_rounds + 1):
        # Agents make decisions
        a_move, a_reason = agent_a.decide(round_num)
        b_move, b_reason = agent_b.decide(round_num)
        
        # Calculate payoffs
        a_payoff, b_payoff = payoff(a_move, b_move)
        a_score += a_payoff
        b_score += b_payoff
        
        # Update histories
        agent_a.update_history(a_move, b_move)
        agent_b.update_history(b_move, a_move)
        
        # Record round
        history.append({
            "Round": round_num,
            "A_Strategy": strategy_a,
            "B_Strategy": strategy_b,
            "A_Move": "Cooperate" if a_move == "C" else "Defect",
            "B_Move": "Cooperate" if b_move == "C" else "Defect",
            "A_Payoff": a_payoff,
            "B_Payoff": b_payoff,
            "A_Cumulative": a_score,
            "B_Cumulative": b_score,
            "A_Reason": a_reason,
            "B_Reason": b_reason
        })
    
    return pd.DataFrame(history)

def tournament_simulation(strategies: List[str], rounds_per_match: int = 50) -> pd.DataFrame:
    """Run a tournament between multiple strategies"""
    
    results = []
    
    for i, strategy_a in enumerate(strategies):
        for j, strategy_b in enumerate(strategies):
            if i <= j:  # Avoid duplicate matches
                print(f"🥊 {strategy_a} vs {strategy_b}")
                
                # Run simulation
                df = run_rule_based_simulation(strategy_a, strategy_b, rounds_per_match)
                
                # Calculate final scores
                final_a = df['A_Cumulative'].iloc[-1]
                final_b = df['B_Cumulative'].iloc[-1]
                
                # Calculate cooperation rates
                coop_a = df['A_Move'].value_counts().get('Cooperate', 0) / len(df)
                coop_b = df['B_Move'].value_counts().get('Cooperate', 0) / len(df)
                
                results.append({
                    'Strategy_A': strategy_a,
                    'Strategy_B': strategy_b,
                    'Score_A': final_a,
                    'Score_B': final_b,
                    'Cooperation_A': coop_a,
                    'Cooperation_B': coop_b,
                    'Total_Score': final_a + final_b,
                    'Rounds': rounds_per_match
                })
    
    return pd.DataFrame(results)

def visualize_results(df: pd.DataFrame, title: str = "Rule-Based Prisoner's Dilemma"):
    """Visualize simulation results"""
    
    plt.style.use('dark_background')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, color='white')
    
    # 1. Cumulative scores
    ax1.plot(df['Round'], df['A_Cumulative'], label='Agent A', color='#4F86F7', linewidth=2)
    ax1.plot(df['Round'], df['B_Cumulative'], label='Agent B', color='#A020F0', linewidth=2)
    ax1.set_title('Cumulative Scores', color='white')
    ax1.set_xlabel('Round', color='white')
    ax1.set_ylabel('Score', color='white')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Move distribution
    a_moves = df['A_Move'].value_counts()
    b_moves = df['B_Move'].value_counts()
    
    x = np.arange(len(a_moves))
    width = 0.35
    
    ax2.bar(x - width/2, a_moves.values, width, label='Agent A', color='#4F86F7')
    ax2.bar(x + width/2, b_moves.values, width, label='Agent B', color='#A020F0')
    ax2.set_title('Move Distribution', color='white')
    ax2.set_xlabel('Move Type', color='white')
    ax2.set_ylabel('Count', color='white')
    ax2.set_xticks(x)
    ax2.set_xticklabels(a_moves.index)
    ax2.legend()
    
    # 3. Payoff per round
    ax3.bar(df['Round'], df['A_Payoff'], alpha=0.7, label='Agent A', color='#4F86F7')
    ax3.bar(df['Round'], df['B_Payoff'], alpha=0.7, label='Agent B', color='#A020F0')
    ax3.set_title('Payoff per Round', color='white')
    ax3.set_xlabel('Round', color='white')
    ax3.set_ylabel('Payoff', color='white')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Cooperation over time (rolling average)
    window = 10
    a_coop = (df['A_Move'] == 'Cooperate').rolling(window=window).mean()
    b_coop = (df['B_Move'] == 'Cooperate').rolling(window=window).mean()
    
    ax4.plot(df['Round'], a_coop, label='Agent A', color='#4F86F7', linewidth=2)
    ax4.plot(df['Round'], b_coop, label='Agent B', color='#A020F0', linewidth=2)
    ax4.set_title(f'Cooperation Rate (Rolling {window})', color='white')
    ax4.set_xlabel('Round', color='white')
    ax4.set_ylabel('Cooperation Rate', color='white')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main demonstration of rule-based agents"""
    
    print("🤖 Rule-Based Prisoner's Dilemma Agents")
    print("=" * 50)
    print("✅ Zero cost implementation")
    print("✅ No API keys required")
    print("✅ Intelligent strategies")
    print("✅ Educational and research ready")
    print()
    
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
        "reputation_based"
    ]
    
    print("🎯 Available Strategies:")
    for i, strategy in enumerate(strategies, 1):
        print(f"{i:2d}. {strategy.replace('_', ' ').title()}")
    
    print("\n🥊 Running Sample Simulations:")
    
    # Example 1: Tit-for-Tat vs Always Cooperate
    print("\n1. Tit-for-Tat vs Always Cooperate")
    df1 = run_rule_based_simulation("tit_for_tat", "always_cooperate", 50)
    print(f"   Final scores: TfT={df1['A_Cumulative'].iloc[-1]}, AC={df1['B_Cumulative'].iloc[-1]}")
    
    # Example 2: Adaptive vs Fear-based
    print("\n2. Adaptive vs Fear-based")
    df2 = run_rule_based_simulation("adaptive", "fear_based", 50)
    print(f"   Final scores: Adaptive={df2['A_Cumulative'].iloc[-1]}, Fear={df2['B_Cumulative'].iloc[-1]}")
    
    # Example 3: Tournament
    print("\n3. Mini Tournament")
    tournament_strategies = ["tit_for_tat", "always_cooperate", "always_defect", "adaptive"]
    tournament_results = tournament_simulation(tournament_strategies, 30)
    
    print("\n📊 Tournament Results:")
    for _, row in tournament_results.iterrows():
        print(f"   {row['Strategy_A']} vs {row['Strategy_B']}: {row['Score_A']}-{row['Score_B']}")
    
    # Visualize one simulation
    print("\n📈 Generating visualization...")
    visualize_results(df1, "Tit-for-Tat vs Always Cooperate")
    
    # Save results
    df1.to_csv("rule_based_results.csv", index=False)
    tournament_results.to_csv("tournament_results.csv", index=False)
    
    print("\n💾 Results saved to:")
    print("   - rule_based_results.csv")
    print("   - tournament_results.csv")
    
    print("\n🎉 Demo complete! No AI API costs incurred.")

if __name__ == "__main__":
    main()
