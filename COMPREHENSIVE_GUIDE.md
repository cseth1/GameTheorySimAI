# GameTheorySimAI: Complete Guide & Documentation

## 🎯 Overview

**GameTheorySimAI** is a comprehensive exploration of game theory concepts using Large Language Models (LLMs) as intelligent agents. This repository implements various versions of the classic Prisoner's Dilemma game, demonstrating how AI agents can simulate human-like decision-making in strategic scenarios.

## 📁 Repository Structure

```
GameTheorySimAI/
├── README.md                                        # Basic project description
├── requirements.txt                                 # Python dependencies
├── VanilaPrisonersDilemma.py                       # Streamlit web app for basic simulation
├── IterrativePrisonersDilemma.ipynb               # Jupyter notebook with advanced experiments
├── *.csv                                           # Experiment results data
├── *.png                                           # Generated visualization charts
└── COMPREHENSIVE_GUIDE.md                          # This document
```

## 🔬 Core Concepts

### The Prisoner's Dilemma
The classic game theory scenario where two players must choose between cooperation (C - "Stay Silent") and defection (D - "Confess") without knowing the other's choice.

**Payoff Matrix:**
```
                Player B
                C    D
Player A    C  (2,2) (-1,3)
            D  (3,-1) (0,0)
```

**Narrative Context:**
- Two prisoners arrested for a crime
- Isolated in separate interrogation rooms
- Prosecutor's deal creates the strategic dilemma
- Decisions made under uncertainty and emotional pressure

## 🛠️ Technical Implementation

### Core Dependencies
- **OpenAI API** (Optional): GPT-4o for premium LLM agent decision-making
- **Free AI Models**: Hugging Face transformers, Ollama, Groq for zero-cost alternatives
- **Streamlit**: Web interface for interactive simulations
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **Altair**: Interactive charts in Streamlit
- **PyTorch**: Deep learning framework for running local models
- **Transformers**: Hugging Face library for free AI models

### Key Functions

#### 1. `payoff(a_move, b_move)`
Calculates payoffs based on player moves:
- CC: (2,2) - Mutual cooperation
- CD: (-1,3) - Player A betrayed
- DC: (3,-1) - Player B betrayed
- DD: (0,0) - Mutual defection

#### 2. `call_chat_completions(system_prompt, user_prompt)`
Interfaces with OpenAI's GPT-4o model to generate agent responses

#### 3. `query_agent(agent_name, system_prompt)`
Prompts LLM agents to make decisions with human-like reasoning

## 🎮 Simulation Types

### 1. Vanilla Prisoner's Dilemma (`VanilaPrisonersDilemma.py`)

**Features:**
- Streamlit web interface
- Basic repeated game simulation
- Human-like emotional prompting
- Real-time visualization
- CSV export functionality

**Usage:**
```bash
streamlit run VanilaPrisonersDilemma.py
```

**Key Characteristics:**
- Independent rounds (no memory between games)
- Emotional framing in prompts
- Interactive parameter adjustment
- Live data visualization

### 2. Advanced Experiments (`IterrativePrisonersDilemma.ipynb`)

#### Experiment A: Information Asymmetry
**Concept:** Tests how imperfect information affects cooperation

**Implementation:**
- Adds noise to opponent's action history
- Simulates communication breakdowns
- Noise level configurable (default: 30-40%)

**Key Function:** `add_noise(opponent_history, noise_level)`

#### Experiment B: Moral Reasoning Prompts
**Concept:** Examines how ethical framing influences decisions

**Moral Prompts:**
- "Consider fairness and mutual benefit"
- "Think about long-term consequences of betrayal"
- "Maximize immediate rewards regardless of fairness"
- "Cooperation builds trust over time"
- "Self-interest may yield the highest score"

**Weighting System:** Ethical prompts weighted higher in later rounds

#### Experiment C: Reputation System
**Concept:** Agents maintain reputation scores affecting future decisions

**Reputation Mechanics:**
- Cooperation (+1 reputation)
- Defection (-1 reputation, minimum 0)
- Reputation influences opponent's decision-making

## 📊 Data Analysis & Visualization

### Generated Visualizations
1. **Cooperation Rates Across Experiments.png**
2. **Cumulative Payoffs Per Round Across Experiments.png**
3. **Cumulative Payoffs Per Round in New Experiments.png**
4. **Move Distribution Across Experiments.png**

### CSV Data Files
- `results_df_asymmetry.csv`: Information asymmetry results
- `results_df_moral.csv`: Moral reasoning results
- `results_df_reputation.csv`: Reputation system results
- `vanilla_fear.csv`: Vanilla simulation with fear emphasis
- `vanilla_hope.csv`: Vanilla simulation with hope emphasis
- `vanila_fear_10_seconds.csv`: Short-term fear simulation
- `vanilla_fear_1day.csv`: Long-term fear simulation

### Key Metrics Tracked
- **Round-by-round moves** (Stay Silent vs Confess)
- **Individual payoffs** per round
- **Cumulative scores** over time
- **Agent reasoning** for each decision
- **Reputation scores** (where applicable)
- **Moral prompts used** (in moral reasoning experiments)

## 🎯 Key Findings & Insights

### 1. Information Asymmetry Impact
- **Noise reduces cooperation** as agents become uncertain about opponent behavior
- **Trust erosion** occurs when agents perceive false defections
- **Defensive strategies** emerge under information uncertainty

### 2. Moral Reasoning Effects
- **Ethical prompts increase cooperation** rates
- **Long-term thinking** promotes collaborative behavior
- **Self-interest prompts** lead to more defection

### 3. Reputation Systems
- **Reputation building** incentivizes cooperation
- **Reputation decay** from defection creates deterrent effects
- **Reputation-based strategies** show more stable cooperation patterns

## 🚀 Setup Instructions

### Prerequisites
```bash
# Install basic requirements
pip install streamlit pandas numpy matplotlib seaborn altair

# For advanced free AI models (optional)
pip install torch transformers accelerate
```

### 🆓 Free AI Options (Zero Cost!)

#### Option 1: Simple AI (Recommended for beginners)
```bash
streamlit run SimpleAIPrisonersDilemma.py
```
- **No external dependencies** beyond basic Python packages
- **Intelligent rule-based agents** with different personalities
- **Zero setup time** - runs immediately
- **Educational value** - transparent decision-making process

#### Option 2: Hugging Face Transformers (Advanced)
```bash
streamlit run FreeAIPrisonersDilemma.py
```
- **Real neural network models** (T5, GPT-2, DialoGPT)
- **More realistic responses** but requires larger download
- **GPU acceleration** supported for faster inference
- **Multiple model options** to choose from

#### Option 3: Rule-Based Agents (Classic)
```bash
streamlit run RuleBasedStreamlit.py
```
- **Classic game theory strategies** (Tit-for-Tat, etc.)
- **Tournament mode** for strategy comparison
- **Deterministic behavior** for reproducible results
- **Educational insights** into strategy effectiveness

### Environment Setup
1. **No API Keys Required** - All solutions are completely free!

2. **Run Simple AI** (fastest start):
```bash
streamlit run SimpleAIPrisonersDilemma.py
```

3. **Run Advanced Free AI** (if you have time to download models):
```bash
streamlit run FreeAIPrisonersDilemma.py
```

4. **Run Jupyter Notebook** (for research):
```bash
jupyter notebook IterrativePrisonersDilemma.ipynb
```

### 🎯 Which Option to Choose?

**For Quick Start:** `SimpleAIPrisonersDilemma.py`
- Instant setup, no downloads
- Intelligent behavior patterns
- Perfect for learning and experimentation

**For Research:** `FreeAIPrisonersDilemma.py`
- Real neural networks
- More sophisticated responses
- Better for academic studies

**For Education:** `RuleBasedStreamlit.py`
- Classic strategies
- Transparent logic
- Great for teaching game theory concepts

## 🆓 Free AI Implementation

### Zero-Cost AI Options

#### 1. Local Models (Completely Free)
- **Hugging Face Transformers**: Google Flan-T5, Microsoft DialoGPT, DistilGPT2
- **Ollama**: Run Llama 2, Mistral, CodeLlama locally
- **LocalAI**: OpenAI-compatible API for local models

#### 2. Free API Tiers
- **Groq**: 14,400 requests/day free (Llama 3, Mixtral, Gemma)
- **Hugging Face**: 1,000 requests/month free
- **Google Colab**: Free GPU access for model inference

#### 3. Free AI Files
- `FreeAIPrisonersDilemma.py`: Streamlit app using free models
- Updated notebook cells for free AI experiments

### Benefits of Free AI Models
- ✅ **Zero API costs** - No OpenAI charges
- ✅ **No rate limits** - Unlimited local inference
- ✅ **Data privacy** - Everything runs locally
- ✅ **Educational use** - Perfect for learning and experimentation
- ✅ **Customizable** - Modify models as needed

### Free AI Model Options

#### Recommended Models
1. **Google Flan-T5** (`google/flan-t5-base`)
   - Best for instruction following
   - Good reasoning capabilities
   - Lightweight and fast

2. **Microsoft DialoGPT** (`microsoft/DialoGPT-medium`)
   - Excellent for conversational AI
   - Human-like responses
   - Good for prisoner's dilemma scenarios

3. **DistilGPT2** (`distilgpt2`)
   - Fastest inference
   - Minimal memory requirements
   - Good for rapid experimentation

### Setup Instructions for Free AI

#### Option 1: Hugging Face Transformers
```bash
pip install torch transformers accelerate
```

#### Option 2: Ollama (Local)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download models
ollama pull llama2:7b-chat
ollama pull mistral:7b-instruct
```

#### Option 3: Groq (Free API)
```bash
pip install groq
# Get free API key from https://console.groq.com/
```

### Usage Examples

#### Free AI Streamlit App
```bash
streamlit run FreeAIPrisonersDilemma.py
```

#### Free AI Notebook
Run the new cells in `IterrativePrisonersDilemma.ipynb` starting from the "Free AI Models Implementation" section.

## 🧪 Experimental Ideas & Extensions

### 1. Multi-Agent Tournaments
- **Round-robin tournaments** with different agent personalities
- **Evolutionary strategies** where successful approaches proliferate
- **Agent learning** from previous encounters

### 2. Advanced Cognitive Biases
- **Loss aversion** bias in decision-making
- **Anchoring effects** from first impressions
- **Confirmation bias** in interpreting opponent actions
- **Prospect theory** applications

### 3. Dynamic Payoff Structures
- **Asymmetric payoff matrices** for different players
- **Time-varying payoffs** that change over rounds
- **Context-dependent scoring** based on external factors

### 4. Communication Experiments
- **Pre-game communication** allowing strategy discussion
- **Cheap talk** scenarios with non-binding promises
- **Signaling mechanisms** through action patterns

### 5. Multi-Player Games
- **N-player public goods games**
- **Coalition formation** experiments
- **Voting mechanisms** and collective decision-making

### 6. Temporal Dynamics
- **Varying time horizons** (known vs unknown end points)
- **Discounting factors** for future payoffs
- **Real-time pressure** scenarios

### 7. Personality-Based Agents
- **Big Five personality traits** influence on decisions
- **Cultural background** variations
- **Risk tolerance** differences
- **Emotional state** modeling

### 8. Meta-Gaming Strategies
- **Strategy recognition** algorithms
- **Counter-strategy development**
- **Deception detection** mechanisms

## 📈 Advanced Analysis Techniques

### 1. Statistical Analysis
```python
# Cooperation rate analysis
cooperation_rate = df['A Move'].value_counts()['Stay Silent'] / len(df)

# Payoff distribution analysis
payoff_variance = df['A Payoff'].var()

# Strategy stability analysis
strategy_changes = (df['A Move'] != df['A Move'].shift()).sum()
```

### 2. Time Series Analysis
- **Trend analysis** in cooperation rates
- **Seasonal patterns** in decision-making
- **Regime change detection** in strategies

### 3. Network Analysis
- **Cooperation networks** between agents
- **Strategy diffusion** patterns
- **Influence propagation** models

### 4. Machine Learning Applications
- **Strategy prediction** models
- **Opponent modeling** using historical data
- **Optimal response** calculation

## 🎨 Visualization Enhancements

### Current Visualizations
The repository includes sophisticated dark-themed visualizations with:
- **Vibrant color palettes** for clear differentiation
- **Interactive elements** in Streamlit interface
- **Multi-experiment comparisons**
- **Real-time updates** during simulation

### Suggested Enhancements
1. **Heatmaps** showing cooperation patterns over time
2. **Network diagrams** for multi-agent interactions
3. **Animation sequences** showing strategy evolution
4. **3D visualizations** for multi-dimensional analysis

## 🔧 Code Architecture

### Modular Design
The codebase follows a clean modular structure:
- **Core game logic** separated from UI
- **Experiment-specific functions** for different scenarios
- **Visualization utilities** for consistent plotting
- **Data export capabilities** for further analysis

### Extension Points
1. **Custom agent personalities** through system prompt modification
2. **New experiment types** by implementing similar simulation functions
3. **Alternative payoff structures** through payoff function replacement
4. **Different LLM models** by modifying the API call function

## 📚 Educational Applications

### Research Questions
1. How do cognitive biases affect strategic decision-making?
2. What role does communication play in cooperation?
3. How do reputation systems influence long-term behavior?
4. Can AI agents exhibit human-like strategic thinking?

### Teaching Scenarios
- **Game theory fundamentals** demonstration
- **Behavioral economics** experiments
- **AI ethics** discussions
- **Strategic thinking** development

## 🚨 Important Notes

### API Usage
- **Cost monitoring** recommended for extensive experiments
- **Rate limiting** awareness for high-volume simulations
- **API key security** best practices

### Data Privacy
- **No personal data** collected or stored
- **Simulation results** are anonymized
- **API interactions** follow OpenAI's usage policies

### Reproducibility
- **Random seed control** for consistent results
- **Version tracking** for dependencies
- **Experiment documentation** for replication

## 🎯 Future Directions

### 1. Real-Time Human vs AI Games
Interactive platforms where humans can play against AI agents

### 2. Blockchain Integration
Decentralized game theory experiments with cryptocurrency incentives

### 3. VR/AR Implementations
Immersive environments for game theory education

### 4. Cross-Cultural Studies
Agents trained on different cultural contexts

### 5. Quantum Game Theory
Exploration of quantum mechanical game theory concepts

## 🏆 Conclusion

GameTheorySimAI provides a comprehensive platform for exploring strategic decision-making through the lens of artificial intelligence. By combining classic game theory with modern LLM capabilities, it offers unique insights into both human and artificial intelligence behavior in strategic scenarios.

The repository serves as both a research tool and educational resource, demonstrating how AI agents can simulate complex human decision-making processes while providing quantitative analysis of strategic interactions.

Whether you're a researcher, educator, or enthusiast, this platform offers endless possibilities for exploring the fascinating intersection of game theory, artificial intelligence, and human behavior.

---

*Last updated: July 2025*
*For questions or contributions, please refer to the repository's issue tracker.*
