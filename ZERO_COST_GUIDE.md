# 🆓 Zero-Cost Game Theory AI Guide

## Overview
This guide provides multiple ways to run intelligent game theory simulations without any API costs or paid services. Perfect for students, researchers, and educators working on tight budgets.

## 🎯 Zero-Cost Options

### 1. Rule-Based Agents (Recommended for Beginners)
**File:** `RuleBasedAgents.py`
- ✅ **Zero cost** - No internet required
- ✅ **Instant setup** - No downloads needed
- ✅ **Educational** - Clear strategy implementations
- ✅ **Fast** - Thousands of simulations per second

**Available Strategies:**
- Tit-for-Tat
- Generous Tit-for-Tat
- Always Cooperate
- Always Defect
- Grudger
- Pavlov (Win-stay, Lose-shift)
- Adaptive
- Fear-based
- Reputation-based

**Usage:**
```bash
python RuleBasedAgents.py
```

### 2. Free AI Models (Advanced)
**File:** `FreeAIPrisonersDilemma.py`
- ✅ **Zero API costs** - Uses local AI models
- ✅ **Real AI reasoning** - Not just rule-based
- ✅ **Customizable** - Modify prompts and models
- ✅ **Privacy** - Everything runs locally

**Supported Models:**
- Google Flan-T5 (Recommended)
- Microsoft DialoGPT
- DistilGPT2
- GPT-2

**Usage:**
```bash
pip install torch transformers
streamlit run FreeAIPrisonersDilemma.py
```

### 3. Jupyter Notebook Experiments
**File:** `IterrativePrisonersDilemma.ipynb`
- ✅ **Interactive** - Step-by-step analysis
- ✅ **Visualization** - Built-in charts
- ✅ **Experiments** - Multiple scenarios
- ✅ **Educational** - Learn by doing

**Free AI cells added:**
- Information Asymmetry with free models
- Moral Reasoning with free models
- Reputation Systems with free models

## 🚀 Quick Start Guide

### Option 1: Rule-Based (Easiest)
```bash
# 1. Clone repository
git clone https://github.com/cseth1/GameTheorySimAI.git
cd GameTheorySimAI

# 2. Install basic requirements
pip install pandas matplotlib numpy

# 3. Run rule-based simulation
python RuleBasedAgents.py
```

### Option 2: Free AI Models
```bash
# 1. Install AI requirements
pip install torch transformers streamlit

# 2. Run free AI app
streamlit run FreeAIPrisonersDilemma.py
```

### Option 3: Local AI with Ollama
```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Download free models
ollama pull llama2:7b-chat
ollama pull mistral:7b-instruct

# 3. Use in your code
import ollama
response = ollama.chat(model='llama2:7b-chat', messages=[...])
```

## 📊 Comparison of Zero-Cost Options

| Feature | Rule-Based | Free AI | Ollama | Colab |
|---------|------------|---------|--------|-------|
| **Cost** | Free | Free | Free | Free |
| **Setup Time** | 1 min | 5 min | 10 min | 2 min |
| **Internet Required** | No | No | No | Yes |
| **AI Reasoning** | No | Yes | Yes | Yes |
| **Speed** | Very Fast | Medium | Medium | Fast |
| **Memory Usage** | Low | High | High | None |
| **Customization** | High | High | High | Medium |

## 🔧 Advanced Zero-Cost Setups

### Google Colab (Free GPU)
```python
# Install in Colab
!pip install transformers torch

# Use free GPU
from transformers import pipeline
generator = pipeline('text-generation', model='gpt2', device=0)
```

### Hugging Face Spaces (Free Hosting)
- Deploy your simulation as a web app
- Free hosting for public projects
- No setup required

### GitHub Codespaces (Free Tier)
- 120 hours/month free
- Full development environment
- Run AI models in the cloud

## 🧪 Experimental Ideas (Zero Cost)

### 1. Strategy Evolution
```python
# Evolve strategies over generations
def evolve_strategies(population, generations):
    for gen in range(generations):
        # Tournament selection
        scores = run_tournament(population)
        # Mutation and crossover
        population = create_next_generation(population, scores)
    return population
```

### 2. Multi-Agent Ecosystems
```python
# Simulate entire ecosystems
agents = [
    RuleBasedAgent("tit_for_tat"),
    RuleBasedAgent("generous_tit_for_tat"),
    RuleBasedAgent("adaptive"),
    # ... more agents
]
```

### 3. Dynamic Environments
```python
# Changing payoff matrices
def dynamic_payoff(round_num):
    if round_num < 50:
        return standard_payoff
    else:
        return modified_payoff
```

### 4. Learning Agents
```python
# Agents that learn from experience
class LearningAgent:
    def __init__(self):
        self.q_table = {}
        self.learning_rate = 0.1
    
    def update_q_value(self, state, action, reward):
        # Q-learning update
        pass
```

## 📈 Performance Optimization

### Speed Up Simulations
```python
# Parallel processing
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def parallel_simulation(strategies, rounds):
    with ProcessPoolExecutor() as executor:
        results = executor.map(run_simulation, strategies)
    return list(results)
```

### Memory Optimization
```python
# Generator-based simulation
def memory_efficient_simulation(rounds):
    for round_num in range(rounds):
        yield simulate_round(round_num)
```

## 🎓 Educational Applications

### Classroom Activities
1. **Strategy Tournament**: Students create their own strategies
2. **Hypothesis Testing**: Test game theory predictions
3. **Behavioral Analysis**: Compare AI vs human decisions
4. **Ethics Discussion**: Moral implications of different strategies

### Research Projects
1. **Cultural Differences**: How do cultural backgrounds affect strategies?
2. **Evolutionary Dynamics**: Which strategies survive over time?
3. **Network Effects**: How does network structure affect cooperation?
4. **Bounded Rationality**: Limited information decision-making

## 🔍 Troubleshooting

### Common Issues
1. **Memory Error**: Use smaller models (distilgpt2)
2. **Slow Performance**: Reduce number of rounds
3. **Import Errors**: Install missing dependencies
4. **CUDA Issues**: Use CPU-only mode

### Performance Tips
- Use CPU for small experiments
- Batch multiple simulations
- Cache model loading
- Use progress bars for long runs

## 📚 Learning Resources

### Game Theory Basics
- [Game Theory 101](https://www.youtube.com/playlist?list=PLKI1h_nAkaQoDzI4xDIXzx6U2ergFmedo)
- [Prisoner's Dilemma](https://en.wikipedia.org/wiki/Prisoner%27s_dilemma)
- [Evolutionary Game Theory](https://www.youtube.com/watch?v=YNMkADpvO4w)

### AI Model Resources
- [Hugging Face Model Hub](https://huggingface.co/models)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

### Programming Resources
- [Python for Game Theory](https://nashpy.readthedocs.io/)
- [Pandas for Data Analysis](https://pandas.pydata.org/docs/)
- [Matplotlib for Visualization](https://matplotlib.org/stable/tutorials/index.html)

## 🌟 Contributing

Want to add more zero-cost options? Here's how:

1. Fork the repository
2. Add your implementation
3. Update documentation
4. Submit a pull request

### Ideas for Contributions
- New rule-based strategies
- Integration with more free AI services
- Educational materials
- Performance optimizations
- Visualization improvements

## 🎉 Conclusion

You now have multiple ways to explore game theory with AI agents without spending money:

1. **Start with rule-based agents** for immediate results
2. **Try free AI models** for more realistic behavior
3. **Use Jupyter notebooks** for interactive learning
4. **Explore advanced setups** as you learn more

The field of game theory AI is vast and fascinating - and now completely accessible regardless of budget!

---

*Happy experimenting! 🚀*
