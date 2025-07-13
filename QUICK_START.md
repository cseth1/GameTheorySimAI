# 🚀 Quick Start Guide - GameTheorySimAI

## 📋 Installation

1. **Install Python 3.8+** (if not already installed)
2. **Clone/Download this repository**
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run setup (optional but recommended):**
   ```bash
   python setup.py
   ```

## 🎮 Usage Options

### 1. 🤖 Rule-Based Agents (100% FREE)
**Zero cost, no API keys, instant results!**

```bash
streamlit run RuleBasedStreamlit.py
```

**Features:**
- ✅ Completely free
- ✅ No API keys required
- ✅ 10+ intelligent strategies
- ✅ Tournament mode
- ✅ Real-time visualization
- ✅ Educational explanations

### 2. 🆓 Free AI Models (Also FREE!)
**Use real AI models without OpenAI costs**

```bash
streamlit run FreeAIPrisonersDilemma.py
```

**Features:**
- ✅ Google Flan-T5 (recommended)
- ✅ Microsoft DialoGPT
- ✅ DistilGPT2 (fastest)
- ✅ No API costs
- ✅ Runs locally
- ✅ Privacy-focused

### 3. 📊 Original OpenAI Version
**If you have OpenAI API access**

```bash
# Set your API key first
export OPENAI_API_KEY="your-key-here"
streamlit run VanilaPrisonersDilemma.py
```

### 4. 📓 Advanced Jupyter Notebook
**For research and experimentation**

```bash
jupyter notebook IterrativePrisonersDilemma.ipynb
```

**Features:**
- Information asymmetry experiments
- Moral reasoning prompts
- Reputation systems
- Advanced visualizations

## 🧪 Testing Your Setup

```bash
python test_setup.py
```

This will test:
- Rule-based agents
- Free AI models
- All dependencies

## 💡 Recommendations

### For Beginners:
1. **Start with Rule-Based Agents** - completely free, instant results
2. **Try different strategies** - see how they compete
3. **Run tournaments** - find the best strategies

### For Advanced Users:
1. **Use Free AI Models** - more realistic human-like behavior
2. **Jupyter Notebook** - advanced experiments
3. **Customize strategies** - modify the code

### For Researchers:
1. **All of the above** plus:
2. **Data analysis** - export results to CSV
3. **Custom experiments** - modify the simulation parameters
4. **Academic use** - cite patterns and results

## 🔧 Troubleshooting

### Common Issues:

1. **"Module not found" errors:**
   ```bash
   pip install -r requirements.txt
   ```

2. **PyTorch installation issues:**
   ```bash
   # For CPU only
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Streamlit not working:**
   ```bash
   pip install streamlit
   streamlit --version
   ```

4. **Free AI models slow:**
   - Use DistilGPT2 (fastest)
   - Reduce number of rounds
   - Run on GPU if available

### Getting Help:
- Check `COMPREHENSIVE_GUIDE.md` for full documentation
- Run `python test_setup.py` to diagnose issues
- Make sure Python 3.8+ is installed

## 🎯 What to Try First

1. **Quick Demo (1 minute):**
   ```bash
   streamlit run RuleBasedStreamlit.py
   ```
   - Select "Tit for Tat" vs "Always Cooperate"
   - Run 20 rounds
   - See the results!

2. **AI Demo (5 minutes):**
   ```bash
   streamlit run FreeAIPrisonersDilemma.py
   ```
   - Load "Google Flan-T5"
   - Run basic simulation
   - Compare with rule-based results

3. **Tournament (10 minutes):**
   - Use RuleBasedStreamlit.py
   - Select multiple strategies
   - Run tournament mode
   - See which strategy wins!

## 📊 Example Results

Typical findings:
- **Tit-for-Tat** performs well against most strategies
- **Always Defect** vs **Always Cooperate** = predictable outcomes
- **Generous Tit-for-Tat** often wins tournaments
- **Adaptive** strategies learn opponent patterns
- **Fear-based** agents show realistic human-like behavior

## 🆓 Cost Breakdown

- **Rule-based agents:** $0 forever
- **Free AI models:** $0 (runs locally)
- **Jupyter notebook:** $0 (except OpenAI calls)
- **OpenAI version:** ~$0.01-0.10 per simulation

## 🎓 Educational Value

Perfect for:
- Game theory courses
- AI/ML education
- Behavioral economics
- Strategy analysis
- Programming practice

## 📈 Next Steps

1. **Explore strategies** - try all 10 rule-based strategies
2. **Compare AI vs rules** - see how they differ
3. **Modify the code** - create your own strategies
4. **Run experiments** - collect data for analysis
5. **Share results** - contribute to game theory research

---

**🎉 Have fun exploring the fascinating world of game theory with AI!**
