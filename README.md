# DeepTrade: RL Agent for Portfolio Management

An AI trading agent that learns optimal stock trading strategies using Deep Q-Learning (DQN).

## 🎯 Project Goal

Build a reinforcement learning agent that learns to trade stocks by:
- Making sequential buy/sell/hold decisions
- Maximizing risk-adjusted returns
- Adapting to different market conditions

## 🏗️ Architecture

- **Environment:** Custom Gym environment simulating stock trading
- **Agent:** Deep Q-Network (DQN) with Double DQN
- **State:** Stock price + technical indicators + portfolio state
- **Actions:** HOLD (0), BUY (1), SELL (2)
- **Reward:** Risk-adjusted portfolio returns

## 📊 Features

### Technical Indicators
- Simple Moving Averages (SMA 20, 50)
- Relative Strength Index (RSI)
- MACD & Signal Line
- Bollinger Bands
- Volume Ratio

### DQN Improvements
- [x] Experience Replay
- [x] Target Network
- [x] Double DQN
- [ ] Dueling DQN (Week 3)
- [ ] Prioritized Replay (Week 3)

## 🚀 Quick Start

### Installation
```bash
# Clone repo
git clone <your-repo>
cd DeepTrade

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
python src/train.py --config configs/config.yaml
```

### Evaluation
```bash
python src/evaluate.py --model models/checkpoints/best_model.pth
```

## 📁 Project Structure
```bash
DeepTrade/
├── data/               # Stock data
├── src/                # Source code
├── configs/            # Configuration files
├── models/             # Saved models
├── results/            # Training results
└── notebooks/          # Jupyter notebooks
```

## 📈 Results

*Coming soon after training...*

## 🛠️ Tech Stack

- Python 3.8+
- PyTorch
- Gymnasium
- yfinance
- pandas, numpy
- matplotlib, plotly

## 📝 Development Log

- [x] Day 1: RL Theory
- [x] Day 2: Project Setup
- [x] Day 3: Data Collection
- [ ] Day 4-5: Environment Implementation
- [ ] Day 6-7: DQN Agent
- [ ] Day 8-14: Training & Debugging
- [ ] Day 15-21: Evaluation & Polish

## 📚 References

- [DQN Paper](https://arxiv.org/abs/1312.5602)
- [Double DQN Paper](https://arxiv.org/abs/1509.06461)
- Sutton & Barto: Reinforcement Learning

## 👤 Author

[Your Name]
- LinkedIn: [https://www.linkedin.com/in/ayaan-skh/]
- GitHub: [https://github.com/Ayaan-Skh]

## 📄 License

MIT License
⚠️ Educational project.
This system is not financial advice and is not intended for live trading