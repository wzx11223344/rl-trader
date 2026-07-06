# RL Trader

**Reinforcement Learning for Trading -- DQN/PPO/A2C implemented FROM SCRATCH in NumPy. Not calling OpenAI Gym or stable-baselines. This is a teaching tool and research platform.**

---

## Why RL Trader?

Most RL-for-trading projects are thin wrappers around stable-baselines3. RL Trader is different:

- Every neural network layer (`Dense`, `Conv1D`, `LSTM`) is written from scratch in NumPy with manual forward/backward passes and Adam optimization.
- Every RL algorithm (DQN, PPO, A2C) is implemented in pure NumPy -- study the code, understand the math, modify anything.
- A trading environment with RSI, MACD, Bollinger Bands, stop-loss, take-profit, transaction costs, and running normalization.
- A complete backtesting engine with Sharpe ratio, max drawdown, Calmar ratio, profit factor, and Plotly visualization.

No PyTorch. No TensorFlow. No gym. No stable-baselines. Just NumPy and hard work.

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `numpy`, `pandas`, `akshare`, `plotly`, `pyyaml`

---

## Quick Start

### Train a PPO agent on CSI 300 ETF:

```bash
python train.py --algo ppo --ticker 000300 --episodes 200
```

### Train a DQN agent with custom settings:

```bash
python train.py --algo dqn --ticker 000001 --episodes 500 --lr 0.0001 --batch-size 128
```

### Train an A2C agent on custom data:

```bash
python train.py --algo a2c --data ./data/mydata.csv --episodes 300
```

### Run a standalone backtest:

```bash
python examples/run_backtest.py --model ./output/ppo_000300.pkl --algo ppo --ticker 000300
```

---

## Architecture

```
rl-trader/
├── train.py                   # CLI training entry point
├── rl_trader/
│   ├── env.py                 # Trading environment with technical indicators
│   ├── networks.py            # Dense, Conv1D, LSTM from scratch (NumPy)
│   ├── dqn.py                 # DQN + Double DQN + Dueling + PER
│   ├── ppo.py                 # PPO with GAE, clipped objective
│   ├── a2c.py                 # A2C with n-step returns
│   ├── replay.py              # Uniform + Prioritized Experience Replay
│   └── backtest.py            # Backtest engine + Plotly charts
├── examples/
│   └── run_backtest.py        # Standalone backtest runner
└── output/                    # Models, CSVs, reports
```

---

## Environment

**State space** (14 features): price, holdings, cash, returns(5/10/20d), RSI, MACD, MACD signal, Bollinger Bands (upper/lower), volume ratio, SMA ratio -- all normalized to [-1, 1].

**Actions**: Discrete (Short=0, Hold=1, Long=2) or Continuous [-1, 1].

**Reward**: Sharpe-like (mean return / std return over window).

**Risk controls**: Stop loss (5%), take profit (10%), max steps, bankruptcy (90% loss).

---

## Algorithms

### DQN
- Epsilon-greedy exploration with decay
- **Double DQN**: Online network selects actions, target network evaluates
- **Dueling Architecture**: Separate value and advantage streams: `Q(s,a) = V(s) + A(s,a) - mean(A)`
- **Prioritized Experience Replay**: Sum-tree based, TD-error proportional sampling
- **N-step returns** (default n=3): Reduces bias-variance tradeoff
- Soft target network updates (tau=0.005)

### PPO
- **Clipped surrogate objective**: `min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)`
- **GAE** (Generalized Advantage Estimation, lambda=0.95)
- Entropy bonus for exploration
- Value function clipping
- Multiple epochs per batch (default 4)
- Shared actor-critic architecture

### A2C
- Synchronous Advantage Actor-Critic
- N-step returns (default n=5)
- Entropy regularization
- Separate actor and critic heads with shared feature extractor

---

## Neural Networks (from scratch)

```python
from rl_trader.networks import Dense, Conv1D, LSTM, build_network

# Build a network from config
config = [
    {"type": "dense", "units": 128, "activation": "relu"},
    {"type": "lstm", "units": 64, "return_sequences": False},
    {"type": "dense", "units": 3, "activation": "linear"},
]
model = build_network(config)

# Forward pass
output = model[0].forward(x)
for layer in model[1:]:
    output = layer.forward(output)

# Backward pass (automatic gradient computation)
for layer in reversed(model):
    grad = layer.backward(grad)
```

All layers include:
- Xavier uniform initialization
- Adam optimizer with momentum tracking
- Proper forward/backward chain rule

---

## Backtesting Metrics

- **Total Return** -- overall percentage return
- **Annualized Return** -- CAGR
- **Sharpe Ratio** -- risk-adjusted return (252 trading days)
- **Sortino Ratio** -- downside risk-adjusted
- **Maximum Drawdown** -- worst peak-to-trough
- **Calmar Ratio** -- return / max drawdown
- **Win Rate** -- percentage of profitable trades
- **Profit Factor** -- gross profit / gross loss
- **Avg Win / Avg Loss** -- expectancy metrics

---

## CLI Reference

```
python train.py --help

Arguments:
  --algo {dqn,ppo,a2c}       RL algorithm (default: ppo)
  --ticker TICKER            Stock/ETF code (default: 000300)
  --data PATH                CSV file path (overrides --ticker)
  --start DATE               Start date YYYYMMDD (default: 20150101)
  --end DATE                 End date YYYYMMDD (default: 20231231)
  --episodes N               Training episodes (default: 200)
  --lr LR                    Learning rate
  --gamma GAMMA              Discount factor (default: 0.99)
  --batch-size N             Batch size (default: 64)
  --capital N                Initial capital (default: 100000)
  --commission RATE          Transaction cost (default: 0.001)
  --stop-loss PCT            Stop loss (default: 0.05)
  --take-profit PCT          Take profit (default: 0.10)
  --output DIR               Output directory (default: ./output)
  --save-model PATH          Model save path
  --load-model PATH          Model load path
  --no-backtest              Skip backtesting
  --no-plot                  Skip Plotly charts
  --seed SEED                Random seed (default: 42)
  --verbose                  Verbose output
```

---

## License

MIT
