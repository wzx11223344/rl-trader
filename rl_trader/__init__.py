# RL Trader - Reinforcement Learning Trading Engine
# DQN/PPO/A2C implemented from scratch using only NumPy

from .env import TradingEnv
from .networks import Dense, Conv1D, LSTM, build_network
from .replay import ReplayBuffer, PrioritizedReplayBuffer
from .dqn import DQNAgent
from .ppo import PPOAgent
from .a2c import A2CAgent
from .backtest import BacktestEngine

__version__ = "1.0.0"
__all__ = [
    "TradingEnv",
    "Dense", "Conv1D", "LSTM", "build_network",
    "ReplayBuffer", "PrioritizedReplayBuffer",
    "DQNAgent", "PPOAgent", "A2CAgent",
    "BacktestEngine",
]
