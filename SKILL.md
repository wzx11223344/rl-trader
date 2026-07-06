---
name: rl-trader
description: 纯 NumPy 强化学习交易引擎，DQN/PPO/A2C 从零实现，含 LSTM 网络、优先经验回放、完整回测系统和 Plotly 可视化报告。适用于强化学习交易策略训练与评估。触发词：RL交易、强化学习交易、DQN交易、PPO交易、A2C交易、DRL回测、AI炒股。
---

# RL-Trader 强化学习交易引擎

## 能力边界

### ✅ 支持的能力
- **三大 RL 算法从零实现**：DQN（含 Double DQN + Dueling Architecture + Prioritized Experience Replay）、PPO（含 GAE + Clipped Objective + Entropy Bonus）、A2C（Advantage Actor-Critic），全部用纯 NumPy 编写，无 PyTorch/TensorFlow 依赖
- **LSTM 循环网络**：支持在 Q 网络 / Actor-Critic 网络中嵌入 LSTM 层，捕捉时序依赖
- **优先经验回放**：Prioritized Experience Replay Buffer，按 TD-Error 优先级采样
- **完整回测引擎**：逐笔交易记录、权益曲线、Sharpe Ratio / Max Drawdown / Win Rate / Profit Factor / Calmar Ratio / 年化收益与波动等全套指标计算
- **Plotly 交互报告**：权益曲线图、回撤热力图、交易分布直方图、月度收益热力图
- **akshare 数据集成**：一行命令拉取 A 股 / ETF 历史行情数据进行训练
- **模型保存与加载**：pickle 格式持久化，支持中断恢复训练和部署推理

### ❌ 不支持的能力
- 实时交易接口或券商 API 对接
- 多资产组合同时交易（单标的 Only）
- 高频/Level 2/Tick 级数据处理
- OpenAI Gym / Stable-Baselines3 接口兼容
- GPU 加速训练（纯 NumPy CPU 实现）
- 订单簿仿真或滑点/冲击成本建模

## 触发条件

当用户提及以下关键词或意图时，应优先调用本 Skill：
- "强化学习交易"、"RL交易"、"DQN交易"、"PPO交易"、"A2C交易"
- "训练交易智能体"、"AI炒股"、"DRL stock trading"
- "用强化学习做股票/期货/ETF交易"
- "深度强化学习回测"、"RL backtest"
- "从零实现 DQN/PPO"、"NumPy强化学习"

## 使用方法

### CLI 快速训练
```bash
# PPO 训练（推荐默认算法）
python train.py --algo ppo --ticker 000300 --episodes 200

# DQN 训练 + 自定义学习率
python train.py --algo dqn --ticker 000001 --episodes 500 --lr 0.0001

# A2C 训练 + 自定义数据文件
python train.py --algo a2c --data ./data/mydata.csv --episodes 300

# 启用 LSTM + 优先经验回放
python train.py --algo dqn --ticker 000300 --episodes 300 --lstm --prioritized
```

### Python API 调用
```python
from rl_trader.env import TradingEnv, load_data_from_akshare
from rl_trader.dqn import DQNAgent
from rl_trader.ppo import PPOAgent
from rl_trader.a2c import A2CAgent
from rl_trader.backtest import BacktestEngine

# 加载沪深300数据
df = load_data_from_akshare("000300", start="2018-01-01", end="2023-12-31")

# 创建交易环境
env = TradingEnv(df, initial_capital=100000.0, commission=0.0003)

# 训练 PPO 智能体
agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim)
agent.train(env, episodes=200)

# 回测评估
engine = BacktestEngine(env, initial_capital=100000.0)
report = engine.run(agent)
print(f"Sharpe: {report['sharpe_ratio']:.2f}, MaxDD: {report['max_drawdown']:.2%}")
```

## 输出示例

训练日志：
```
Episode 50/200 | Reward: -0.0342 | Portfolio: 100,234 | Epsilon: 0.42
Episode 100/200 | Reward: 0.1521 | Portfolio: 103,891 | Epsilon: 0.18
Episode 200/200 | Reward: 0.2876 | Portfolio: 112,450 | Epsilon: 0.01
```

回测报告：
```
===== Backtest Report =====
Total Return:      12.45%
Annualized Return:  8.32%
Sharpe Ratio:       1.24
Max Drawdown:      -8.71%
Win Rate:          58.3%
Profit Factor:      1.67
Calmar Ratio:       0.96
============================
```

## FAQ

**Q: 三个算法选哪个？**
A: PPO 是最稳定推荐，收敛性好且调参相对容易。DQN 适合离散动作空间。A2C 训练速度快但有时方差大。建议先用 PPO 跑 baseline。

**Q: 为什么不用 PyTorch/TensorFlow？**
A: 全部网络层（Dense、LSTM）、反向传播、优化器均用 NumPy 手写实现，适合学习 RL 底层原理，无框架黑盒，完全可控可定制。

**Q: 支持哪些品种？**
A: 通过 akshare 支持 A 股个股、ETF、指数数据。也可导入自定义 CSV 数据文件。

**Q: LSTM 有什么作用？**
A: LSTM 能捕捉历史价格序列的时序依赖（趋势、动量、波动率聚集），在 DQN/PPO/A2C 网络中均可选开启，通常能提升趋势跟踪策略的表现。

**Q: 能实盘交易吗？**
A: 不能。本引擎仅供研究与回测使用，不连接任何券商接口，不提供实盘下单功能。

## 技术栈

- **核心依赖**: NumPy, Pandas
- **可视化**: Plotly
- **数据获取**: akshare（A股市数据）
- **网络层**: 纯 NumPy 手写（Dense / LSTM / ReplayBuffer）
- **模型持久化**: pickle
- **语言**: Python 3.8+
