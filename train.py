#!/usr/bin/env python
"""
RL Trader - Training CLI
==========================
Train DQN, PPO, or A2C agents on stock/ETF data.

Usage:
    python train.py --algo ppo --ticker 000300 --episodes 200
    python train.py --algo dqn --ticker 000001 --episodes 500 --lr 0.0001
    python train.py --algo a2c --data ./data/mydata.csv --episodes 300
"""

import argparse
import os
import sys
import numpy as np
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_trader.env import TradingEnv, load_data_from_akshare
from rl_trader.dqn import DQNAgent
from rl_trader.ppo import PPOAgent
from rl_trader.a2c import A2CAgent
from rl_trader.backtest import BacktestEngine


def parse_args():
    parser = argparse.ArgumentParser(
        description="RL Trader - Train reinforcement learning trading agents"
    )
    # Algorithm
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["dqn", "ppo", "a2c"],
                        help="RL algorithm to use (default: ppo)")
    # Data
    parser.add_argument("--ticker", type=str, default="000300",
                        help="Stock/ETF ticker code (e.g., 000300, 000001)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to CSV data file (overrides --ticker)")
    parser.add_argument("--start", type=str, default="20150101",
                        help="Start date for data (default: 20150101)")
    parser.add_argument("--end", type=str, default="20231231",
                        help="End date for data (default: 20231231)")
    # Training
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of training episodes (default: 200)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: algo-specific)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    # Environment
    parser.add_argument("--capital", type=float, default=100000.0,
                        help="Initial capital (default: 100000)")
    parser.add_argument("--commission", type=float, default=0.001,
                        help="Transaction cost (default: 0.001)")
    parser.add_argument("--stop-loss", type=float, default=0.05,
                        help="Stop loss threshold (default: 0.05)")
    parser.add_argument("--take-profit", type=float, default=0.10,
                        help="Take profit threshold (default: 0.10)")
    # Output
    parser.add_argument("--output", type=str, default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--save-model", type=str, default=None,
                        help="Path to save trained model")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to load pre-trained model")
    parser.add_argument("--no-backtest", action="store_true",
                        help="Skip backtesting after training")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip Plotly chart generation")
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed training progress")

    return parser.parse_args()


def create_agent(algo: str, state_dim: int, action_dim: int, args):
    """Create the appropriate RL agent based on algorithm choice."""
    lr = args.lr

    if algo == "dqn":
        lr = lr or 0.001
        return DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=[128, 128],
            lr=lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            buffer_capacity=100000,
            use_PER=True,
            use_double=True,
            use_dueling=True,
            n_step=3,
        )
    elif algo == "ppo":
        lr = lr or 0.0003
        return PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            discrete=True,
            hidden_layers=[256, 256],
            lr=lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            horizon=2048,
            epochs=4,
            clip_epsilon=0.2,
        )
    elif algo == "a2c":
        lr = lr or 0.001
        return A2CAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            discrete=True,
            hidden_layers=[128, 128],
            lr=lr,
            gamma=args.gamma,
            n_steps=5,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def train_dqn(agent, env, episodes, verbose):
    """Training loop for DQN."""
    episode_rewards = []
    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        step = 0

        while not done:
            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.push(state, action, reward, next_state, done)

            loss_info = agent.update()
            state = next_state
            ep_reward += reward
            step += 1

        agent.flush_n_step()

        episode_rewards.append(ep_reward)

        if verbose or (ep + 1) % 10 == 0:
            avg_r = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else ep_reward
            print(f"Ep {ep + 1:4d}/{episodes} | "
                  f"Reward: {ep_reward:8.2f} | "
                  f"Avg10: {avg_r:8.2f} | "
                  f"Eps: {agent.epsilon:.3f} | "
                  f"Steps: {step}")

    return episode_rewards


def train_ppo(agent, env, episodes, verbose):
    """Training loop for PPO."""
    episode_rewards = []
    total_steps = 0

    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        step = 0

        while not done:
            action, log_prob, value = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.push(state, action, reward, done, value, log_prob)

            state = next_state
            ep_reward += reward
            step += 1
            total_steps += 1

            # Update when buffer is full
            if len(agent.states) >= agent.horizon:
                update_info = agent.update()
                if update_info and verbose:
                    print(f"  Update at step {total_steps}: "
                          f"P_Loss={update_info['policy_loss']:.4f}, "
                          f"V_Loss={update_info['value_loss']:.4f}")

        # Flush remaining
        if len(agent.states) > 0:
            update_info = agent.update()

        episode_rewards.append(ep_reward)

        if verbose or (ep + 1) % 5 == 0:
            avg_r = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else ep_reward
            print(f"Ep {ep + 1:4d}/{episodes} | "
                  f"Reward: {ep_reward:8.2f} | "
                  f"Avg10: {avg_r:8.2f} | "
                  f"Steps: {step}")

    agent.flush()
    return episode_rewards


def train_a2c(agent, env, episodes, verbose):
    """Training loop for A2C."""
    episode_rewards = []
    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        step = 0

        while not done:
            action, value = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.push(state, action, reward)

            update_info = agent.update(next_state, done)
            state = next_state
            ep_reward += reward
            step += 1

        agent.flush()
        episode_rewards.append(ep_reward)

        if verbose or (ep + 1) % 10 == 0:
            avg_r = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else ep_reward
            print(f"Ep {ep + 1:4d}/{episodes} | "
                  f"Reward: {ep_reward:8.2f} | "
                  f"Avg10: {avg_r:8.2f} | "
                  f"Steps: {step}")

    return episode_rewards


def main():
    args = parse_args()

    # Set seed
    np.random.seed(args.seed)

    print("=" * 60)
    print("  RL TRADER - Training")
    print("=" * 60)
    print(f"  Algorithm:    {args.algo.upper()}")
    print(f"  Ticker:       {args.ticker}")
    print(f"  Episodes:     {args.episodes}")
    print(f"  Capital:      {args.capital:,.0f}")
    print(f"  Commission:   {args.commission:.3%}")
    print(f"  Stop Loss:    {args.stop_loss:.1%}")
    print(f"  Take Profit:  {args.take_profit:.1%}")
    print("=" * 60)

    # Load data
    if args.data:
        print(f"\nLoading data from: {args.data}")
        df = None  # Will be loaded by env
        data_source = args.data
    else:
        print(f"\nDownloading data for {args.ticker} from akshare...")
        try:
            df = load_data_from_akshare(args.ticker, args.start, args.end)
            print(f"  Loaded {len(df)} rows")
            data_source = df
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)

    # Create environment
    print("\nInitializing trading environment...")
    env = TradingEnv(
        data=data_source,
        initial_capital=args.capital,
        commission=args.commission,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        action_type="discrete",
        reward_type="sharpe_like",
    )
    print(f"  State dim: {env.state_dim}, Action dim: {env.action_dim}")
    print(f"  Data range: {len(env.close)} bars")

    # Create agent
    agent = create_agent(args.algo, env.state_dim, env.action_dim, args)

    # Load pre-trained model if specified
    if args.load_model:
        print(f"\nLoading pre-trained model from: {args.load_model}")
        agent.load(args.load_model)

    # Train
    print(f"\nTraining {args.algo.upper()} for {args.episodes} episodes...\n")

    if args.algo == "dqn":
        rewards = train_dqn(agent, env, args.episodes, args.verbose)
    elif args.algo == "ppo":
        rewards = train_ppo(agent, env, args.episodes, args.verbose)
    elif args.algo == "a2c":
        rewards = train_a2c(agent, env, args.episodes, args.verbose)

    # Save model
    os.makedirs(args.output, exist_ok=True)
    if args.save_model:
        model_path = args.save_model
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(args.output, f"{args.algo}_{args.ticker}_{timestamp}.pkl")
    agent.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # Save training history
    history_path = os.path.join(args.output, "training_history.json")
    with open(history_path, "w") as f:
        json.dump({
            "algo": args.algo,
            "ticker": args.ticker,
            "episodes": args.episodes,
            "rewards": [float(r) for r in rewards],
            "final_avg_reward_10": float(np.mean(rewards[-10:])) if len(rewards) >= 10 else float(np.mean(rewards)),
        }, f, indent=2)
    print(f"Training history saved to: {history_path}")

    # Training summary
    print("\nTraining complete!")
    print(f"  Final Reward:       {rewards[-1]:.4f}")
    if len(rewards) >= 10:
        print(f"  Avg (last 10):      {np.mean(rewards[-10:]):.4f}")
    print(f"  Best Reward:        {max(rewards):.4f}")

    # Backtest
    if not args.no_backtest:
        print("\n" + "=" * 60)
        print("  BACKTESTING")
        print("=" * 60)

        backtest = BacktestEngine(env, initial_capital=args.capital)
        result = backtest.run(agent, render=args.verbose)

        backtest.export_csv(args.output)
        backtest.export_summary(result["metrics"], args.output)

        if not args.no_plot:
            backtest.plot(args.output, show=False)

    print("\nDone!")


if __name__ == "__main__":
    main()
