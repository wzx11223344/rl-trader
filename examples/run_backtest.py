#!/usr/bin/env python
"""
Example: Run a Backtest with a Trained Model
=============================================
Loads a saved model, runs a full backtest, and generates
an HTML report with equity curve, drawdown, and trade analysis.

Usage:
    python examples/run_backtest.py --model ./output/ppo_000300.pkl --ticker 000300
    python examples/run_backtest.py --model ./output/dqn_000001.pkl --data ./data/stock.csv
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_trader.env import TradingEnv, load_data_from_akshare
from rl_trader.dqn import DQNAgent
from rl_trader.ppo import PPOAgent
from rl_trader.a2c import A2CAgent
from rl_trader.backtest import BacktestEngine


def main():
    parser = argparse.ArgumentParser(description="RL Trader - Run Backtest")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model file (.pkl)")
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["dqn", "ppo", "a2c"],
                        help="Algorithm used in training")
    parser.add_argument("--ticker", type=str, default="000300",
                        help="Ticker symbol for data")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to CSV data file (overrides --ticker)")
    parser.add_argument("--start", type=str, default="20200101",
                        help="Start date for backtest")
    parser.add_argument("--end", type=str, default="20231231",
                        help="End date for backtest")
    parser.add_argument("--capital", type=float, default=100000.0,
                        help="Initial capital")
    parser.add_argument("--commission", type=float, default=0.001,
                        help="Transaction cost")
    parser.add_argument("--output", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--render", action="store_true",
                        help="Show step-by-step backtest output")

    args = parser.parse_args()

    print("=" * 60)
    print("  RL TRADER - Backtest")
    print("=" * 60)
    print(f"  Model:     {args.model}")
    print(f"  Algorithm: {args.algo.upper()}")
    print(f"  Ticker:    {args.ticker}")
    print(f"  Capital:   {args.capital:,.0f}")
    print("=" * 60)

    # Load data
    if args.data:
        print(f"\nLoading data from: {args.data}")
        data_source = args.data
    else:
        print(f"\nDownloading data for {args.ticker}...")
        df = load_data_from_akshare(args.ticker, args.start, args.end)
        print(f"  Loaded {len(df)} rows")
        data_source = df

    # Create environment
    print("\nInitializing trading environment...")
    env = TradingEnv(
        data=data_source,
        initial_capital=args.capital,
        commission=args.commission,
        action_type="discrete",
        reward_type="sharpe_like",
    )
    print(f"  State dim: {env.state_dim}, Action dim: {env.action_dim}")

    # Create agent and load weights
    print(f"\nLoading {args.algo.upper()} model from {args.model}...")
    if args.algo == "dqn":
        agent = DQNAgent(env.state_dim, env.action_dim)
    elif args.algo == "ppo":
        agent = PPOAgent(env.state_dim, env.action_dim, discrete=True)
    elif args.algo == "a2c":
        agent = A2CAgent(env.state_dim, env.action_dim, discrete=True)

    agent.load(args.model)
    print("  Model loaded successfully!")

    # Run backtest
    print("\nRunning backtest...")
    backtest = BacktestEngine(env, initial_capital=args.capital)
    result = backtest.run(agent, render=args.render)

    # Export results
    backtest.export_csv(args.output)
    backtest.export_summary(result["metrics"], args.output)

    # Generate Plotly chart
    print("\nGenerating visualization...")
    backtest.plot(args.output, show=False)

    # Print top-level metrics
    m = result["metrics"]
    print(f"\nKey Metrics:")
    print(f"  Sharpe Ratio:     {m['sharpe_ratio']:8.2f}")
    print(f"  Total Return:     {m['total_return']:8.2%}")
    print(f"  Max Drawdown:     {m['max_drawdown']:8.2%}")
    print(f"  Win Rate:         {m['win_rate']:8.2%}")
    print(f"  Profit Factor:    {m['profit_factor']:8.2f}")
    print(f"  Calmar Ratio:     {m['calmar_ratio']:8.2f}")
    print(f"\nDone! Report saved to {args.output}/")


if __name__ == "__main__":
    main()
