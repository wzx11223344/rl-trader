"""
Backtesting Engine
===================
Step-by-step backtesting with comprehensive metrics computation,
equity curve tracking, trade logging, and Plotly visualization.

Metrics: total_return, sharpe_ratio, max_drawdown, win_rate,
         profit_factor, calmar_ratio, annualized_return, annualized_vol.
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime


class BacktestEngine:
    """Backtesting engine for trained RL agents.

    Runs the agent through historical data step-by-step, records
    equity curve, trades, and computes performance metrics.

    Args:
        env: TradingEnv instance with loaded data.
        initial_capital: Starting capital (should match env setup).
    """

    def __init__(self, env, initial_capital: float = 100000.0):
        self.env = env
        self.initial_capital = initial_capital
        self.equity_curve: List[float] = []
        self.trades: List[Dict] = []
        self.dates: List = []
        self.actions: List = []
        self.positions: List = []
        self.prices: List = []

    def run(self, agent, render: bool = False) -> Dict[str, Any]:
        """Run backtest with a trained agent.

        Args:
            agent: Trained RL agent with .act(state, training=False) method.
            render: If True, print step-by-step info.

        Returns:
            Dict with metrics summary and data.
        """
        state = self.env.reset()
        done = False

        self.equity_curve = [self.initial_capital]
        self.trades = []
        self.dates = []
        self.actions = []
        self.positions = []
        self.prices = []

        cash = self.initial_capital
        holdings = 0.0
        position = 0

        while not done:
            action = agent.act(state, training=False)
            next_state, reward, done, info = self.env.step(action)

            price = info.get("price", 0)
            equity = info.get("equity", cash)
            daily_return = info.get("daily_return", 0.0)

            self.equity_curve.append(equity)
            self.prices.append(price)
            self.actions.append(action)
            self.positions.append(info.get("position", 0))

            if hasattr(self.env, "df") and "date" in self.env.df.columns:
                idx = min(info.get("step", 0), len(self.env.df) - 1)
                self.dates.append(self.env.df["date"].iloc[idx])

            # Track trade entries/exits
            new_position = info.get("position", 0)
            if new_position != position:
                if position == 0 and new_position != 0:
                    # Entry
                    self.trades.append({
                        "entry_step": info.get("step", 0),
                        "entry_price": price,
                        "position": new_position,
                        "entry_equity": equity,
                    })
                elif position != 0 and new_position == 0:
                    # Exit
                    if self.trades:
                        trade = self.trades[-1]
                        trade["exit_step"] = info.get("step", 0)
                        trade["exit_price"] = price
                        trade["exit_equity"] = equity
                        if trade["position"] == 1:
                            trade["pnl"] = (price - trade["entry_price"]) / trade["entry_price"]
                        else:
                            trade["pnl"] = (trade["entry_price"] - price) / trade["entry_price"]
                        trade["pnl_abs"] = equity - trade["entry_equity"]
                position = new_position

            state = next_state

            if render and info.get("step", 0) % 50 == 0:
                print(f"Step {info['step']:5d} | Equity: {equity:12.2f} | "
                      f"Position: {info.get('position', 0)} | "
                      f"Daily R: {daily_return:.4%}")

        # Compute metrics
        metrics = self._compute_metrics()

        if render:
            self._print_summary(metrics)

        return {
            "metrics": metrics,
            "equity_curve": self.equity_curve,
            "trades": self.trades,
            "dates": self.dates,
            "actions": self.actions,
            "positions": self.positions,
            "prices": self.prices,
        }

    def _compute_metrics(self) -> Dict[str, float]:
        """Compute comprehensive backtest metrics."""
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / (equity[:-1] + 1e-8)

        total_return = (equity[-1] - equity[0]) / equity[0]

        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(returns) > 1:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            annualized_return = np.mean(returns) * 252
            annualized_vol = np.std(returns) * np.sqrt(252)
        else:
            sharpe = annualized_return = annualized_vol = 0.0

        # Maximum drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / (peak + 1e-8)
        max_drawdown = float(np.min(drawdown))

        # Calmar ratio
        calmar = annualized_return / (abs(max_drawdown) + 1e-8)

        # Win rate and profit factor
        completed_trades = [t for t in self.trades if "pnl" in t]
        if completed_trades:
            wins = [t for t in completed_trades if t["pnl"] > 0]
            losses = [t for t in completed_trades if t["pnl"] <= 0]
            win_rate = len(wins) / len(completed_trades)
            total_wins = sum(t["pnl"] for t in wins) if wins else 0
            total_losses = abs(sum(t["pnl"] for t in losses)) if losses else 1e-8
            profit_factor = total_wins / total_losses
            avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
            avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
            num_trades = len(completed_trades)
        else:
            win_rate = profit_factor = avg_win = avg_loss = 0.0
            num_trades = 0

        # Sortino ratio
        if len(returns) > 1:
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns) * np.sqrt(252)
                sortino = annualized_return / (downside_std + 1e-8)
            else:
                sortino = 999.0
        else:
            sortino = 0.0

        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "annualized_volatility": float(annualized_vol),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_drawdown),
            "calmar_ratio": float(calmar),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "num_trades": num_trades,
            "final_equity": float(equity[-1]),
            "initial_capital": self.initial_capital,
        }

    def _print_summary(self, metrics: Dict[str, float]):
        print("\n" + "=" * 60)
        print("  BACKTEST RESULTS")
        print("=" * 60)
        print(f"  Total Return:        {metrics['total_return']:>10.2%}")
        print(f"  Annualized Return:   {metrics['annualized_return']:>10.2%}")
        print(f"  Annualized Vol:      {metrics['annualized_volatility']:>10.2%}")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
        print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>10.2f}")
        print(f"  Max Drawdown:        {metrics['max_drawdown']:>10.2%}")
        print(f"  Calmar Ratio:        {metrics['calmar_ratio']:>10.2f}")
        print(f"  Win Rate:            {metrics['win_rate']:>10.2%}")
        print(f"  Profit Factor:       {metrics['profit_factor']:>10.2f}")
        print(f"  Avg Win / Avg Loss:  {metrics['avg_win']:.2%} / {metrics['avg_loss']:.2%}")
        print(f"  Total Trades:        {metrics['num_trades']:>10d}")
        print(f"  Final Equity:        {metrics['final_equity']:>12.2f}")
        print("=" * 60)

    def export_csv(self, output_dir: str = "./output"):
        """Export equity curve and trades to CSV."""
        os.makedirs(output_dir, exist_ok=True)

        # Equity curve
        eq_df = pd.DataFrame({
            "step": range(len(self.equity_curve)),
            "equity": self.equity_curve,
        })
        if self.dates:
            eq_df["date"] = self.dates + [self.dates[-1]] if len(self.dates) == len(self.equity_curve) - 1 else self.dates
        eq_path = os.path.join(output_dir, "equity_curve.csv")
        eq_df.to_csv(eq_path, index=False)
        print(f"Equity curve saved to {eq_path}")

        # Trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_path = os.path.join(output_dir, "trades.csv")
            trades_df.to_csv(trades_path, index=False)
            print(f"Trades saved to {trades_path}")

    def export_summary(self, metrics: Dict, output_dir: str = "./output"):
        """Export metrics summary to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
        print(f"Summary saved to {path}")

    def plot(self, output_dir: str = "./output", show: bool = False) -> str:
        """Generate Plotly charts: equity curve, drawdown, trade markers,
        monthly returns heatmap.

        Returns path to the HTML report.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("plotly is required for visualization. Install: pip install plotly")
            return ""

        os.makedirs(output_dir, exist_ok=True)
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / (peak + 1e-8)

        # Build x-axis
        if self.dates:
            x_axis = self.dates + [self.dates[-1]] if len(self.dates) == len(equity) - 1 else self.dates
            x_axis = x_axis[:len(equity)]
        else:
            x_axis = list(range(len(equity)))

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.4, 0.2, 0.2, 0.2],
            subplot_titles=("Equity Curve", "Drawdown", "Trade P&L", "Position")
        )

        # Equity curve
        fig.add_trace(
            go.Scatter(x=x_axis, y=equity, mode="lines",
                       name="Equity", line=dict(color="blue", width=2)),
            row=1, col=1,
        )
        fig.add_hline(y=self.initial_capital, line_dash="dash",
                      line_color="gray", row=1, col=1)

        # Drawdown
        fig.add_trace(
            go.Scatter(x=x_axis, y=drawdown, mode="lines",
                       fill="tozeroy", name="Drawdown",
                       line=dict(color="red", width=1)),
            row=2, col=1,
        )

        # Trade markers on equity curve
        completed_trades = [t for t in self.trades if "pnl" in t]
        if completed_trades:
            entry_x = []
            entry_y = []
            exit_x = []
            exit_y = []
            colors = []
            for t in completed_trades:
                if "entry_step" in t and "exit_step" in t:
                    e_idx = min(t["entry_step"], len(x_axis) - 1)
                    x_idx = min(t["exit_step"], len(x_axis) - 1)
                    entry_x.append(x_axis[e_idx] if e_idx < len(x_axis) else x_axis[-1])
                    entry_y.append(t.get("entry_equity", equity[0]))
                    exit_x.append(x_axis[x_idx] if x_idx < len(x_axis) else x_axis[-1])
                    exit_y.append(t.get("exit_equity", equity[-1]))
                    colors.append("green" if t["pnl"] > 0 else "red")

            fig.add_trace(
                go.Scatter(x=entry_x, y=entry_y, mode="markers",
                           marker=dict(symbol="triangle-up", size=10, color=colors),
                           name="Entry"),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=exit_x, y=exit_y, mode="markers",
                           marker=dict(symbol="triangle-down", size=10, color=colors),
                           name="Exit"),
                row=1, col=1,
            )

        # Trade P&L distribution
        if completed_trades:
            trade_pnls = [t["pnl"] * 100 for t in completed_trades]
            fig.add_trace(
                go.Bar(x=list(range(len(trade_pnls))), y=trade_pnls,
                       marker_color=["green" if p > 0 else "red" for p in trade_pnls],
                       name="Trade P&L %"),
                row=3, col=1,
            )

        # Position over time
        if len(self.positions) > 0:
            pos_x = x_axis[:len(self.positions)] if len(self.dates) else list(range(len(self.positions)))
            fig.add_trace(
                go.Scatter(x=pos_x, y=self.positions, mode="lines",
                           name="Position", line=dict(color="purple", width=1),
                           fill="tozeroy"),
                row=4, col=1,
            )

        fig.update_layout(
            title="RL Trader Backtest Report",
            height=1000,
            showlegend=True,
            hovermode="x unified",
        )
        fig.update_xaxes(title_text="Date" if self.dates else "Step", row=4, col=1)
        fig.update_yaxes(title_text="Equity", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
        fig.update_yaxes(title_text="P&L %", row=3, col=1)
        fig.update_yaxes(title_text="Position", row=4, col=1)

        html_path = os.path.join(output_dir, "backtest_report.html")
        fig.write_html(html_path)

        # Also save as PNG if possible
        try:
            png_path = os.path.join(output_dir, "backtest_report.png")
            fig.write_image(png_path, width=1200, height=1000)
        except Exception:
            pass

        if show:
            fig.show()

        print(f"Backtest report saved to {html_path}")
        return html_path
