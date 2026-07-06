"""
Trading Environment
===================
A custom trading environment for reinforcement learning agents.
State includes price, holdings, cash, returns, and technical indicators.
Supports discrete (Short/Hold/Long) and continuous [-1, 1] actions.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Union
from collections import deque


class RunningStats:
    """Online mean/std normalization using Welford's algorithm."""

    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
            self.count = batch_count
            return

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.var = M2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.var + 1e-8)
        return np.clip((x - self.mean) / std, -10, 10)


class TradingEnv:
    """Trading environment for RL agents.

    State vector (14 features):
        [price_norm, holdings_norm, cash_norm,
         returns_5d, returns_10d, returns_20d,
         rsi, macd, macd_signal, bbands_upper, bbands_lower,
         volume_norm, sma_ratio]

    Actions:
        discrete: 0 = Short, 1 = Hold, 2 = Long
        continuous: float in [-1, 1], negative = short, positive = long

    Reward:
        sharpe_like = mean(daily_return) / (std(daily_return) + 1e-8) for a window
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, str],
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        max_steps: Optional[int] = None,
        stop_loss: float = 0.05,
        take_profit: float = 0.10,
        window_size: int = 50,
        action_type: str = "discrete",
        reward_type: str = "sharpe_like",
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.max_steps = max_steps
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.window_size = window_size
        self.action_type = action_type
        self.reward_type = reward_type

        # Load data
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data.copy()

        self._validate_and_prepare()
        self._compute_indicators()
        self._compute_state_cols()

        self.state_dim = len(self.state_columns)
        self.action_dim = 3 if action_type == "discrete" else 1

        self.stats = RunningStats()
        self._precompute_stats()

        self.reset()

    def _validate_and_prepare(self):
        """Ensure required columns exist."""
        required = {"open", "high", "low", "close", "volume"}
        df_cols = set(c.lower() for c in self.df.columns)

        if not required.issubset(df_cols):
            # Try renaming common Chinese column names
            rename = {
                "开盘": "open", "最高": "high", "最低": "low",
                "收盘": "close", "成交量": "volume",
            }
            self.df.rename(columns={k: v for k, v in rename.items()
                                    if k in self.df.columns}, inplace=True)
            df_cols = set(c.lower() for c in self.df.columns)

        if not required.issubset(df_cols):
            missing = required - df_cols
            raise ValueError(
                f"DataFrame missing required columns: {missing}. "
                f"Columns found: {list(self.df.columns)}. "
                f"Expected: open, high, low, close, volume (case insensitive)"
            )

        # Normalize column names to lowercase
        self.df.columns = [c.lower() for c in self.df.columns]
        self.df = self.df.sort_values("open").reset_index(drop=True)  # fallback sort
        # Try to use a date column if available, else sort by index
        if "date" in self.df.columns:
            self.df["date"] = pd.to_datetime(self.df["date"])
            self.df = self.df.sort_values("date").reset_index(drop=True)

        self.close = self.df["close"].values.astype(np.float64)
        self.high = self.df["high"].values.astype(np.float64)
        self.low = self.df["low"].values.astype(np.float64)
        self.volume = self.df["volume"].values.astype(np.float64)

    def _compute_indicators(self):
        """Compute technical indicators."""
        close = self.close

        # Returns
        self.returns = np.diff(close) / (close[:-1] + 1e-8)
        self.returns = np.concatenate([[0], self.returns])

        # SMA
        def _sma(data, period):
            result = np.full_like(data, np.nan)
            cumsum = np.cumsum(np.insert(data, 0, 0))
            result[period - 1:] = (cumsum[period:] - cumsum[:-period]) / period
            return result

        sma5 = _sma(close, 5)
        sma10 = _sma(close, 10)
        sma20 = _sma(close, 20)
        sma50 = _sma(close, 50)
        sma200 = _sma(close, min(200, len(close) // 2))

        self.sma_ratio = np.where(sma50 > 0, close / sma50, 1.0)
        self.sma_ratio = np.nan_to_num(self.sma_ratio, nan=1.0)

        # RSI (14-period)
        delta = np.diff(close)
        delta = np.insert(delta, 0, 0)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = _sma(gain, 14)
        avg_loss = _sma(loss, 14)
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
        self.rsi = 100 - (100 / (1 + rs))
        self.rsi = np.nan_to_num(self.rsi, nan=50.0)

        # MACD
        ema12 = self._ema(close, 12)
        ema26 = self._ema(close, 26)
        self.macd = ema12 - ema26
        self.macd_signal = self._ema(self.macd, 9)

        # Bollinger Bands (20-period)
        bb_mid = sma20
        bb_std = np.array([np.nanstd(close[max(0, i - 19):i + 1])
                           if i >= 19 else np.nan for i in range(len(close))])
        bb_std = np.nan_to_num(bb_std, nan=np.nanmean(bb_std[19:]) if len(close) > 19 else 1.0)
        self.bb_upper = bb_mid + 2 * bb_std
        self.bb_lower = bb_mid - 2 * bb_std

        # Volume normalized
        vol_ma = _sma(self.volume, 20)
        self.volume_norm = np.where(vol_ma > 0, self.volume / vol_ma, 1.0)
        self.volume_norm = np.nan_to_num(self.volume_norm, nan=1.0)

    @staticmethod
    def _ema(data, period):
        result = np.zeros_like(data)
        result[0] = data[0]
        alpha = 2.0 / (period + 1)
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    def _compute_state_cols(self):
        """Define state column names for reference."""
        self.state_columns = [
            "price", "holdings", "cash",
            "returns_5", "returns_10", "returns_20",
            "rsi", "macd", "macd_signal",
            "bb_upper", "bb_lower", "volume",
            "sma_ratio",
        ]

    def _precompute_stats(self):
        """Precompute normalization stats over the warmup period."""
        warmup = self.window_size
        for i in range(warmup, min(warmup + 500, len(self.close))):
            s = self._raw_state(i)
            self.stats.update(s.reshape(1, -1))

    def _raw_state(self, idx: int) -> np.ndarray:
        """Build raw un-normalized state vector at index."""
        start = max(0, idx - max(self.window_size, 20))
        price = self.close[idx]

        # Price returns over different windows
        if idx > 0:
            ret_5 = (self.close[idx] / self.close[max(0, idx - 5)] - 1) if idx >= 5 else 0.0
            ret_10 = (self.close[idx] / self.close[max(0, idx - 10)] - 1) if idx >= 10 else 0.0
            ret_20 = (self.close[idx] / self.close[max(0, idx - 20)] - 1) if idx >= 20 else 0.0
        else:
            ret_5 = ret_10 = ret_20 = 0.0

        state = np.array([
            price,
            self._holdings,
            self._cash,
            ret_5,
            ret_10,
            ret_20,
            self.rsi[idx],
            self.macd[idx],
            self.macd_signal[idx],
            self.bb_upper[idx],
            self.bb_lower[idx],
            self.volume_norm[idx],
            self.sma_ratio[idx],
        ], dtype=np.float32)

        return state

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        return self.stats.normalize(state.reshape(1, -1)).flatten()

    def reset(self) -> np.ndarray:
        """Reset the environment and return initial state."""
        self._step = self.window_size
        self._cash = self.initial_capital
        self._holdings = 0.0  # number of shares held
        self._position = 0    # 0=none, 1=long, -1=short
        self._entry_price = 0.0
        self._returns_history = deque(maxlen=20)
        self._equity_history = deque(maxlen=100)

        if self.max_steps is None:
            self.max_steps = len(self.close) - self.window_size - 1

        raw_state = self._raw_state(self._step)
        return self._normalize_state(raw_state)

    def step(self, action: Union[int, float]) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment."""
        if self.action_type == "discrete":
            action = int(action)
        else:
            action = float(np.clip(action, -1.0, 1.0))

        prev_equity = self._cash + self._holdings * self.close[self._step]
        self._equity_history.append(prev_equity)

        # Parse action
        if self.action_type == "discrete":
            self._execute_discrete_action(action)
        else:
            self._execute_continuous_action(action)

        self._step += 1
        current_price = self.close[self._step]
        current_equity = self._cash + self._holdings * current_price
        self._equity_history.append(current_equity)

        # Daily return
        if len(self._equity_history) >= 2 and self._equity_history[-2] > 0:
            daily_return = (current_equity - self._equity_history[-2]) / self._equity_history[-2]
        else:
            daily_return = 0.0
        self._returns_history.append(daily_return)

        # Calculate reward
        reward = self._calculate_reward()

        # Check termination
        done = False
        info = {"equity": current_equity, "step": self._step, "price": current_price,
                "position": self._position, "daily_return": daily_return}

        # Stop loss / take profit
        if self._position != 0 and self._entry_price > 0:
            pnl_pct = (current_price - self._entry_price) / self._entry_price
            if self._position == 1:  # Long
                if pnl_pct <= -self.stop_loss:
                    done = True
                    info["termination"] = "stop_loss"
                elif pnl_pct >= self.take_profit:
                    done = True
                    info["termination"] = "take_profit"
            else:  # Short
                if pnl_pct >= self.stop_loss:
                    done = True
                    info["termination"] = "stop_loss"
                elif pnl_pct <= -self.take_profit:
                    done = True
                    info["termination"] = "take_profit"

        # Max steps
        if self._step >= len(self.close) - 1:
            done = True
            info["termination"] = "end_of_data"
        elif self._step - self.window_size >= self.max_steps:
            done = True
            info["termination"] = "max_steps"

        # Liquidation
        if current_equity <= self.initial_capital * 0.1:
            done = True
            info["termination"] = "bankruptcy"

        next_state = self._raw_state(self._step)
        return self._normalize_state(next_state), reward, done, info

    def _execute_discrete_action(self, action: int):
        """0=Short, 1=Hold, 2=Long"""
        current_price = self.close[self._step]
        target_position = action - 1  # -1, 0, 1

        if target_position == self._position:
            return  # Hold

        # Close existing position
        if self._position != 0:
            self._cash += self._holdings * current_price - abs(self._holdings) * current_price * self.commission
            self._holdings = 0

        self._position = target_position
        if target_position != 0:
            # Open new position - use all cash (or half for short)
            if target_position == 1:  # Long
                shares = self._cash * 0.99 / current_price
                cost = shares * current_price * (1 + self.commission)
                self._holdings = shares
                self._cash -= cost
            else:  # Short
                shares = self._cash * 0.49 / current_price
                self._holdings = -shares
                self._cash += shares * current_price * (1 - self.commission)
            self._entry_price = current_price

    def _execute_continuous_action(self, action: float):
        """Continuous action in [-1, 1]. Target position = action."""
        current_price = self.close[self._step]
        target_position = np.clip(action, -1.0, 1.0)

        # Close existing if sign flips
        if self._position * target_position < 0 and self._position != 0:
            self._cash += self._holdings * current_price - abs(self._holdings) * current_price * self.commission
            self._holdings = 0
            self._position = 0

        # Adjust to target
        if abs(target_position) > 0:
            if target_position > 0:
                target_holdings = self._cash * target_position / current_price
            else:
                target_holdings = -self._cash * abs(target_position) / current_price

            delta = target_holdings - self._holdings
            if abs(delta) > 1e-6:
                cost = abs(delta) * current_price * (1 + self.commission)
                if target_position > 0 and cost <= self._cash:
                    self._holdings = target_holdings
                    self._cash -= cost
                elif target_position < 0:
                    self._holdings = target_holdings
                    self._cash -= abs(delta) * current_price * self.commission
                    self._cash += -delta * current_price
            self._position = np.sign(self._holdings) if abs(self._holdings) > 1e-6 else 0

    def _calculate_reward(self) -> float:
        if len(self._returns_history) < 2:
            return 0.0

        returns = np.array(list(self._returns_history))

        if self.reward_type == "sharpe_like":
            if len(returns) >= 3:
                mu = np.mean(returns[-20:])
                sigma = np.std(returns[-20:]) + 1e-8
                return mu / sigma
            return float(returns[-1])

        elif self.reward_type == "differential_sharpe":
            if len(self._returns_history) >= 5:
                recent = np.array(list(self._returns_history)[-5:])
                prev = np.array(list(self._returns_history)[-10:-5])
                sr_recent = np.mean(recent) / (np.std(recent) + 1e-8)
                sr_prev = np.mean(prev) / (np.std(prev) + 1e-8) if len(prev) >= 3 else 0
                return sr_recent - sr_prev
            return 0.0

        elif self.reward_type == "pnl":
            return float(returns[-1])

        else:  # default: differential daily return
            return float(returns[-1]) - 0.5 * np.std(returns[-10:]) if len(returns) >= 10 else float(returns[-1])

    def seed(self, seed: Optional[int] = None):
        np.random.seed(seed)


def load_data_from_akshare(ticker: str, start_date: str = "20100101",
                           end_date: str = "20231231", period: str = "daily") -> pd.DataFrame:
    """Load OHLCV data from akshare for Chinese stocks/ETFs.

    ticker: stock code like '000300', '000001', ETF code etc.
    """
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("akshare is required for data loading. Install: pip install akshare")

    # Detect market by ticker prefix
    if ticker.startswith(("60", "68")):
        symbol = f"sh{ticker}"
    elif ticker.startswith(("000", "001", "002", "003", "300", "301")):
        symbol = ticker
    elif ticker.startswith("8") or ticker.startswith("4"):
        symbol = ticker
    else:
        symbol = ticker

    try:
        if period == "daily":
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                    start_date=start_date, end_date=end_date,
                                    adjust="qfq")
        else:
            df = ak.stock_zh_a_hist(symbol=symbol, period=period,
                                    start_date=start_date, end_date=end_date,
                                    adjust="qfq")
    except Exception:
        # Try without adjust parameter for older akshare versions
        try:
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                    start_date=start_date, end_date=end_date)
        except Exception as e:
            raise RuntimeError(f"Failed to load data for {ticker}: {e}")

    if df is None or len(df) == 0:
        raise ValueError(f"No data returned for ticker {ticker}")

    # Map akshare columns to standard names
    col_map = {
        "日期": "date", "开盘": "open", "最高": "high",
        "最低": "low", "收盘": "close", "成交量": "volume",
        "成交额": "amount", "振幅": "amplitude", "涨跌幅": "pct_change",
        "涨跌额": "change", "换手率": "turnover",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
    return df
