import numpy as np
import pandas as pd
from typing import Dict
from dataclasses import dataclass


@dataclass
class RiskLimits:
    max_position_size: float = 0.10
    max_sector_exposure: float = 0.30
    max_portfolio_drawdown: float = 0.15
    daily_var_limit: float = 0.02
    max_correlation: float = 0.60


class PositionSizer:
    """Kelly criterion with half-Kelly conservative adjustment."""

    def __init__(self, kelly_fraction=0.5):
        self.kelly_fraction = kelly_fraction

    def compute_kelly_size(self, win_rate, avg_win, avg_loss):
        if avg_loss == 0:
            return 0.0
        b = avg_win / abs(avg_loss)
        kelly = (win_rate * b - (1 - win_rate)) / b
        return max(0, self.kelly_fraction * kelly)

    def size_from_backtest(self, returns, max_size=0.10):
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        wr = len(wins) / len(returns[returns != 0])
        return min(self.compute_kelly_size(wr, wins.mean(), losses.mean()), max_size)


class DrawdownMonitor:
    """Track portfolio drawdown and enforce stop-loss limits."""

    def __init__(self, max_drawdown=0.15):
        self.max_drawdown = max_drawdown
        self.peak = 1.0
        self.current_value = 1.0
        self.is_stopped = False

    def update(self, value):
        self.current_value = value
        self.peak = max(self.peak, value)
        dd = (self.peak - self.current_value) / self.peak
        if dd >= self.max_drawdown:
            self.is_stopped = True
        return {'drawdown': dd, 'peak': self.peak, 'is_stopped': self.is_stopped}

    def reset(self):
        self.peak = self.current_value
        self.is_stopped = False


class VaRCalculator:
    """Historical simulation Value at Risk."""

    @staticmethod
    def historical_var(returns, confidence=0.95, horizon=1):
        sorted_r = returns.dropna().sort_values()
        idx = int(len(sorted_r) * (1 - confidence))
        return abs(sorted_r.iloc[idx]) * np.sqrt(horizon)

    @staticmethod
    def expected_shortfall(returns, confidence=0.95):
        sorted_r = returns.dropna().sort_values()
        idx = int(len(sorted_r) * (1 - confidence))
        tail = sorted_r.iloc[:idx]
        return abs(tail.mean()) if len(tail) > 0 else 0.0
