import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    LONG_SPREAD = 1
    SHORT_SPREAD = -1
    FLAT = 0


@dataclass
class Signal:
    timestamp: pd.Timestamp
    signal_type: SignalType
    z_score: float
    spread_value: float
    hedge_ratio: float
    confidence: float


class ZScoreSignalGenerator:
    """
    Mean-reversion signal generator based on spread z-score.
    Entry: |z| > entry_z, Exit: |z| < exit_z, Stop: |z| > stop_z
    """

    def __init__(self, entry_z=2.0, exit_z=0.5, stop_z=3.5,
                 lookback=60, use_ewm=True, ewm_halflife=30):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.lookback = lookback
        self.use_ewm = use_ewm
        self.ewm_halflife = ewm_halflife

    def compute_spread(self, y, x, hedge_ratio, intercept=0.0):
        return y - hedge_ratio * x - intercept

    def compute_zscore(self, spread):
        if self.use_ewm:
            mean = spread.ewm(halflife=self.ewm_halflife).mean()
            std = spread.ewm(halflife=self.ewm_halflife).std()
        else:
            mean = spread.rolling(self.lookback).mean()
            std = spread.rolling(self.lookback).std()
        return (spread - mean) / std

    def generate_signals(self, y, x, hedge_ratio, intercept=0.0):
        """Generate entry/exit signals based on z-score thresholds."""
        spread = self.compute_spread(y, x, hedge_ratio, intercept)
        zscore = self.compute_zscore(spread)

        signals = pd.DataFrame(index=spread.index)
        signals['spread'] = spread
        signals['zscore'] = zscore
        signals['position'] = 0

        position = 0
        for i in range(1, len(signals)):
            z = zscore.iloc[i]
            if np.isnan(z):
                continue

            if position == 0:
                if z < -self.entry_z:
                    position = 1
                elif z > self.entry_z:
                    position = -1
            elif position == 1:
                if z > -self.exit_z or z > self.stop_z:
                    position = 0
            elif position == -1:
                if z < self.exit_z or z < -self.stop_z:
                    position = 0

            signals.iloc[i, signals.columns.get_loc('position')] = position

        signals['confidence'] = np.clip(
            (np.abs(signals['zscore']) - self.exit_z) /
            (self.entry_z - self.exit_z), 0, 1)
        return signals

    def compute_signal_pnl(self, signals, y, x, hedge_ratio, tc_bps=5.0):
        """Compute strategy P&L with transaction costs."""
        pnl = pd.DataFrame(index=signals.index)
        spread_ret = y.pct_change() - hedge_ratio * x.pct_change()
        pnl['strategy_return'] = signals['position'].shift(1) * spread_ret
        pnl['transaction_costs'] = signals['position'].diff().abs() * tc_bps / 10000
        pnl['net_return'] = pnl['strategy_return'] - pnl['transaction_costs']
        pnl['cumulative_return'] = (1 + pnl['net_return']).cumprod()
        return pnl
