import numpy as np
import pandas as pd
from typing import Dict, Optional
from ..cointegration import CointegrationTester
from ..signals import ZScoreSignalGenerator
from ..estimation import KalmanHedgeRatio
from ..risk import PositionSizer, DrawdownMonitor
import logging

logger = logging.getLogger(__name__)


class PairsBacktester:
    """Event-driven backtest: screen pairs, generate signals, compute P&L."""

    def __init__(self, config: Dict):
        self.config = config
        self.coint_tester = CointegrationTester(
            significance=config['cointegration']['significance'],
            min_half_life=config['cointegration']['min_half_life'],
            max_half_life=config['cointegration']['max_half_life'])
        self.signal_gen = ZScoreSignalGenerator(
            entry_z=config['signals']['entry_z'],
            exit_z=config['signals']['exit_z'],
            stop_z=config['signals']['stop_z'],
            lookback=config['signals']['lookback_zscore'])
        self.kalman = KalmanHedgeRatio()
        self.position_sizer = PositionSizer()
        self.drawdown_monitor = DrawdownMonitor(
            max_drawdown=config['risk']['max_portfolio_drawdown'])

    def run(self, prices, start_date=None, end_date=None):
        if start_date: prices = prices.loc[start_date:]
        if end_date: prices = prices.loc[:end_date]

        window = self.config['cointegration']['rolling_window']
        pairs = self.coint_tester.screen_universe(prices.iloc[:window])
        if not pairs:
            return {'metrics': {}, 'portfolio_returns': pd.Series()}

        pairs = pairs[:self.config['execution']['max_pairs']]
        trading = prices.iloc[window:]
        pair_pnls = {}

        for pair in pairs:
            y, x = trading[pair['ticker_y']], trading[pair['ticker_x']]
            self.kalman.filter(y, x)
            signals = self.signal_gen.generate_signals(y, x, pair['hedge_ratio'], pair['intercept'])
            pnl = self.signal_gen.compute_signal_pnl(signals, y, x, pair['hedge_ratio'])
            pair_pnls[f"{pair['ticker_y']}/{pair['ticker_x']}"] = pnl['net_return']

        all_ret = pd.DataFrame(pair_pnls)
        port_ret = all_ret.mean(axis=1)
        return {'portfolio_returns': port_ret, 'pair_results': all_ret,
                'metrics': self._compute_metrics(port_ret), 'pairs_selected': pairs}

    @staticmethod
    def _compute_metrics(returns):
        r = returns.dropna()
        if len(r) == 0: return {}
        total = (1 + r).prod() - 1
        ann = (1 + total) ** (252/len(r)) - 1
        vol = r.std() * np.sqrt(252)
        cum = (1 + r).cumprod()
        dd = ((cum - cum.cummax()) / cum.cummax()).min()
        wins = (r > 0).sum()
        trades = (r != 0).sum()
        gp = r[r > 0].sum()
        gl = abs(r[r < 0].sum())
        return {
            'total_return': total, 'annualized_return': ann,
            'annualized_volatility': vol,
            'sharpe_ratio': ann/vol if vol else 0,
            'max_drawdown': dd,
            'win_rate': wins/trades if trades else 0,
            'profit_factor': gp/gl if gl else np.inf,
            'num_trading_days': len(r),
            'avg_daily_return': r.mean(),
            'skewness': r.skew(), 'kurtosis': r.kurtosis()
        }
