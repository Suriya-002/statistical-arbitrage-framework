import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from itertools import combinations
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class CointegrationTester:
    """
    Screen pairs for cointegration using Engle-Granger two-step method.

    Methodology:
    1. Pre-filter by correlation (must be > threshold)
    2. Run Engle-Granger test on each candidate pair
    3. Estimate half-life of mean reversion via OLS
    4. Filter by half-life bounds
    """

    def __init__(self, significance=0.05, min_half_life=5, max_half_life=120,
                 min_correlation=0.5, rolling_window=252):
        self.significance = significance
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.min_correlation = min_correlation
        self.rolling_window = rolling_window

    def engle_granger_test(self, y: pd.Series, x: pd.Series) -> Dict:
        """
        Engle-Granger two-step cointegration test.
        Step 1: OLS regression y on x for hedge ratio
        Step 2: ADF test on residuals for stationarity
        """
        x_with_const = np.column_stack([np.ones(len(x)), x.values])
        beta = np.linalg.lstsq(x_with_const, y.values, rcond=None)[0]
        spread = y.values - beta[1] * x.values - beta[0]

        adf_stat, p_value, _, _, critical_values, _ = adfuller(spread, autolag='AIC')
        coint_stat, coint_pvalue, _ = coint(y, x)

        return {
            'adf_statistic': adf_stat, 'adf_pvalue': p_value,
            'coint_statistic': coint_stat, 'coint_pvalue': coint_pvalue,
            'hedge_ratio': beta[1], 'intercept': beta[0],
            'spread': spread, 'critical_values': critical_values
        }

    def estimate_half_life(self, spread: np.ndarray) -> float:
        """
        Half-life of mean reversion via OLS.
        Model: delta_spread(t) = theta * spread(t-1) + eps
        Half-life = -ln(2) / theta
        """
        spread_lag = spread[:-1]
        spread_diff = np.diff(spread)
        x = np.column_stack([np.ones(len(spread_lag)), spread_lag])
        theta = np.linalg.lstsq(x, spread_diff, rcond=None)[0][1]

        if theta >= 0:
            return np.inf
        return -np.log(2) / theta

    def screen_universe(self, prices: pd.DataFrame) -> List[Dict]:
        """Screen all pairs for cointegration, sorted by half-life."""
        tickers = prices.columns.tolist()
        corr_matrix = prices.pct_change().dropna().corr()
        valid_pairs = []

        for ticker_y, ticker_x in combinations(tickers, 2):
            correlation = corr_matrix.loc[ticker_y, ticker_x]
            if abs(correlation) < self.min_correlation:
                continue

            y = prices[ticker_y].dropna()
            x = prices[ticker_x].dropna()
            common_idx = y.index.intersection(x.index)
            if len(common_idx) < self.rolling_window:
                continue

            y, x = y.loc[common_idx], x.loc[common_idx]

            try:
                results = self.engle_granger_test(y, x)
            except Exception as e:
                logger.warning(f"EG test failed for {ticker_y}-{ticker_x}: {e}")
                continue

            if results['coint_pvalue'] > self.significance:
                continue

            half_life = self.estimate_half_life(results['spread'])
            if not (self.min_half_life <= half_life <= self.max_half_life):
                continue

            valid_pairs.append({
                'ticker_y': ticker_y, 'ticker_x': ticker_x,
                'hedge_ratio': results['hedge_ratio'],
                'intercept': results['intercept'],
                'coint_pvalue': results['coint_pvalue'],
                'adf_pvalue': results['adf_pvalue'],
                'half_life': half_life, 'correlation': correlation
            })

        valid_pairs.sort(key=lambda p: p['half_life'])
        logger.info(f"Found {len(valid_pairs)} cointegrated pairs")
        return valid_pairs

    def rolling_cointegration_check(self, y, x, window=252, step=21):
        """Rolling cointegration test to monitor pair stability."""
        results = []
        for end in range(window, len(y), step):
            try:
                test = self.engle_granger_test(y.iloc[end-window:end], x.iloc[end-window:end])
                hl = self.estimate_half_life(test['spread'])
                results.append({
                    'date': y.index[end-1], 'coint_pvalue': test['coint_pvalue'],
                    'hedge_ratio': test['hedge_ratio'], 'half_life': hl,
                    'is_cointegrated': test['coint_pvalue'] < self.significance
                })
            except Exception:
                continue
        return pd.DataFrame(results).set_index('date')
