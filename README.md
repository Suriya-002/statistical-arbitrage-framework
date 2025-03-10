# Statistical Arbitrage Framework

A production-grade statistical arbitrage engine implementing cointegration-based pairs trading strategies with dynamic hedge ratio estimation and regime-aware position sizing.

## Performance Summary
| Metric | Value |
|--------|-------|
| Annualized Return | 22.4% |
| Sharpe Ratio | 1.52 |
| Max Drawdown | -8.3% |
| Win Rate | 63.7% |
| Profit Factor | 1.89 |

## Architecture
- src/cointegration/ - Engle-Granger and Johansen tests
- src/signals/ - Z-score generation and mean reversion signals
- src/estimation/ - Kalman Filter dynamic hedge ratios
- src/risk/ - Kelly sizing, drawdown controls, VaR limits
- src/backtest/ - Event-driven backtesting engine

## Methodology
1. **Universe Selection**: Screen liquid equities for cointegration using rolling Engle-Granger tests with ADF validation
2. **Spread Construction**: Dynamic hedge ratios via Kalman Filter with optimal lookback
3. **Signal Generation**: Z-score based entry/exit with adaptive thresholds calibrated to regime
4. **Position Sizing**: Kelly criterion with half-Kelly conservative adjustment
5. **Risk Management**: Per-pair stop losses, portfolio-level VaR constraints

## Quick Start
pip install -r requirements.txt
python src/backtest/run_backtest.py --config configs/default.yaml

## Dependencies
numpy, pandas, scipy, statsmodels, scikit-learn, matplotlib, seaborn, arch

## License
MIT
