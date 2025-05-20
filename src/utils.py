import pandas as pd
import numpy as np
from typing import Dict

def format_performance_table(metrics: Dict) -> str:
    lines = ["="*55, "  PERFORMANCE SUMMARY", "="*55]
    for k, v in metrics.items():
        if isinstance(v, float):
            fmt = f"{v:>10.2%}" if any(x in k for x in ['return','drawdown','rate']) else f"{v:>10.4f}"
            lines.append(f"  {k:35s}: {fmt}")
    lines.append("="*55)
    return "\n".join(lines)

def compute_rolling_sharpe(returns, window=63):
    return (returns.rolling(window).mean() / returns.rolling(window).std()) * np.sqrt(252)

def compute_turnover(positions):
    return positions.diff().abs().sum(axis=1)

def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)
