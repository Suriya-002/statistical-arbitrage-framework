import numpy as np
import pandas as pd

def generate_cointegrated_pair(n=500, beta=1.5, mr=0.1):
    np.random.seed(42)
    x_ret = np.random.normal(0.0005, 0.02, n)
    x_px = 100 * np.exp(np.cumsum(x_ret))
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = (1 - mr) * spread[i-1] + np.random.normal(0, 0.5)
    y_px = beta * x_px + spread + 50
    return pd.DataFrame({'Y': y_px, 'X': x_px},
                         index=pd.date_range('2020-01-01', periods=n, freq='B'))

def test_synthetic_cointegrated():
    data = generate_cointegrated_pair()
    assert len(data) == 500

def test_half_life_positive():
    data = generate_cointegrated_pair()
    spread = data['Y'].values - 1.5 * data['X'].values
    x = np.column_stack([np.ones(len(spread)-1), spread[:-1]])
    theta = np.linalg.lstsq(x, np.diff(spread), rcond=None)[0][1]
    assert theta < 0
    assert -np.log(2)/theta > 0
