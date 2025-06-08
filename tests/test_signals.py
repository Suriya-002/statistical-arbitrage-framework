import numpy as np
import pandas as pd

def test_zscore_symmetry():
    np.random.seed(42)
    spread = pd.Series(np.random.normal(0, 1, 500))
    zscore = ((spread - spread.rolling(60).mean()) / spread.rolling(60).std()).dropna()
    assert abs(zscore.mean()) < 0.3

def test_position_entry():
    z = np.linspace(-3, 3, 100)
    assert np.argmax(z > 2.0) > 0
