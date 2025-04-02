import numpy as np
import pandas as pd
from typing import Tuple


class KalmanHedgeRatio:
    """
    Online Kalman Filter for dynamic hedge ratio estimation.
    State: beta_t, Observation: y_t = beta_t * x_t + eps
    Transition: beta_t = beta_{t-1} + eta (random walk)
    """

    def __init__(self, delta=1e-4, observation_noise=1.0,
                 initial_mean=0.0, initial_cov=1.0):
        self.delta = delta
        self.Ve = observation_noise
        self.initial_mean = initial_mean
        self.initial_cov = initial_cov

    def filter(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        """Run Kalman Filter to estimate dynamic hedge ratio."""
        n = len(y)
        beta, P = np.zeros(n), np.zeros(n)
        spread, forecast_error = np.zeros(n), np.zeros(n)

        beta[0], P[0] = self.initial_mean, self.initial_cov
        Vw = self.delta / (1 - self.delta)

        for t in range(1, n):
            beta_pred = beta[t-1]
            P_pred = P[t-1] + Vw
            x_t, y_t = x.iloc[t], y.iloc[t]

            e = y_t - beta_pred * x_t
            S = x_t * P_pred * x_t + self.Ve
            K = P_pred * x_t / S

            beta[t] = beta_pred + K * e
            P[t] = (1 - K * x_t) * P_pred
            spread[t] = e
            forecast_error[t] = e

        results = pd.DataFrame(index=y.index)
        results['hedge_ratio'] = beta
        results['hedge_ratio_std'] = np.sqrt(P)
        results['spread'] = y.values - beta * x.values
        results['forecast_error'] = forecast_error
        return results

    def online_update(self, y_new, x_new, beta_prev, P_prev):
        """Single-step update for live trading."""
        Vw = self.delta / (1 - self.delta)
        P_pred = P_prev + Vw
        S = x_new * P_pred * x_new + self.Ve
        K = P_pred * x_new / S
        e = y_new - beta_prev * x_new
        return beta_prev + K * e, (1 - K * x_new) * P_pred, e
