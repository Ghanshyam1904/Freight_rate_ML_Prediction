"""Two-stage rate model.

Stage 1: HistGradientBoosting on log(rate per mile), absolute-error loss
         (robust to the ~1.4% of loads with extreme rate spikes).
Stage 2: Huber-linear adjustment of the stage-1 residual on market_index.
Prediction: rate = distance * exp(stage1 + stage2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor

from .features import TREE_FEATURES


class FreightRateModel:
    def __init__(self, features=None, use_market_index=True, seed=0, **hgb_params):
        self.features = list(features or TREE_FEATURES)
        self.use_market_index = use_market_index
        params = dict(loss="absolute_error", learning_rate=0.05, max_iter=300,
                      categorical_features="from_dtype", random_state=seed)
        params.update(hgb_params)
        self.tree = HistGradientBoostingRegressor(**params)
        self.adj = HuberRegressor()

    def fit(self, df: pd.DataFrame):
        y = np.log(df["posted_rate"] / df["distance"])
        self.tree.fit(df[self.features], y)
        if self.use_market_index:
            resid = y - self.tree.predict(df[self.features])
            self.adj.fit(df[["market_index"]], resid)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        log_rpm = self.tree.predict(df[self.features])
        if self.use_market_index:
            log_rpm = log_rpm + self.adj.predict(df[["market_index"]])
        return df["distance"].to_numpy() * np.exp(log_rpm)
