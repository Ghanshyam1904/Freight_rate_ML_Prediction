"""Validation strategy: expanding-window time-based CV + unseen-city check."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import TREE_FEATURES
from .model import FreightRateModel

# Each fold trains on everything BEFORE the window and tests on the next 2 months,
# mirroring the final task (train Jan-Oct -> predict the following 2 months).
FOLDS = [("2025-05-01", "2025-07-01"), ("2025-07-01", "2025-09-01"), ("2025-09-01", "2025-11-01")]


def metrics(pred, actual) -> dict:
    err = np.abs(pred - actual)
    return {"MAE": err.mean(), "RMSE": float(np.sqrt(((pred - actual) ** 2).mean())),
            "MAPE_%": (err / actual).mean() * 100, "MdAPE_%": float(np.median(err / actual)) * 100}


def candidates() -> dict:
    no_city = TREE_FEATURES
    return {
        "A. baseline: median $/mile by equipment": None,
        "B. HGB + market_index inside trees": dict(features=TREE_FEATURES + ["market_index"], use_market_index=False),
        "C. HGB, no market_index": dict(features=no_city, use_market_index=False),
        "D. HGB + linear market_index (CHOSEN)": dict(features=no_city, use_market_index=True),
        "E. D without quote_signal": dict(features=[f for f in no_city if f != "quote_signal"], use_market_index=True),
    }


def time_cv(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lo, hi in FOLDS:
        train, test = df[df.date < lo], df[(df.date >= lo) & (df.date < hi)]
        for name, cfg in candidates().items():
            if cfg is None:
                rpm = (train.posted_rate / train.distance).groupby(train.equipment, observed=True).median()
                pred = test.distance * test.equipment.map(rpm).astype(float)
            else:
                pred = FreightRateModel(**cfg).fit(train).predict(test)
            rows.append({"fold_test_start": lo, "model": name, **metrics(np.asarray(pred), test.posted_rate.to_numpy())})
    res = pd.DataFrame(rows)
    return res


def unseen_city_check(df: pd.DataFrame, n_hold=8, seeds=(0, 1, 2)) -> dict:
    """validation.csv has 8 cities never seen in training (12% of rows).
    Simulate it: hold out 8 cities entirely from training, test on them in Sep-Oct."""
    cities = np.array(sorted(set(df.pickup) | set(df.delivery)))
    out = []
    for s in seeds:
        hold = set(np.random.RandomState(s).choice(cities, n_hold, replace=False))
        unseen = df.pickup.isin(hold) | df.delivery.isin(hold)
        train = df[(df.date < "2025-09-01") & ~unseen]
        test = df[(df.date >= "2025-09-01") & unseen]
        pred = FreightRateModel().fit(train).predict(test)
        out.append(metrics(pred, test.posted_rate.to_numpy()))
    return pd.DataFrame(out).mean().to_dict()
