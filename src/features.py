"""Data cleaning and feature engineering for the freight-rate model."""
from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]

# Features fed to the gradient-boosted trees.
# NOTE: market_index is deliberately NOT here. It is a date-level signal with a
# regime shift between train (Jan-Oct) and validation (Nov-Dec); trees memorise
# individual days. It is applied as a linear second-stage adjustment instead.
TREE_FEATURES = [
    "equipment", "distance", "log_distance", "weight", "weight_missing",
    "quote_signal", "dow", "circuity",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
]


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix data-quality issues. Works on train and validation files."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # 1) Negative weights are sign errors (|min| == max == 47,500), not real loads.
    df["weight"] = df["weight"].abs()
    df["weight_missing"] = df["weight"].isna().astype(int)  # keep NaN; trees handle it
    # 2) Missing market_index: it is a date-level signal (within-day std ~0.025),
    #    so fill with the median of the same date.
    df["market_index"] = df["market_index"].fillna(df.groupby("date")["market_index"].transform("median"))
    df["equipment"] = pd.Categorical(df["equipment"], categories=EQUIPMENT)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_distance"] = np.log(df["distance"])
    df["dow"] = df["date"].dt.dayofweek
    straight = haversine_miles(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
    df["circuity"] = df["distance"] / straight  # road miles / straight-line miles
    return df


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """city -> (lat, lon) lookup, used to build features for the December chart rows."""
    parts = []
    for f in frames:
        parts.append(f[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1))
        parts.append(f[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).drop_duplicates("city").set_index("city")
