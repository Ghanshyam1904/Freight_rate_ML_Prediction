"""End-to-end pipeline: EDA checks -> validation -> final fit -> predictions.

Usage:  python train.py
Outputs: validation_predictions.csv, december_predictions.csv, reports/*.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features import add_features, city_coordinates, clean
from src.model import FreightRateModel
from src.validate import time_cv, unseen_city_check

DATA = Path("data")
REPORTS = Path("reports")


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    raw_train = pd.read_csv(DATA / "train_test.csv")
    raw_val = pd.read_csv(DATA / "validation.csv")
    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    dec = pd.read_csv(DATA / "december_chart_inputs.csv")

    # ---- data-quality summary (used in the report / Loom) -----------------
    quality = {
        "train_rows": len(raw_train),
        "train_date_range": f"{raw_train.date.min()} .. {raw_train.date.max()}",
        "validation_date_range": f"{raw_val.date.min()} .. {raw_val.date.max()}",
        "negative_weights_train": int((raw_train.weight < 0).sum()),
        "negative_weights_validation": int((raw_val.weight < 0).sum()),
        "missing_weight_train": int(raw_train.weight.isna().sum()),
        "missing_market_index_train": int(raw_train.market_index.isna().sum()),
        "rate_per_mile_gt_4_train": int((raw_train.posted_rate / raw_train.distance > 4).sum()),
        "rate_per_mile_lt_1_train": int((raw_train.posted_rate / raw_train.distance < 1).sum()),
        "cities_in_validation_not_in_train": sorted(
            (set(raw_val.pickup) | set(raw_val.delivery)) - (set(raw_train.pickup) | set(raw_train.delivery))),
    }
    pd.Series(quality).to_csv(REPORTS / "data_quality.csv", header=["value"])
    print("Data quality:", *[f"  {k}: {v}" for k, v in quality.items()], sep="\n")

    train = add_features(clean(raw_train))
    val = add_features(clean(raw_val))

    # ---- validation -------------------------------------------------------
    cv = time_cv(train)
    cv.to_csv(REPORTS / "cv_by_fold.csv", index=False)
    summary = cv.groupby("model")[["MAE", "RMSE", "MAPE_%", "MdAPE_%"]].mean().round(3)
    summary.to_csv(REPORTS / "cv_summary.csv")
    print("\nExpanding-window time CV (mean of 3 folds):\n", summary.to_string())
    cold = unseen_city_check(train)
    pd.Series(cold).round(3).to_csv(REPORTS / "unseen_city_check.csv", header=["value"])
    print("\nUnseen-city check (Sep-Oct, 8 cities held out):", {k: round(v, 3) for k, v in cold.items()})

    # ---- final fit on ALL labeled data, predict validation.csv -----------
    model = FreightRateModel().fit(train)
    out = template[["load_id"]].merge(
        pd.DataFrame({"load_id": val.load_id, "predicted_rate": model.predict(val)}), on="load_id", how="left")
    assert len(out) == 12_000 and out.predicted_rate.notna().all() and (out.predicted_rate > 0).all()
    out.round(2).to_csv("validation_predictions.csv", index=False)

    # ---- fixed December chart rows ---------------------------------------
    # Only 6 columns are given. Fill the rest WITHOUT touching the 5 fixed inputs:
    #   lat/lon      <- city lookup from the data
    #   market_index <- mean of validation.csv for that same date (date-level signal)
    #   quote_signal <- median of validation.csv December (row-level noise, no date signal)
    coords = city_coordinates(raw_train, raw_val)
    d = dec.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["pickup_lat"], d["pickup_lon"] = coords.loc[d.pickup, "lat"].values, coords.loc[d.pickup, "lon"].values
    d["delivery_lat"], d["delivery_lon"] = coords.loc[d.delivery, "lat"].values, coords.loc[d.delivery, "lon"].values
    mi_by_date = val.groupby("date")["market_index"].mean()
    d["market_index"] = d["date"].map(mi_by_date)
    d["quote_signal"] = val.loc[val.date >= "2025-12-01", "quote_signal"].median()
    d = clean(d.assign(weight=d.weight.astype(float)))
    d = add_features(d)
    assert d.market_index.notna().all()
    dec_out = dec.copy()
    dec_out["predicted_rate"] = model.predict(d).round(2)
    dec_out.to_csv("december_predictions.csv", index=False)

    print("\nDecember predictions:\n", dec_out[["date", "predicted_rate"]].to_string(index=False))
    print("\nWrote validation_predictions.csv and december_predictions.csv")


if __name__ == "__main__":
    main()
