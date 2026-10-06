# Freight Rate Prediction – Solution

Predicts `posted_rate` for 12,000 loads (Nov–Dec 2025) from 48,000 labeled loads (Jan–Oct 2025).

## Run

```bash
python -m pip install -r requirements.txt
# put the 4 CSVs in data/ : train_test.csv, validation.csv,
#   validation_predictions_template.csv, december_chart_inputs.csv
python train.py
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

`train.py` writes `validation_predictions.csv`, `december_predictions.csv` and `reports/*.csv`
(CV results, data-quality counts). `score.py` writes `scorer_results/candidate_december.png`.

## Approach

| Finding | Handling |
|---|---|
| `validation.csv` is a **later period** (Nov–Dec) than train (Jan–Oct) | Time-based validation only (expanding window, 3 folds × 2 months). No random split. |
| 292 / 145 **negative weights** (min = −max = 47,500) | Sign errors → `abs()` |
| 300 / 165 missing weight | Left as NaN + `weight_missing` flag (trees handle NaN) |
| 374 / 249 missing `market_index` | It is date-level (within-day std ≈ 0.025) → fill with same-date median |
| ~1.4% extreme rate-per-mile spikes (<$1 or >$4/mi) | Not deleted; robust **absolute-error loss** on log(rate/mile) |
| 8 cities in validation never seen in training (12% of rows) | No city-name categoricals; use lat/lon + distance. Checked with a held-out-city simulation |
| `market_index` shifts regime (train mean 1.08 → validation 0.93); trees on it overfit individual days | Kept **out of the trees**, applied as a Huber-linear residual adjustment |
| December chart rows lack `market_index`, `quote_signal`, lat/lon | lat/lon from city lookup; `market_index` = same-date mean from `validation.csv`; `quote_signal` = December median |

**Model:** `HistGradientBoostingRegressor` (absolute-error loss) on `log(rate / distance)` + linear `market_index` adjustment.
Prediction = `distance × exp(...)`.

## Results (mean of 3 expanding-window folds)

| Model | MAE | MAPE % | Median APE % |
|---|---|---|---|
| Median $/mile by equipment | 219.3 | 10.5 | 6.8 |
| HGB, market_index inside trees | 124.8 | 5.4 | 2.2 |
| HGB, no market_index | 113.8 | 4.9 | 1.7 |
| **HGB + linear market_index (final)** | **113.4** | **4.9** | **1.7** |

The final model is retrained on all of `train_test.csv` before predicting.
