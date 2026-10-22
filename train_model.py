from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = [
    "pickup", "delivery", "distance", "equipment", "weight",
    "month", "dayofweek", "dayofmonth", "dayofyear", "is_weekend",
]
CATEGORICAL = ["pickup", "delivery", "equipment"]
TARGET = "posted_rate"
CUTOFF = pd.Timestamp("2025-10-01")
SEED = 42


def add_features(frame: pd.DataFrame, median_weight: float | None = None):
    """Build the same feature set for training, validation and December inputs."""
    x = frame.copy()
    if "date" not in x.columns:
        raise ValueError("Required column 'date' is missing.")
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    if x["date"].isna().any():
        raise ValueError("One or more rows have invalid/missing dates.")

    for col in ["pickup", "delivery", "equipment", "distance", "weight"]:
        if col not in x.columns:
            raise ValueError(f"Required feature column {col!r} is missing.")

    x["month"] = x["date"].dt.month
    x["dayofweek"] = x["date"].dt.dayofweek
    x["dayofmonth"] = x["date"].dt.day
    x["dayofyear"] = x["date"].dt.dayofyear
    x["is_weekend"] = (x["dayofweek"] >= 5).astype(int)
    x["distance"] = pd.to_numeric(x["distance"], errors="coerce")
    x["weight"] = pd.to_numeric(x["weight"], errors="coerce")

    if x["distance"].isna().any():
        raise ValueError("Distance contains missing or non-numeric values; review data before training.")
    if median_weight is None:
        median_weight = float(x["weight"].median())
    if not np.isfinite(median_weight):
        raise ValueError("Could not calculate a finite training median for weight.")
    x["weight"] = x["weight"].fillna(median_weight)
    for col in CATEGORICAL:
        x[col] = x[col].fillna("Unknown").astype(str)
    return x[FEATURES], median_weight


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    nonzero = np.abs(y_true) > 1e-12
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_percent": mape,
        "R2": float(r2_score(y_true, y_pred)),
    }


def validate_inputs(train: pd.DataFrame, validation: pd.DataFrame, template: pd.DataFrame, december: pd.DataFrame):
    required_train = {"load_id", "date", TARGET, *FEATURES[:5]}
    missing = required_train - set(train.columns)
    if missing:
        raise ValueError(f"Training data missing required columns: {sorted(missing)}")
    if train[TARGET].isna().any() or (pd.to_numeric(train[TARGET], errors="coerce") <= 0).any():
        raise ValueError("Training target posted_rate has missing, non-numeric, or non-positive values.")
    if "load_id" not in validation or "load_id" not in template:
        raise ValueError("Validation data and prediction template must both contain load_id.")
    if validation["load_id"].duplicated().any() or template["load_id"].duplicated().any():
        raise ValueError("Duplicate load_id values found in validation data or template.")
    if set(validation["load_id"].astype(str)) != set(template["load_id"].astype(str)):
        raise ValueError("Validation load_id values do not exactly match the prediction template.")
    if len(december) != 31:
        raise ValueError(f"Expected 31 December chart rows, found {len(december)}.")


def main():
    parser = argparse.ArgumentParser(description="Train and validate the Spotter freight-rate model.")
    parser.add_argument("--data-dir", default="data", help="Directory containing the supplied CSV files")
    parser.add_argument("--output-dir", default="outputs", help="Output directory; defaults to outputs/ to preserve supplied files")
    args = parser.parse_args()
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "train_test.csv")
    validation = pd.read_csv(data_dir / "validation.csv")
    template = pd.read_csv(data_dir / "validation_predictions_template.csv")
    december = pd.read_csv(data_dir / "december_chart_inputs.csv")
    train["date"] = pd.to_datetime(train["date"], errors="coerce")
    if train["date"].isna().any():
        raise ValueError("Training data contains invalid/missing dates.")
    validate_inputs(train, validation, template, december)

    # train before October - validate on October only
    train_mask = train["date"] < CUTOFF
    holdout_mask = (train["date"] >= CUTOFF) & (train["date"] < pd.Timestamp("2025-11-01"))
    fit_part, holdout = train.loc[train_mask].copy(), train.loc[holdout_mask].copy()
    if fit_part.empty or holdout.empty:
        raise ValueError("Chronological split produced an empty training or October holdout set.")
    if (train.loc[~(train_mask | holdout_mask), "date"] < pd.Timestamp("2025-11-01")).any():
        raise ValueError("Unexpected dates found outside the intended training/October holdout periods.")

    x_fit, median_weight = add_features(fit_part)
    x_holdout, _ = add_features(holdout, median_weight)
    y_fit = pd.to_numeric(fit_part[TARGET], errors="raise").to_numpy(dtype=float)
    y_holdout = pd.to_numeric(holdout[TARGET], errors="raise").to_numpy(dtype=float)

    model = CatBoostRegressor(
        iterations=1200, depth=8, learning_rate=0.05,
        loss_function="RMSE", eval_metric="RMSE", random_seed=SEED,
        l2_leaf_reg=5, verbose=100, allow_writing_files=False,
    )
    model.fit(
        x_fit, np.log1p(y_fit), cat_features=CATEGORICAL,
        eval_set=(x_holdout, np.log1p(y_holdout)),
        early_stopping_rounds=100, use_best_model=True,
    )
    best_iterations = int(model.best_iteration_ + 1)
    holdout_pred = np.maximum(np.expm1(model.predict(x_holdout)), 1.0)
    metrics = calculate_metrics(y_holdout, holdout_pred)
    metrics["selected_iterations"] = best_iterations
    metrics["training_rows"] = int(len(fit_part))
    metrics["holdout_rows"] = int(len(holdout))
    metrics["holdout_start"] = str(holdout["date"].min().date())
    metrics["holdout_end"] = str(holdout["date"].max().date())
    (output_dir / "holdout_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"load_id": holdout["load_id"].to_numpy(), "actual_rate": y_holdout,
                  "predicted_rate": holdout_pred}).to_csv(output_dir / "october_holdout_predictions.csv", index=False)

    # Refit on all labeled development data 
    x_full, full_median_weight = add_features(train)
    final = CatBoostRegressor(
        iterations=best_iterations, depth=8, learning_rate=0.05,
        loss_function="RMSE", eval_metric="RMSE", random_seed=SEED,
        l2_leaf_reg=5, verbose=100, allow_writing_files=False,
    )
    final.fit(x_full, np.log1p(pd.to_numeric(train[TARGET], errors="raise").to_numpy(dtype=float)),
              cat_features=CATEGORICAL)
    final.save_model(str(output_dir / "model.cbm"))
    (output_dir / "model_metadata.json").write_text(json.dumps({
        "model": "CatBoostRegressor", "features": FEATURES, "categorical_features": CATEGORICAL,
        "target_transform": "log1p; predictions inverse-transformed with expm1",
        "seed": SEED, "iterations": best_iterations, "weight_median_full_training": full_median_weight,
    }, indent=2), encoding="utf-8")

    # Predict validation rows and preserve the exact template order
    x_validation, _ = add_features(validation, full_median_weight)
    validation_pred = np.maximum(np.expm1(final.predict(x_validation)), 1.0)
    prediction_map = dict(zip(validation["load_id"].astype(str), validation_pred))
    final_template = template.copy()
    final_template["predicted_rate"] = final_template["load_id"].astype(str).map(prediction_map)
    if final_template["predicted_rate"].isna().any():
        raise ValueError("Some template IDs did not receive predictions.")
    final_template = final_template[["load_id", "predicted_rate"]]
    final_template.to_csv(
    output_dir / "validation_predictions.csv",
    index=False,
    float_format="%.2f",
)

    # December source is treated 
    x_december, _ = add_features(december, full_median_weight)
    december_pred = np.maximum(np.expm1(final.predict(x_december)), 1.0)
    december_out = december.copy()
    december_out["predicted_rate"] = december_pred
    december_out.to_csv(output_dir / "december_predictions.csv", index=False, float_format="%.2f")

    print("\n=== October chronological holdout metrics (computed in this run) ===")
    for key in ["MAE", "RMSE", "MAPE_percent", "R2"]:
        print(f"{key}: {metrics[key]:.6f}")
    print(f"Selected iterations: {best_iterations}")
    print(f"Rows: train={len(fit_part):,}; holdout={len(holdout):,}; final training={len(train):,}")
    print(f"Saved outputs to: {output_dir.resolve()}")
    print("Next: run score.py against outputs/validation_predictions.csv and outputs/december_predictions.csv")


if __name__ == "__main__":
    main()
