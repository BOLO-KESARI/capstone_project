"""
Walk-Forward Validation for time-series ML models.

Implements:
  - Expanding window walk-forward
  - Sliding window walk-forward
  - Per-fold MAPE, RMSE, MAE tracking
  - Forecast bias detection
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger("ml_advanced.validation")


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE metric – avoids division by zero."""
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def walk_forward_validation(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_factory: Callable,
    n_splits: int = 5,
    min_train_size: int = 100,
    strategy: str = "expanding",
    date_col: str = None,
) -> dict:
    """
    Walk-forward cross-validation.

    Parameters
    ----------
    df : full dataset (sorted by time if date_col provided)
    feature_cols : feature column names
    target_col : target column name
    model_factory : callable that returns an untrained model
    n_splits : number of forward-walk folds
    min_train_size : minimum training set size
    strategy : 'expanding' or 'sliding'
    date_col : date column for time-ordered split

    Returns
    -------
    dict with per-fold metrics, aggregate metrics, and bias analysis
    """
    if date_col and date_col in df.columns:
        df = df.sort_values(date_col).reset_index(drop=True)

    n = len(df)
    fold_size = (n - min_train_size) // n_splits

    if fold_size < 10:
        return {"error": "Insufficient data for walk-forward validation",
                "n_rows": n, "min_needed": min_train_size + 10 * n_splits}

    folds = []
    all_y_true = []
    all_y_pred = []

    for i in range(n_splits):
        if strategy == "expanding":
            train_end = min_train_size + i * fold_size
            train_start = 0
        else:  # sliding
            train_end = min_train_size + i * fold_size
            train_start = max(0, train_end - min_train_size)

        test_start = train_end
        test_end = min(test_start + fold_size, n)

        if test_end <= test_start:
            break

        X_train = df[feature_cols].iloc[train_start:train_end]
        y_train = df[target_col].iloc[train_start:train_end]
        X_test = df[feature_cols].iloc[test_start:test_end]
        y_test = df[target_col].iloc[test_start:test_end]

        # Train and predict
        model = model_factory()
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        except Exception as e:
            logger.warning(f"[WFV] Fold {i + 1} failed: {e}")
            continue

        y_true = y_test.values
        mape = mean_absolute_percentage_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))

        # Bias: positive = over-predicting, negative = under-predicting
        bias = float(np.mean(y_pred - y_true))
        bias_pct = float(bias / np.mean(y_true) * 100) if np.mean(y_true) != 0 else 0

        folds.append({
            "fold": i + 1,
            "train_size": train_end - train_start,
            "test_size": test_end - test_start,
            "mape": round(mape, 2),
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "bias": round(bias, 2),
            "bias_pct": round(bias_pct, 2),
        })

        all_y_true.extend(y_true.tolist())
        all_y_pred.extend(y_pred.tolist())

    if not folds:
        return {"error": "All folds failed"}

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # Aggregate
    avg_mape = round(float(np.mean([f["mape"] for f in folds])), 2)
    avg_rmse = round(float(np.mean([f["rmse"] for f in folds])), 2)
    overall_bias = round(float(np.mean(all_y_pred - all_y_true)), 2)
    overall_bias_pct = round(float(overall_bias / np.mean(all_y_true) * 100), 2) if np.mean(all_y_true) != 0 else 0

    # Bias detection
    bias_direction = "neutral"
    if overall_bias_pct > 5:
        bias_direction = "over-predicting"
    elif overall_bias_pct < -5:
        bias_direction = "under-predicting"

    # Trend: is MAPE getting worse over folds?
    mape_values = [f["mape"] for f in folds]
    degrading = len(mape_values) >= 3 and all(mape_values[i] < mape_values[i + 1] for i in range(len(mape_values) - 2))

    return {
        "strategy": strategy,
        "n_splits": len(folds),
        "folds": folds,
        "aggregate": {
            "avg_mape": avg_mape,
            "avg_rmse": avg_rmse,
            "avg_mae": round(float(np.mean([f["mae"] for f in folds])), 2),
            "overall_bias": overall_bias,
            "overall_bias_pct": overall_bias_pct,
            "bias_direction": bias_direction,
        },
        "alerts": {
            "high_mape": avg_mape > 15,
            "significant_bias": abs(overall_bias_pct) > 10,
            "degrading_performance": degrading,
        },
        "validated_at": datetime.now().isoformat(),
    }


def mape_by_route(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    route_col: str,
    model: Any,
) -> dict:
    """Compute MAPE per route for an already-trained model."""
    routes = df[route_col].unique()
    result = {}

    for route in routes:
        subset = df[df[route_col] == route]
        if len(subset) < 10:
            continue
        X = subset[feature_cols]
        y = subset[target_col].values
        try:
            y_pred = model.predict(X)
            mape = mean_absolute_percentage_error(y, y_pred)
            result[route] = {
                "mape": round(mape, 2),
                "n_samples": len(subset),
                "avg_actual": round(float(np.mean(y)), 2),
                "avg_predicted": round(float(np.mean(y_pred)), 2),
            }
        except Exception:
            pass

    return {"per_route_mape": result, "avg_mape": round(float(np.mean([v["mape"] for v in result.values()])), 2) if result else 0}
