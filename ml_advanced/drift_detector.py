"""
Model Drift Detection & Performance Monitoring.

Implements:
  - Data drift detection (PSI, KS-test)
  - Concept drift via performance degradation
  - Feature distribution monitoring
  - Automated retraining triggers
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("ml_advanced.drift_detector")


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, bins: int = 10
) -> float:
    """
    Population Stability Index (PSI).
    PSI < 0.1 → no drift
    0.1 ≤ PSI < 0.25 → moderate drift
    PSI ≥ 0.25 → significant drift
    """
    eps = 1e-6
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()) + eps,
        bins + 1,
    )
    expected_pcts = np.histogram(expected, bins=breakpoints)[0] / len(expected) + eps
    actual_pcts = np.histogram(actual, bins=breakpoints)[0] / len(actual) + eps
    psi = float(np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts)))
    return round(psi, 6)


def ks_test(expected: np.ndarray, actual: np.ndarray) -> dict:
    """Kolmogorov-Smirnov test for distribution difference."""
    try:
        from scipy.stats import ks_2samp
        stat, p_value = ks_2samp(expected, actual)
        return {"statistic": round(float(stat), 4), "p_value": round(float(p_value), 6), "significant": p_value < 0.05}
    except ImportError:
        # Fallback: manual KS
        n1, n2 = len(expected), len(actual)
        combined = np.sort(np.concatenate([expected, actual]))
        cdf1 = np.searchsorted(np.sort(expected), combined, side="right") / n1
        cdf2 = np.searchsorted(np.sort(actual), combined, side="right") / n2
        stat = float(np.max(np.abs(cdf1 - cdf2)))
        return {"statistic": round(stat, 4), "p_value": None, "significant": stat > 0.05}


class DriftDetector:
    """
    Monitors drift for registered models.

    Usage:
        detector = DriftDetector()
        detector.register_baseline("pricing", X_train, y_train, model, baseline_mape=5.2)
        report = detector.check_drift("pricing", X_new, y_new_optional)
    """

    def __init__(self):
        self._baselines: dict[str, dict] = {}
        self._history: list[dict] = []

    def register_baseline(
        self,
        model_name: str,
        X_train: pd.DataFrame | np.ndarray,
        y_train: np.ndarray | pd.Series = None,
        model: Any = None,
        baseline_metrics: dict = None,
    ):
        """Register training data distribution as baseline."""
        if isinstance(X_train, pd.DataFrame):
            feature_stats = {}
            for col in X_train.columns:
                vals = X_train[col].dropna().values.astype(float)
                feature_stats[col] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "values": vals,
                }
        else:
            arr = np.array(X_train)
            feature_stats = {
                f"feature_{i}": {
                    "mean": float(np.mean(arr[:, i])),
                    "std": float(np.std(arr[:, i])),
                    "values": arr[:, i],
                }
                for i in range(arr.shape[1])
            }

        self._baselines[model_name] = {
            "feature_stats": feature_stats,
            "y_values": np.array(y_train) if y_train is not None else None,
            "model": model,
            "baseline_metrics": baseline_metrics or {},
            "registered_at": datetime.now().isoformat(),
            "n_samples": len(X_train),
        }
        logger.info(f"[DriftDetector] Baseline registered for '{model_name}' ({len(X_train)} samples)")

    def check_drift(
        self,
        model_name: str,
        X_new: pd.DataFrame | np.ndarray,
        y_new: np.ndarray | pd.Series = None,
    ) -> dict:
        """Check for data and concept drift."""
        if model_name not in self._baselines:
            return {"error": f"No baseline registered for '{model_name}'"}

        baseline = self._baselines[model_name]
        report = {
            "model_name": model_name,
            "checked_at": datetime.now().isoformat(),
            "n_baseline_samples": baseline["n_samples"],
            "n_new_samples": len(X_new),
            "feature_drift": {},
            "overall_data_drift": "none",
            "concept_drift": None,
            "retraining_recommended": False,
            "alerts": [],
        }

        # --- Data Drift: per-feature PSI + KS ---
        if isinstance(X_new, pd.DataFrame):
            cols = X_new.columns
        else:
            arr = np.array(X_new)
            cols = [f"feature_{i}" for i in range(arr.shape[1])]

        psi_values = []
        drifted_features = []

        for col in cols:
            if col not in baseline["feature_stats"]:
                continue

            baseline_vals = baseline["feature_stats"][col]["values"]
            if isinstance(X_new, pd.DataFrame):
                new_vals = X_new[col].dropna().values.astype(float)
            else:
                idx = int(col.split("_")[1])
                new_vals = arr[:, idx].astype(float)

            if len(new_vals) < 5:
                continue

            psi = population_stability_index(baseline_vals, new_vals)
            ks = ks_test(baseline_vals, new_vals)

            drift_level = "none"
            if psi >= 0.25:
                drift_level = "significant"
            elif psi >= 0.1:
                drift_level = "moderate"

            report["feature_drift"][col] = {
                "psi": psi,
                "ks_statistic": ks["statistic"],
                "ks_significant": ks["significant"],
                "drift_level": drift_level,
                "baseline_mean": round(float(np.mean(baseline_vals)), 4),
                "new_mean": round(float(np.mean(new_vals)), 4),
                "mean_shift_pct": round(
                    abs(np.mean(new_vals) - np.mean(baseline_vals))
                    / (abs(np.mean(baseline_vals)) + 1e-9)
                    * 100,
                    2,
                ),
            }

            psi_values.append(psi)
            if drift_level in ("moderate", "significant"):
                drifted_features.append(col)

        # Overall data drift
        if psi_values:
            avg_psi = np.mean(psi_values)
            if avg_psi >= 0.25 or len(drifted_features) > len(cols) * 0.3:
                report["overall_data_drift"] = "significant"
                report["alerts"].append("Significant data drift detected across multiple features")
                report["retraining_recommended"] = True
            elif avg_psi >= 0.1 or len(drifted_features) > 0:
                report["overall_data_drift"] = "moderate"
                report["alerts"].append(f"Moderate drift in {len(drifted_features)} feature(s)")
            report["avg_psi"] = round(float(avg_psi), 6)

        # --- Concept Drift: performance degradation ---
        model = baseline.get("model")
        if model and y_new is not None:
            try:
                y_pred = model.predict(X_new if isinstance(X_new, pd.DataFrame) else arr)
                y_actual = np.array(y_new)

                mask = y_actual != 0
                current_mape = float(
                    np.mean(np.abs((y_actual[mask] - y_pred[mask]) / y_actual[mask])) * 100
                ) if mask.any() else 0

                baseline_mape = baseline["baseline_metrics"].get("mape")
                if baseline_mape and baseline_mape > 0:
                    mape_increase = current_mape - baseline_mape
                    mape_increase_pct = mape_increase / baseline_mape * 100

                    report["concept_drift"] = {
                        "baseline_mape": round(baseline_mape, 2),
                        "current_mape": round(current_mape, 2),
                        "mape_increase": round(mape_increase, 2),
                        "mape_increase_pct": round(mape_increase_pct, 2),
                        "degraded": mape_increase_pct > 20,
                    }

                    if mape_increase_pct > 50:
                        report["alerts"].append(f"CRITICAL: MAPE increased {mape_increase_pct:.0f}% — retraining required")
                        report["retraining_recommended"] = True
                    elif mape_increase_pct > 20:
                        report["alerts"].append(f"WARNING: MAPE increased {mape_increase_pct:.0f}%")
                        report["retraining_recommended"] = True
            except Exception as e:
                report["concept_drift"] = {"error": str(e)}

        self._history.append(report)
        if len(self._history) > 500:
            self._history = self._history[-300:]

        return report

    def get_history(self, model_name: str = None, limit: int = 50) -> list[dict]:
        """Return drift check history."""
        history = self._history
        if model_name:
            history = [h for h in history if h["model_name"] == model_name]
        return history[-limit:]

    def check_all_models_drift(self, new_data: dict[str, tuple]) -> dict:
        """
        Check drift for all registered models.

        new_data: {model_name: (X_new, y_new_or_None)}
        """
        results = {}
        for name in self._baselines:
            if name in new_data:
                X_new, y_new = new_data[name]
                results[name] = self.check_drift(name, X_new, y_new)
        return {
            "checked_at": datetime.now().isoformat(),
            "models_checked": len(results),
            "models_needing_retrain": sum(
                1 for r in results.values() if r.get("retraining_recommended")
            ),
            "details": results,
        }


# Singleton
drift_detector = DriftDetector()
