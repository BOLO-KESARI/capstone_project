"""
SHAP Explainability for all ML models.

Provides:
  - SHAP value computation for tree-based models
  - Feature importance visualization data
  - Per-prediction explanations
  - Top demand/pricing drivers per route
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("ml_advanced.shap_explainer")


class SHAPExplainer:
    """SHAP wrapper for model explainability."""

    def __init__(self):
        self._explainers: dict[str, Any] = {}
        self._shap_values_cache: dict[str, Any] = {}
        self._feature_names: dict[str, list] = {}
        self._shap_available = False

        try:
            import shap
            self._shap_available = True
            logger.info("[SHAP] Library loaded successfully.")
        except ImportError:
            logger.warning("[SHAP] Not installed. Using fallback feature importance. Run: pip install shap")

    def register_model(self, name: str, model: Any, X_train: pd.DataFrame,
                       model_type: str = "tree") -> None:
        """
        Register a trained model for SHAP analysis.

        Parameters
        ----------
        name : model name (e.g., 'demand', 'pricing')
        model : trained sklearn/xgboost model
        X_train : training features
        model_type : 'tree' | 'linear' | 'kernel'
        """
        self._feature_names[name] = list(X_train.columns) if hasattr(X_train, 'columns') else [f"f{i}" for i in range(X_train.shape[1])]

        if not self._shap_available:
            self._explainers[name] = {"model": model, "type": model_type, "mode": "fallback"}
            return

        import shap

        # Sample for speed (max 500 rows)
        sample = X_train.sample(min(500, len(X_train)), random_state=42) if len(X_train) > 500 else X_train

        try:
            if model_type == "tree":
                explainer = shap.TreeExplainer(model)
            elif model_type == "linear":
                explainer = shap.LinearExplainer(model, sample)
            else:
                explainer = shap.KernelExplainer(model.predict, sample)

            self._explainers[name] = explainer
            # Pre-compute SHAP values for cached analysis
            self._shap_values_cache[name] = explainer.shap_values(sample)
            logger.info(f"[SHAP] Registered model '{name}' ({model_type})")
        except Exception as e:
            logger.warning(f"[SHAP] Failed to create explainer for {name}: {e}")
            self._explainers[name] = {"model": model, "type": model_type, "mode": "fallback"}

    def feature_importance(self, name: str) -> dict:
        """Get global feature importance for a model."""
        if name not in self._explainers:
            return {"error": f"Model '{name}' not registered"}

        features = self._feature_names.get(name, [])

        # SHAP-based importance
        if name in self._shap_values_cache:
            shap_vals = self._shap_values_cache[name]
            if isinstance(shap_vals, list):
                # Multi-class: average across classes
                mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
            else:
                mean_abs = np.abs(shap_vals).mean(axis=0)

            importance = sorted(
                zip(features, mean_abs.tolist()),
                key=lambda x: x[1], reverse=True,
            )
            return {
                "model": name,
                "method": "shap",
                "features": [{"name": f, "importance": round(v, 4)} for f, v in importance],
                "top_3_drivers": [f for f, _ in importance[:3]],
            }

        # Fallback: sklearn feature_importances_ or coef_
        explainer_info = self._explainers.get(name, {})
        model = explainer_info.get("model") if isinstance(explainer_info, dict) else None

        if model is not None:
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
            elif hasattr(model, "coef_"):
                imp = np.abs(model.coef_).flatten()
            else:
                imp = np.ones(len(features)) / len(features)

            importance = sorted(zip(features, imp.tolist()), key=lambda x: x[1], reverse=True)
            return {
                "model": name,
                "method": "sklearn_builtin",
                "features": [{"name": f, "importance": round(v, 4)} for f, v in importance],
                "top_3_drivers": [f for f, _ in importance[:3]],
            }

        return {"model": name, "method": "none", "features": []}

    def explain_prediction(self, name: str, input_data: pd.DataFrame) -> dict:
        """Explain a single prediction with SHAP values."""
        if name not in self._explainers:
            return {"error": f"Model '{name}' not registered"}

        features = self._feature_names.get(name, [])
        explainer = self._explainers[name]

        if self._shap_available and not isinstance(explainer, dict):
            try:
                import shap
                sv = explainer.shap_values(input_data)
                if isinstance(sv, list):
                    sv = sv[1]  # Positive class
                vals = sv[0] if len(sv.shape) > 1 else sv
                contributions = sorted(
                    zip(features, vals.tolist(), input_data.iloc[0].tolist()),
                    key=lambda x: abs(x[1]), reverse=True,
                )
                return {
                    "model": name,
                    "method": "shap",
                    "base_value": float(explainer.expected_value) if hasattr(explainer, 'expected_value') and not isinstance(explainer.expected_value, np.ndarray) else None,
                    "contributions": [
                        {"feature": f, "shap_value": round(s, 4), "feature_value": v}
                        for f, s, v in contributions
                    ],
                    "top_positive": [f for f, s, _ in contributions if s > 0][:3],
                    "top_negative": [f for f, s, _ in contributions if s < 0][:3],
                }
            except Exception as e:
                logger.warning(f"[SHAP] Prediction explanation failed: {e}")

        # Fallback
        fi = self.feature_importance(name)
        return {
            "model": name,
            "method": "feature_importance_proxy",
            "contributions": fi.get("features", [])[:10],
            "note": "SHAP library unavailable; showing global feature importance as proxy.",
        }

    def route_drivers(self, name: str, df: pd.DataFrame, route_col: str = "route") -> dict:
        """Get top demand/pricing drivers per route."""
        if name not in self._explainers or route_col not in df.columns:
            return {"error": "Model not registered or route column missing"}

        features = self._feature_names.get(name, [])
        routes = df[route_col].unique()
        result = {}

        for route in routes[:20]:  # Limit to 20 routes
            mask = df[route_col] == route
            subset = df.loc[mask].drop(columns=[route_col], errors="ignore")
            # Align columns
            for f in features:
                if f not in subset.columns:
                    subset[f] = 0
            subset = subset[features] if all(f in subset.columns for f in features) else subset

            try:
                fi = self.feature_importance(name)
                result[route] = fi.get("top_3_drivers", [])
            except Exception:
                result[route] = []

        return {"model": name, "route_drivers": result}

    def all_models_summary(self) -> dict:
        """Return explainability summary for all registered models."""
        return {
            name: self.feature_importance(name)
            for name in self._explainers
        }


# Singleton instance
shap_explainer = SHAPExplainer()
