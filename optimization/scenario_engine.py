"""
Scenario Simulation Engine.

Implements what-if analysis for airline revenue management:
  - Fuel price shocks (+10%, +20%, +50%)
  - Demand fluctuations (-30%, -15%, +20% surge)
  - Competitor price changes (-15%, -10%, +10%)
  - Event-driven surges (festivals, cricket, holidays)
  - Combined multi-factor scenarios
  - Monte Carlo simulation for uncertainty quantification
"""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("optimization.scenario_engine")


# ─── Pre-defined scenario templates ────────────────────────────────────────

SCENARIO_TEMPLATES = {
    "fuel_spike_20": {
        "name": "Fuel Price +20%",
        "description": "ATF price increases 20% due to crude oil surge",
        "adjustments": {"fuel_price": 1.20, "operating_cost": 1.08},
    },
    "fuel_spike_50": {
        "name": "Fuel Price +50%",
        "description": "Major geopolitical crisis drives 50% fuel increase",
        "adjustments": {"fuel_price": 1.50, "operating_cost": 1.20},
    },
    "demand_drop_30": {
        "name": "Demand Crash -30%",
        "description": "Pandemic-like demand collapse across all routes",
        "adjustments": {"demand_factor": 0.70, "load_factor": 0.70, "booking_velocity": 0.65},
    },
    "demand_surge_20": {
        "name": "Demand Surge +20%",
        "description": "Festival season / major event driving demand spike",
        "adjustments": {"demand_factor": 1.20, "load_factor": 1.15, "booking_velocity": 1.25},
    },
    "competitor_undercut_15": {
        "name": "Competitor Undercuts -15%",
        "description": "Major competitor launches aggressive pricing",
        "adjustments": {"competitor_avg_price": 0.85, "market_share": 0.92},
    },
    "competitor_exit": {
        "name": "Competitor Route Exit",
        "description": "Competitor withdraws from route, reducing competition",
        "adjustments": {"competitor_avg_price": 1.20, "market_share": 1.15, "demand_factor": 1.10},
    },
    "monsoon_disruption": {
        "name": "Monsoon Disruption",
        "description": "Heavy monsoon season causing cancellations and delays",
        "adjustments": {"cancellation_rate": 1.80, "delay_factor": 2.0, "operating_cost": 1.05, "demand_factor": 0.90},
    },
    "diwali_peak": {
        "name": "Diwali Peak Season",
        "description": "Diwali festival period with extreme demand",
        "adjustments": {"demand_factor": 1.50, "booking_velocity": 1.80, "load_factor": 1.20, "willingness_to_pay": 1.30},
    },
    "new_route_launch": {
        "name": "New Route Launch",
        "description": "Launching a new unproven route with limited data",
        "adjustments": {"demand_factor": 0.40, "load_factor": 0.50, "marketing_cost": 2.0, "competitor_avg_price": 1.0},
    },
    "recession": {
        "name": "Economic Recession",
        "description": "GDP contraction, reduced business travel",
        "adjustments": {"demand_factor": 0.80, "business_pax_share": 0.65, "willingness_to_pay": 0.85, "load_factor": 0.85},
    },
}


class ScenarioEngine:
    """
    Run what-if scenarios against trained ML models and pricing data.
    """

    def __init__(self):
        self._models: dict[str, Any] = {}
        self._baseline_data: pd.DataFrame | None = None
        self._simulation_history: list[dict] = []

    def register_model(self, name: str, model: Any, feature_cols: list[str]):
        """Register a model for scenario analysis."""
        self._models[name] = {"model": model, "features": feature_cols}

    def set_baseline(self, df: pd.DataFrame):
        """Set baseline dataset for scenario comparisons."""
        self._baseline_data = df.copy()

    def list_templates(self) -> dict:
        """Return all available scenario templates."""
        return {
            k: {"name": v["name"], "description": v["description"], "adjustments": v["adjustments"]}
            for k, v in SCENARIO_TEMPLATES.items()
        }

    def run_scenario(
        self,
        scenario_id: str = None,
        custom_adjustments: dict = None,
        model_name: str = None,
        data: pd.DataFrame = None,
    ) -> dict:
        """
        Run a single scenario.

        Parameters
        ----------
        scenario_id : key from SCENARIO_TEMPLATES (or None for custom)
        custom_adjustments : {column_name: multiplier_or_value}
        model_name : which registered model to use for predictions
        data : override data (uses baseline if None)
        """
        if scenario_id and scenario_id in SCENARIO_TEMPLATES:
            template = SCENARIO_TEMPLATES[scenario_id]
            adjustments = template["adjustments"]
            scenario_name = template["name"]
        elif custom_adjustments:
            adjustments = custom_adjustments
            scenario_name = "Custom Scenario"
        else:
            return {"error": "Provide scenario_id or custom_adjustments"}

        df = data if data is not None else self._baseline_data
        if df is None:
            return {"error": "No data available. Call set_baseline() or pass data."}

        df_scenario = df.copy()

        # Apply adjustments
        applied = []
        for col, multiplier in adjustments.items():
            if col in df_scenario.columns:
                original_mean = float(df_scenario[col].mean())
                df_scenario[col] = df_scenario[col] * multiplier
                new_mean = float(df_scenario[col].mean())
                applied.append({
                    "column": col,
                    "multiplier": multiplier,
                    "original_mean": round(original_mean, 2),
                    "adjusted_mean": round(new_mean, 2),
                })

        result = {
            "scenario_name": scenario_name,
            "scenario_id": scenario_id,
            "adjustments_applied": applied,
            "adjustments_skipped": [c for c in adjustments if c not in df_scenario.columns],
            "n_rows": len(df_scenario),
            "timestamp": datetime.now().isoformat(),
        }

        # Run model predictions on modified data
        if model_name and model_name in self._models:
            model_info = self._models[model_name]
            model = model_info["model"]
            features = model_info["features"]

            available = [f for f in features if f in df_scenario.columns]
            if len(available) >= len(features) * 0.5:
                try:
                    # Baseline prediction
                    X_base = df[available] if data is None else data[available]
                    y_base = model.predict(X_base)

                    # Scenario prediction
                    X_scenario = df_scenario[available]
                    y_scenario = model.predict(X_scenario)

                    delta = y_scenario - y_base
                    result["predictions"] = {
                        "baseline_mean": round(float(np.mean(y_base)), 2),
                        "scenario_mean": round(float(np.mean(y_scenario)), 2),
                        "absolute_change": round(float(np.mean(delta)), 2),
                        "pct_change": round(float(np.mean(delta) / (np.mean(y_base) + 1e-9) * 100), 2),
                        "baseline_total": round(float(np.sum(y_base)), 2),
                        "scenario_total": round(float(np.sum(y_scenario)), 2),
                        "revenue_impact": round(float(np.sum(delta)), 2),
                    }
                except Exception as e:
                    result["predictions"] = {"error": str(e)}

        self._simulation_history.append(result)
        if len(self._simulation_history) > 200:
            self._simulation_history = self._simulation_history[-100:]

        return result

    def run_combined_scenario(
        self,
        scenario_ids: list[str],
        model_name: str = None,
        data: pd.DataFrame = None,
    ) -> dict:
        """Run multiple scenarios stacked on top of each other."""
        merged_adjustments = {}
        names = []
        for sid in scenario_ids:
            if sid in SCENARIO_TEMPLATES:
                template = SCENARIO_TEMPLATES[sid]
                names.append(template["name"])
                for col, mult in template["adjustments"].items():
                    if col in merged_adjustments:
                        merged_adjustments[col] *= mult  # compound
                    else:
                        merged_adjustments[col] = mult

        result = self.run_scenario(
            custom_adjustments=merged_adjustments,
            model_name=model_name,
            data=data,
        )
        result["scenario_name"] = " + ".join(names)
        result["combined_from"] = scenario_ids
        return result

    def monte_carlo(
        self,
        base_adjustments: dict,
        n_simulations: int = 1000,
        noise_pct: float = 10.0,
        model_name: str = None,
        data: pd.DataFrame = None,
    ) -> dict:
        """
        Monte Carlo simulation with random noise around base adjustments.

        Each simulation randomly perturbs each adjustment by ±noise_pct%.
        Returns distribution statistics for the predicted outcomes.
        """
        df = data if data is not None else self._baseline_data
        if df is None:
            return {"error": "No data available"}

        if not model_name or model_name not in self._models:
            return {"error": "Model required for Monte Carlo"}

        model_info = self._models[model_name]
        model = model_info["model"]
        features = model_info["features"]
        available = [f for f in features if f in df.columns]

        results = []
        for i in range(n_simulations):
            df_sim = df.copy()

            # Add random noise to adjustments
            noisy = {}
            for col, mult in base_adjustments.items():
                noise = 1 + np.random.uniform(-noise_pct / 100, noise_pct / 100)
                noisy[col] = mult * noise
                if col in df_sim.columns:
                    df_sim[col] = df_sim[col] * noisy[col]

            try:
                X = df_sim[available]
                y_pred = model.predict(X)
                results.append(float(np.mean(y_pred)))
            except Exception:
                pass

        if not results:
            return {"error": "All simulations failed"}

        arr = np.array(results)
        return {
            "scenario": "Monte Carlo Simulation",
            "n_simulations": len(results),
            "noise_pct": noise_pct,
            "base_adjustments": base_adjustments,
            "statistics": {
                "mean": round(float(np.mean(arr)), 2),
                "median": round(float(np.median(arr)), 2),
                "std": round(float(np.std(arr)), 2),
                "p5": round(float(np.percentile(arr, 5)), 2),
                "p25": round(float(np.percentile(arr, 25)), 2),
                "p75": round(float(np.percentile(arr, 75)), 2),
                "p95": round(float(np.percentile(arr, 95)), 2),
                "min": round(float(np.min(arr)), 2),
                "max": round(float(np.max(arr)), 2),
            },
            "var_at_risk_5pct": round(float(np.percentile(arr, 5)), 2),
            "timestamp": datetime.now().isoformat(),
        }

    def sensitivity_analysis(
        self,
        parameter: str,
        range_values: list[float],
        model_name: str,
        data: pd.DataFrame = None,
    ) -> dict:
        """
        Vary one parameter across a range and measure impact.
        range_values: list of multipliers, e.g. [0.8, 0.9, 1.0, 1.1, 1.2]
        """
        points = []
        for mult in range_values:
            r = self.run_scenario(
                custom_adjustments={parameter: mult},
                model_name=model_name,
                data=data,
            )
            pred = r.get("predictions", {})
            points.append({
                "multiplier": mult,
                "predicted_mean": pred.get("scenario_mean"),
                "pct_change": pred.get("pct_change"),
            })

        return {
            "parameter": parameter,
            "points": points,
            "elasticity": self._estimate_elasticity(points),
            "timestamp": datetime.now().isoformat(),
        }

    @staticmethod
    def _estimate_elasticity(points: list[dict]) -> float | None:
        """Estimate price elasticity from sensitivity points."""
        valid = [p for p in points if p["multiplier"] is not None and p["pct_change"] is not None]
        if len(valid) < 2:
            return None
        # elasticity ≈ %change_outcome / %change_input
        x = [(p["multiplier"] - 1) * 100 for p in valid]
        y = [p["pct_change"] for p in valid]
        if max(x) == min(x):
            return None
        slope = np.polyfit(x, y, 1)[0]
        return round(float(slope), 4)

    def get_history(self, limit: int = 50) -> list[dict]:
        return self._simulation_history[-limit:]


# Singleton
scenario_engine = ScenarioEngine()
