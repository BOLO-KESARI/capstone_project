"""
Multi-Objective Optimization Engine.

Implements Pareto-optimal solutions balancing:
  - Revenue maximization
  - Load factor optimization (target 82-88%)
  - Market share preservation
  - Customer churn minimization
  - Competitive positioning

Uses weighted-sum and constraint-based approaches
(no external solver dependency).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("optimization.multi_objective")


class ObjectiveConfig:
    """Configuration for a single optimization objective."""

    def __init__(
        self,
        name: str,
        direction: str = "maximize",  # maximize | minimize
        weight: float = 1.0,
        target: float = None,
        min_bound: float = None,
        max_bound: float = None,
    ):
        self.name = name
        self.direction = direction
        self.weight = weight
        self.target = target
        self.min_bound = min_bound
        self.max_bound = max_bound

    def to_dict(self):
        return {
            "name": self.name,
            "direction": self.direction,
            "weight": self.weight,
            "target": self.target,
            "bounds": [self.min_bound, self.max_bound],
        }


# Default airline objectives
DEFAULT_OBJECTIVES = [
    ObjectiveConfig("revenue", "maximize", weight=0.40),
    ObjectiveConfig("load_factor", "maximize", weight=0.25, target=0.85, min_bound=0.60, max_bound=0.95),
    ObjectiveConfig("market_share", "maximize", weight=0.15, min_bound=0.10),
    ObjectiveConfig("churn_risk", "minimize", weight=0.10, max_bound=0.15),
    ObjectiveConfig("profit_margin", "maximize", weight=0.10, min_bound=0.05),
]


class MultiObjectiveOptimizer:
    """
    Multi-objective optimizer using grid search + Pareto filtering.

    Usage:
        optimizer = MultiObjectiveOptimizer()
        result = optimizer.optimize(
            price_range=(3000, 15000),
            route_data={"base_demand": 150, "capacity": 180, ...},
            models={"demand": demand_model, "churn": churn_model},
        )
    """

    def __init__(self, objectives: list[ObjectiveConfig] = None):
        self.objectives = objectives or DEFAULT_OBJECTIVES
        self._history: list[dict] = []

    def optimize(
        self,
        price_range: tuple[float, float],
        route_data: dict,
        models: dict[str, Any] = None,
        n_candidates: int = 200,
        fare_classes: list[str] = None,
    ) -> dict:
        """
        Find Pareto-optimal price points.

        Parameters
        ----------
        price_range : (min_price, max_price)
        route_data : dict with route characteristics (base_demand, capacity, competitor_price, etc.)
        models : dict of trained models keyed by objective name
        n_candidates : number of price candidates to evaluate
        fare_classes : optional fare class breakdown

        Returns
        -------
        dict with pareto_front, recommended_price, and analysis
        """
        min_price, max_price = price_range
        candidates = np.linspace(min_price, max_price, n_candidates)

        # Evaluate each candidate
        evaluations = []
        for price in candidates:
            scores = self._evaluate_candidate(price, route_data, models)
            scores["price"] = round(float(price), 2)
            evaluations.append(scores)

        # Find Pareto front
        pareto = self._pareto_filter(evaluations)

        # Weighted score ranking
        for e in evaluations:
            e["weighted_score"] = self._weighted_score(e)

        evaluations.sort(key=lambda x: x["weighted_score"], reverse=True)
        pareto.sort(key=lambda x: x["weighted_score"], reverse=True)

        # Recommended = highest weighted score on Pareto front
        recommended = pareto[0] if pareto else evaluations[0]

        # Constraint check
        constraint_violations = self._check_constraints(recommended)

        result = {
            "recommended_price": recommended["price"],
            "recommended_scores": recommended,
            "pareto_front": pareto[:20],
            "top_candidates": evaluations[:10],
            "constraint_violations": constraint_violations,
            "objectives": [o.to_dict() for o in self.objectives],
            "price_range": list(price_range),
            "n_evaluated": len(evaluations),
            "timestamp": datetime.now().isoformat(),
        }

        self._history.append(result)
        return result

    def _evaluate_candidate(
        self, price: float, route_data: dict, models: dict = None
    ) -> dict:
        """Evaluate a single price candidate across all objectives."""
        capacity = route_data.get("capacity", 180)
        base_demand = route_data.get("base_demand", 150)
        competitor_price = route_data.get("competitor_price", price)
        operating_cost = route_data.get("operating_cost_per_seat", 4000)
        market_size = route_data.get("market_size", 500)

        # --- Simple analytical models (fallback if ML models not provided) ---

        # Demand elasticity: higher price → lower demand
        price_ratio = price / (competitor_price + 1e-9)
        elasticity = route_data.get("price_elasticity", -1.2)
        demand = base_demand * (price_ratio ** elasticity)
        demand = max(0, min(demand, capacity * 1.1))  # cap at 110% capacity (overbooking)

        # Load factor
        load_factor = min(demand / capacity, 1.0) if capacity > 0 else 0

        # Revenue
        revenue = price * min(demand, capacity)

        # Market share
        market_share = min(demand / (market_size + 1e-9), 1.0)

        # Churn risk (higher price → higher churn)
        churn_risk = min(max(0.02 + 0.08 * (price_ratio - 0.8), 0), 0.5)

        # Profit margin
        total_cost = operating_cost * capacity
        profit = revenue - total_cost
        profit_margin = profit / (revenue + 1e-9)

        # Override with ML models if provided
        if models:
            if "demand" in models:
                try:
                    features = pd.DataFrame([{**route_data, "price": price, "fare_price": price}])
                    demand = float(models["demand"].predict(features.iloc[:, :models["demand"].n_features_in_])[0])
                except Exception:
                    pass
            if "churn" in models:
                try:
                    features = pd.DataFrame([{**route_data, "price": price}])
                    churn_risk = float(models["churn"].predict_proba(features.iloc[:, :models["churn"].n_features_in_])[:, 1][0])
                except Exception:
                    pass

        return {
            "revenue": round(float(revenue), 2),
            "load_factor": round(float(load_factor), 4),
            "market_share": round(float(market_share), 4),
            "churn_risk": round(float(churn_risk), 4),
            "profit_margin": round(float(profit_margin), 4),
            "estimated_demand": round(float(demand), 0),
            "estimated_profit": round(float(profit), 2),
        }

    def _weighted_score(self, evaluation: dict) -> float:
        """Calculate weighted score for an evaluation."""
        score = 0.0
        for obj in self.objectives:
            val = evaluation.get(obj.name, 0)
            if obj.direction == "minimize":
                val = -val
            score += obj.weight * val

            # Penalize constraint violations
            if obj.min_bound is not None and evaluation.get(obj.name, 0) < obj.min_bound:
                score -= 0.5 * obj.weight
            if obj.max_bound is not None and evaluation.get(obj.name, 0) > obj.max_bound:
                score -= 0.5 * obj.weight
        return round(score, 6)

    def _pareto_filter(self, evaluations: list[dict]) -> list[dict]:
        """Filter to Pareto-optimal solutions."""
        obj_names = [o.name for o in self.objectives]
        obj_dirs = [1 if o.direction == "maximize" else -1 for o in self.objectives]

        n = len(evaluations)
        is_dominated = [False] * n

        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue

                # Check if j dominates i
                all_better_or_equal = True
                strictly_better = False
                for k, name in enumerate(obj_names):
                    vi = evaluations[i].get(name, 0) * obj_dirs[k]
                    vj = evaluations[j].get(name, 0) * obj_dirs[k]
                    if vj < vi:
                        all_better_or_equal = False
                        break
                    if vj > vi:
                        strictly_better = True

                if all_better_or_equal and strictly_better:
                    is_dominated[i] = True
                    break

        return [e for i, e in enumerate(evaluations) if not is_dominated[i]]

    def _check_constraints(self, evaluation: dict) -> list[dict]:
        """Check constraint violations."""
        violations = []
        for obj in self.objectives:
            val = evaluation.get(obj.name, 0)
            if obj.min_bound is not None and val < obj.min_bound:
                violations.append({
                    "objective": obj.name,
                    "constraint": "min_bound",
                    "bound": obj.min_bound,
                    "actual": val,
                })
            if obj.max_bound is not None and val > obj.max_bound:
                violations.append({
                    "objective": obj.name,
                    "constraint": "max_bound",
                    "bound": obj.max_bound,
                    "actual": val,
                })
            if obj.target is not None and abs(val - obj.target) / (obj.target + 1e-9) > 0.1:
                violations.append({
                    "objective": obj.name,
                    "constraint": "target_deviation",
                    "target": obj.target,
                    "actual": val,
                    "deviation_pct": round(abs(val - obj.target) / (obj.target + 1e-9) * 100, 2),
                })
        return violations

    def fare_class_allocation(
        self,
        total_capacity: int,
        fare_classes: dict[str, dict],
    ) -> dict:
        """
        Optimal seat allocation across fare classes using EMSR-b heuristic.

        fare_classes: {
            "Y": {"price": 12000, "demand_mean": 50, "demand_std": 15},
            "M": {"price": 8000, "demand_mean": 80, "demand_std": 20},
            "Q": {"price": 5000, "demand_mean": 120, "demand_std": 30},
        }
        """
        from scipy.stats import norm  # Optional import

        classes = sorted(fare_classes.items(), key=lambda x: x[1]["price"], reverse=True)
        allocation = {}
        remaining = total_capacity

        for i, (cls, info) in enumerate(classes):
            price = info["price"]
            mean = info["demand_mean"]
            std = info["demand_std"]

            if i < len(classes) - 1:
                # EMSR-b: protect seats where marginal revenue > next class price
                next_price = classes[i + 1][1]["price"]
                protection_level = price / (price + 1e-9)

                try:
                    # Inverse normal: how many seats to protect
                    critical_ratio = next_price / price
                    protected = int(norm.ppf(1 - critical_ratio, loc=mean, scale=std))
                    protected = max(0, min(protected, remaining))
                except Exception:
                    protected = min(int(mean * 0.6), remaining)

                allocation[cls] = {
                    "seats": protected,
                    "price": price,
                    "expected_revenue": round(price * min(protected, mean), 2),
                    "protection_level": protected,
                }
                remaining -= protected
            else:
                # Lowest class gets remaining
                allocation[cls] = {
                    "seats": remaining,
                    "price": price,
                    "expected_revenue": round(price * min(remaining, mean), 2),
                    "protection_level": 0,
                }

        total_expected = sum(a["expected_revenue"] for a in allocation.values())
        return {
            "total_capacity": total_capacity,
            "allocation": allocation,
            "total_expected_revenue": round(total_expected, 2),
            "timestamp": datetime.now().isoformat(),
        }

    def get_history(self, limit: int = 20) -> list[dict]:
        return self._history[-limit:]


# Singleton
multi_objective_optimizer = MultiObjectiveOptimizer()
