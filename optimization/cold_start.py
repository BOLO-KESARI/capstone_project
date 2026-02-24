"""
Cold Start Route Strategy.

Handles pricing and demand estimation for new routes with no historical data:
  - Cluster-based similarity (find similar existing routes)
  - Bayesian prior estimation
  - Progressive learning as data accumulates
  - Conservative pricing recommendations
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("optimization.cold_start")


# Indian city metadata for similarity computation
CITY_METADATA = {
    "DEL": {"tier": 1, "region": "north", "lat": 28.56, "lon": 77.10, "traffic_rank": 1},
    "BOM": {"tier": 1, "region": "west", "lat": 19.09, "lon": 72.87, "traffic_rank": 2},
    "BLR": {"tier": 1, "region": "south", "lat": 12.97, "lon": 77.59, "traffic_rank": 3},
    "MAA": {"tier": 1, "region": "south", "lat": 13.08, "lon": 80.27, "traffic_rank": 5},
    "CCU": {"tier": 1, "region": "east", "lat": 22.57, "lon": 88.36, "traffic_rank": 6},
    "HYD": {"tier": 1, "region": "south", "lat": 17.39, "lon": 78.49, "traffic_rank": 4},
    "GOI": {"tier": 2, "region": "west", "lat": 15.38, "lon": 73.83, "traffic_rank": 8},
    "COK": {"tier": 2, "region": "south", "lat": 10.15, "lon": 76.40, "traffic_rank": 9},
    "PNQ": {"tier": 2, "region": "west", "lat": 18.58, "lon": 73.92, "traffic_rank": 10},
    "AMD": {"tier": 2, "region": "west", "lat": 23.07, "lon": 72.63, "traffic_rank": 11},
    "JAI": {"tier": 2, "region": "north", "lat": 26.82, "lon": 75.81, "traffic_rank": 12},
    "LKO": {"tier": 2, "region": "north", "lat": 26.76, "lon": 80.89, "traffic_rank": 13},
    "PAT": {"tier": 3, "region": "east", "lat": 25.59, "lon": 85.09, "traffic_rank": 18},
    "IXC": {"tier": 2, "region": "north", "lat": 30.67, "lon": 76.79, "traffic_rank": 15},
    "GAU": {"tier": 3, "region": "east", "lat": 26.11, "lon": 91.59, "traffic_rank": 20},
    "BBI": {"tier": 3, "region": "east", "lat": 20.24, "lon": 85.82, "traffic_rank": 17},
    "IXR": {"tier": 3, "region": "east", "lat": 23.31, "lon": 85.32, "traffic_rank": 22},
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km."""
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


class ColdStartStrategy:
    """
    Cold start pricing for new routes.

    Usage:
        cs = ColdStartStrategy()
        cs.load_existing_routes(routes_df)
        recommendation = cs.recommend("IXR", "GOI")
    """

    def __init__(self):
        self._existing_routes: pd.DataFrame | None = None
        self._route_profiles: dict[str, dict] = {}

    def load_existing_routes(self, df: pd.DataFrame):
        """
        Load existing route performance data.

        Expected columns: origin, destination, avg_price, avg_demand,
                         avg_load_factor, distance_km, route_type
        """
        self._existing_routes = df.copy()

        # Build route profiles
        for _, row in df.iterrows():
            key = f"{row.get('origin', '')}-{row.get('destination', '')}"
            self._route_profiles[key] = {
                "avg_price": row.get("avg_price", 0),
                "avg_demand": row.get("avg_demand", 0),
                "avg_load_factor": row.get("avg_load_factor", 0),
                "distance_km": row.get("distance_km", 0),
            }

    def find_similar_routes(
        self, origin: str, destination: str, top_k: int = 5
    ) -> list[dict]:
        """Find the most similar existing routes based on multi-dimensional similarity."""
        if self._existing_routes is None or self._existing_routes.empty:
            return self._fallback_similar(origin, destination)

        new_distance = self._get_distance(origin, destination)
        new_tier_pair = self._get_tier_pair(origin, destination)
        new_region_pair = self._get_region_pair(origin, destination)

        similarities = []
        for _, row in self._existing_routes.iterrows():
            o, d = row.get("origin", ""), row.get("destination", "")
            if not o or not d:
                continue

            # Distance similarity (exponential decay)
            route_dist = row.get("distance_km", self._get_distance(o, d))
            dist_sim = np.exp(-abs(new_distance - route_dist) / 500)

            # Tier similarity
            tier_pair = self._get_tier_pair(o, d)
            tier_sim = 1.0 if tier_pair == new_tier_pair else 0.5

            # Region similarity
            region_pair = self._get_region_pair(o, d)
            region_sim = 1.0 if region_pair == new_region_pair else 0.3

            # Overall similarity
            similarity = 0.50 * dist_sim + 0.25 * tier_sim + 0.25 * region_sim

            similarities.append({
                "route": f"{o}-{d}",
                "similarity": round(float(similarity), 4),
                "distance_km": round(float(route_dist), 0),
                "avg_price": round(float(row.get("avg_price", 0)), 2),
                "avg_demand": round(float(row.get("avg_demand", 0)), 0),
                "avg_load_factor": round(float(row.get("avg_load_factor", 0)), 4),
            })

        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]

    def _fallback_similar(self, origin: str, destination: str) -> list[dict]:
        """Fallback when no existing route data available."""
        distance = self._get_distance(origin, destination)
        tier_pair = self._get_tier_pair(origin, destination)

        # Heuristic pricing based on distance and tier
        base_per_km = {
            "T1-T1": 5.5,
            "T1-T2": 6.0,
            "T2-T2": 6.5,
            "T1-T3": 6.5,
            "T2-T3": 7.0,
            "T3-T3": 7.5,
        }
        rate = base_per_km.get(tier_pair, 6.5)
        estimated_price = max(2500, distance * rate)

        return [{
            "route": "heuristic_estimate",
            "similarity": 0.5,
            "distance_km": round(distance, 0),
            "avg_price": round(estimated_price, 2),
            "avg_demand": 80,
            "avg_load_factor": 0.70,
        }]

    def recommend(
        self,
        origin: str,
        destination: str,
        capacity: int = 180,
        confidence_level: float = 0.90,
    ) -> dict:
        """
        Generate cold start pricing recommendation for a new route.

        Returns conservative pricing with confidence intervals.
        """
        similar = self.find_similar_routes(origin, destination)
        distance = self._get_distance(origin, destination)
        tier_pair = self._get_tier_pair(origin, destination)

        if not similar:
            return {"error": "Cannot estimate — no reference data"}

        # Weighted average from similar routes
        weights = np.array([s["similarity"] for s in similar])
        weights = weights / weights.sum()

        prices = np.array([s["avg_price"] for s in similar])
        demands = np.array([s["avg_demand"] for s in similar])
        load_factors = np.array([s["avg_load_factor"] for s in similar])

        est_price = float(np.average(prices, weights=weights))
        est_demand = float(np.average(demands, weights=weights))
        est_load_factor = float(np.average(load_factors, weights=weights))

        # Bayesian prior: conservative adjustment
        # Use wider confidence interval for less-similar routes
        avg_similarity = float(np.mean(weights * len(weights)))  # un-normalized mean
        uncertainty = max(0.1, 1 - avg_similarity)

        # Conservative pricing: start lower to build demand
        conservative_discount = 0.85 if tier_pair.startswith("T3") else 0.90
        launch_price = est_price * conservative_discount

        # Confidence interval
        price_std = float(np.std(prices)) if len(prices) > 1 else est_price * 0.15
        ci_lower = est_price - 1.645 * price_std * uncertainty
        ci_upper = est_price + 1.645 * price_std * uncertainty

        # Fare class recommendation
        fare_classes = {
            "economy_promo": round(launch_price * 0.70, 2),
            "economy_standard": round(launch_price, 2),
            "economy_flex": round(launch_price * 1.30, 2),
            "business": round(launch_price * 2.50, 2),
        }

        # Ramp-up plan
        ramp_up = [
            {"week": "1-2", "load_target": 0.50, "price_multiplier": 0.85, "strategy": "Aggressive promo pricing"},
            {"week": "3-4", "load_target": 0.60, "price_multiplier": 0.90, "strategy": "Moderate discount"},
            {"week": "5-8", "load_target": 0.70, "price_multiplier": 0.95, "strategy": "Gradual normalization"},
            {"week": "9-12", "load_target": 0.75, "price_multiplier": 1.00, "strategy": "Standard pricing"},
            {"week": "13+", "load_target": 0.80, "price_multiplier": 1.05, "strategy": "Dynamic pricing activated"},
        ]

        return {
            "route": f"{origin}-{destination}",
            "distance_km": round(distance, 0),
            "tier_pair": tier_pair,
            "similar_routes": similar,
            "estimated_price": round(est_price, 2),
            "launch_price": round(launch_price, 2),
            "confidence_interval": {
                "lower": round(max(ci_lower, 1500), 2),
                "upper": round(ci_upper, 2),
                "confidence": confidence_level,
            },
            "estimated_demand": round(est_demand, 0),
            "estimated_load_factor": round(est_load_factor, 4),
            "data_confidence": round(avg_similarity, 4),
            "fare_classes": fare_classes,
            "capacity": capacity,
            "ramp_up_plan": ramp_up,
            "data_collection_recommendations": [
                "Monitor booking velocity daily for first 4 weeks",
                "A/B test pricing in week 3-4",
                "Switch to ML-based pricing after 500+ bookings",
                "Track competitor response weekly",
            ],
            "timestamp": datetime.now().isoformat(),
        }

    def _get_distance(self, origin: str, destination: str) -> float:
        o = CITY_METADATA.get(origin, {"lat": 20, "lon": 78})
        d = CITY_METADATA.get(destination, {"lat": 20, "lon": 78})
        return haversine_km(o["lat"], o["lon"], d["lat"], d["lon"])

    def _get_tier_pair(self, origin: str, destination: str) -> str:
        t1 = CITY_METADATA.get(origin, {}).get("tier", 3)
        t2 = CITY_METADATA.get(destination, {}).get("tier", 3)
        return f"T{min(t1, t2)}-T{max(t1, t2)}"

    def _get_region_pair(self, origin: str, destination: str) -> str:
        r1 = CITY_METADATA.get(origin, {}).get("region", "unknown")
        r2 = CITY_METADATA.get(destination, {}).get("region", "unknown")
        return f"{r1}-{r2}" if r1 <= r2 else f"{r2}-{r1}"


# Singleton
cold_start_strategy = ColdStartStrategy()
