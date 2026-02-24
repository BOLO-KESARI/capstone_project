"""
Model Versioning & Registry.

Implements:
  - Model version tracking with metadata
  - Performance metrics history
  - Model promotion / retirement
  - Integration with SQLite model_registry table
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("ml_advanced.model_registry")

MODEL_DIR = Path(__file__).resolve().parent.parent / "model_artifacts"
MODEL_DIR.mkdir(exist_ok=True)


class ModelVersion:
    """Represents a single model version."""

    def __init__(
        self,
        name: str,
        version: int,
        model: Any,
        metrics: dict,
        features: list[str],
        hyperparameters: dict = None,
        notes: str = "",
    ):
        self.name = name
        self.version = version
        self.model = model
        self.metrics = metrics
        self.features = features
        self.hyperparameters = hyperparameters or {}
        self.notes = notes
        self.created_at = datetime.now().isoformat()
        self.status = "staging"  # staging → production → retired
        self.model_hash = self._hash_model(model)

    @staticmethod
    def _hash_model(model: Any) -> str:
        try:
            data = pickle.dumps(model)
            return hashlib.sha256(data).hexdigest()[:16]
        except Exception:
            return hashlib.sha256(str(model).encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "status": self.status,
            "metrics": self.metrics,
            "features": self.features,
            "hyperparameters": self.hyperparameters,
            "notes": self.notes,
            "created_at": self.created_at,
            "model_hash": self.model_hash,
        }


class ModelRegistry:
    """
    In-memory + SQLite model registry.

    Usage:
        registry = ModelRegistry()
        registry.register("pricing_model", model, {"mape": 5.2, "r2": 0.91}, feature_list)
        production_model = registry.get_production("pricing_model")
    """

    def __init__(self):
        self._models: dict[str, list[ModelVersion]] = {}
        self._try_load_from_db()

    def _try_load_from_db(self):
        """Load model metadata from SQLite if available."""
        try:
            from database.schema import query
            rows = query("SELECT * FROM model_registry ORDER BY registered_at DESC")
            for row in rows:
                name = row["model_name"]
                if name not in self._models:
                    self._models[name] = []
                # Metadata only – actual model objects are registered at startup
                logger.debug(f"[Registry] Found DB record: {name} v{row['version']}")
        except Exception:
            pass

    def register(
        self,
        name: str,
        model: Any,
        metrics: dict,
        features: list[str],
        hyperparameters: dict = None,
        notes: str = "",
        auto_promote: bool = False,
    ) -> ModelVersion:
        """Register a new model version."""
        if name not in self._models:
            self._models[name] = []

        version = len(self._models[name]) + 1
        mv = ModelVersion(name, version, model, metrics, features, hyperparameters, notes)

        # Auto-promote if first version or if better than current production
        if auto_promote:
            current_prod = self.get_production(name)
            if current_prod is None:
                mv.status = "production"
            else:
                # Compare MAPE (lower is better) or R² (higher is better)
                if "mape" in metrics and "mape" in current_prod.metrics:
                    if metrics["mape"] < current_prod.metrics["mape"]:
                        current_prod.status = "retired"
                        mv.status = "production"
                elif "r2" in metrics and "r2" in current_prod.metrics:
                    if metrics["r2"] > current_prod.metrics["r2"]:
                        current_prod.status = "retired"
                        mv.status = "production"

        self._models[name].append(mv)
        self._persist_to_db(mv)

        # Save model artifact
        self._save_artifact(mv)

        logger.info(f"[Registry] Registered {name} v{version} (status={mv.status})")
        return mv

    def _persist_to_db(self, mv: ModelVersion):
        """Save model metadata to SQLite."""
        try:
            from database.schema import execute
            execute(
                """INSERT OR REPLACE INTO model_registry
                   (model_name, version, algorithm, hyperparameters, metrics, status, registered_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    mv.name,
                    f"v{mv.version}",
                    type(mv.model).__name__,
                    json.dumps(mv.hyperparameters),
                    json.dumps(mv.metrics),
                    mv.status,
                    mv.created_at,
                ),
            )
        except Exception:
            pass  # DB not available

    def _save_artifact(self, mv: ModelVersion):
        """Persist model binary to disk."""
        try:
            path = MODEL_DIR / f"{mv.name}_v{mv.version}.pkl"
            with open(path, "wb") as f:
                pickle.dump(mv.model, f)
            logger.debug(f"[Registry] Saved artifact to {path}")
        except Exception as e:
            logger.warning(f"[Registry] Could not save artifact: {e}")

    def get_production(self, name: str) -> ModelVersion | None:
        """Get the current production model."""
        if name not in self._models:
            return None
        for mv in reversed(self._models[name]):
            if mv.status == "production":
                return mv
        return None

    def promote(self, name: str, version: int) -> dict:
        """Promote a model version to production; retire the previous."""
        if name not in self._models:
            return {"error": f"Model '{name}' not found"}

        target = None
        for mv in self._models[name]:
            if mv.version == version:
                target = mv
                break

        if not target:
            return {"error": f"Version {version} not found"}

        # Retire current production
        current = self.get_production(name)
        if current:
            current.status = "retired"
            self._persist_to_db(current)

        target.status = "production"
        self._persist_to_db(target)
        return {"promoted": target.to_dict()}

    def retire(self, name: str, version: int) -> dict:
        """Retire a model version."""
        if name not in self._models:
            return {"error": f"Model '{name}' not found"}
        for mv in self._models[name]:
            if mv.version == version:
                mv.status = "retired"
                self._persist_to_db(mv)
                return {"retired": mv.to_dict()}
        return {"error": f"Version {version} not found"}

    def get_all_versions(self, name: str) -> list[dict]:
        """Get all versions of a model."""
        if name not in self._models:
            return []
        return [mv.to_dict() for mv in self._models[name]]

    def compare_versions(self, name: str, v1: int, v2: int) -> dict:
        """Compare two model versions."""
        versions = self._models.get(name, [])
        mv1 = next((mv for mv in versions if mv.version == v1), None)
        mv2 = next((mv for mv in versions if mv.version == v2), None)

        if not mv1 or not mv2:
            return {"error": "Version(s) not found"}

        # metric comparison
        comparison = {}
        all_keys = set(list(mv1.metrics.keys()) + list(mv2.metrics.keys()))
        for key in all_keys:
            val1 = mv1.metrics.get(key)
            val2 = mv2.metrics.get(key)
            if val1 is not None and val2 is not None:
                comparison[key] = {
                    "v1": val1,
                    "v2": val2,
                    "delta": round(val2 - val1, 4),
                    "improved": (val2 < val1 if key in ("mape", "rmse", "mae") else val2 > val1),
                }

        return {
            "model": name,
            "v1": mv1.to_dict(),
            "v2": mv2.to_dict(),
            "metric_comparison": comparison,
        }

    def summary(self) -> dict:
        """Summary of all registered models."""
        result = {}
        for name, versions in self._models.items():
            prod = self.get_production(name)
            result[name] = {
                "total_versions": len(versions),
                "production_version": prod.version if prod else None,
                "production_metrics": prod.metrics if prod else None,
                "latest_version": versions[-1].version if versions else None,
                "latest_status": versions[-1].status if versions else None,
            }
        return {
            "total_models": len(result),
            "models": result,
            "artifact_dir": str(MODEL_DIR),
        }


# Singleton
model_registry = ModelRegistry()
