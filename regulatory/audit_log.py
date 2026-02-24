"""
Persistent Audit Logging System.

Implements:
  - Immutable audit trail for all pricing decisions
  - Model prediction logging
  - User action tracking
  - SQLite-backed persistence
  - Query/filter interface
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger("regulatory.audit_log")


class AuditLogger:
    """
    Audit logger with in-memory buffer + SQLite persistence.

    Usage:
        audit = AuditLogger()
        audit.log("pricing_approved", user="analyst", details={...})
        recent = audit.query(event_type="pricing_approved", limit=50)
    """

    def __init__(self, max_memory: int = 5000):
        self._buffer: list[dict] = []
        self._max_memory = max_memory
        self._counter = 0

    def log(
        self,
        event_type: str,
        user: str = "system",
        details: dict = None,
        route: str = None,
        model_name: str = None,
        severity: str = "info",
    ) -> dict:
        """
        Record an audit event.

        event_type: pricing_approved, pricing_overridden, model_retrained,
                    user_login, config_changed, alert_triggered, etc.
        """
        self._counter += 1
        entry = {
            "id": self._counter,
            "event_type": event_type,
            "user": user,
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "route": route,
            "model_name": model_name,
            "details": details or {},
        }

        self._buffer.append(entry)
        if len(self._buffer) > self._max_memory:
            self._buffer = self._buffer[-self._max_memory:]

        # Persist to SQLite
        self._persist(entry)

        if severity in ("warning", "critical"):
            logger.warning(f"[Audit] {event_type}: {json.dumps(details or {})[:200]}")

        return entry

    def _persist(self, entry: dict):
        """Write to SQLite audit_log table."""
        try:
            from database.schema import execute
            execute(
                """INSERT INTO audit_log (event_type, user, details, timestamp, severity)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    entry["event_type"],
                    entry["user"],
                    json.dumps(entry["details"]),
                    entry["timestamp"],
                    entry["severity"],
                ),
            )
        except Exception:
            pass  # DB may not be initialized

    def query(
        self,
        event_type: str = None,
        user: str = None,
        severity: str = None,
        route: str = None,
        since: str = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query audit log with filters."""
        results = self._buffer

        if event_type:
            results = [e for e in results if e["event_type"] == event_type]
        if user:
            results = [e for e in results if e["user"] == user]
        if severity:
            results = [e for e in results if e["severity"] == severity]
        if route:
            results = [e for e in results if e.get("route") == route]
        if since:
            results = [e for e in results if e["timestamp"] >= since]

        return results[-limit:]

    def pricing_decision(
        self,
        route: str,
        old_price: float,
        new_price: float,
        model_used: str,
        confidence: float,
        approved_by: str,
        reason: str = "",
    ) -> dict:
        """Log a pricing decision with full context."""
        return self.log(
            event_type="pricing_decision",
            user=approved_by,
            route=route,
            model_name=model_used,
            severity="info",
            details={
                "old_price": old_price,
                "new_price": new_price,
                "price_change_pct": round((new_price - old_price) / (old_price + 1e-9) * 100, 2),
                "model_confidence": confidence,
                "reason": reason,
            },
        )

    def model_event(
        self,
        model_name: str,
        event: str,
        metrics: dict = None,
        user: str = "system",
    ) -> dict:
        """Log a model lifecycle event (train, deploy, retire, drift)."""
        return self.log(
            event_type=f"model_{event}",
            user=user,
            model_name=model_name,
            details={"metrics": metrics or {}, "action": event},
        )

    def alert_event(
        self,
        alert_type: str,
        message: str,
        route: str = None,
        severity: str = "warning",
    ) -> dict:
        """Log an alert event."""
        return self.log(
            event_type="alert_triggered",
            severity=severity,
            route=route,
            details={"alert_type": alert_type, "message": message},
        )

    def summary(self, hours: int = 24) -> dict:
        """Summary of recent audit activity."""
        from datetime import timedelta

        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        recent = [e for e in self._buffer if e["timestamp"] >= cutoff]

        by_type = {}
        for e in recent:
            t = e["event_type"]
            by_type[t] = by_type.get(t, 0) + 1

        by_severity = {}
        for e in recent:
            s = e["severity"]
            by_severity[s] = by_severity.get(s, 0) + 1

        return {
            "period_hours": hours,
            "total_events": len(recent),
            "by_event_type": by_type,
            "by_severity": by_severity,
            "total_buffered": len(self._buffer),
            "generated_at": datetime.now().isoformat(),
        }


# Singleton
audit_logger = AuditLogger()
