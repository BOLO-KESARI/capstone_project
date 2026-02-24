"""
Fare Compliance & Regulatory Rules Engine.

Implements Indian DGCA aviation compliance:
  - Maximum fare limits
  - Denied boarding compensation rules
  - Cancellation compensation
  - Fare class regulations
  - Pricing transparency checks
  - Emergency pricing caps
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger("regulatory.compliance")


# ─── DGCA Fare Regulations (simplified) ─────────────────────────────────────

# Maximum base fare caps per distance slab (INR) – based on DGCA guidelines
FARE_CAPS = {
    "ultra_short": {"max_km": 400, "cap_economy": 5000, "cap_business": 15000},
    "short": {"max_km": 700, "cap_economy": 7500, "cap_business": 22000},
    "medium": {"max_km": 1200, "cap_economy": 10000, "cap_business": 30000},
    "long": {"max_km": 2000, "cap_economy": 15000, "cap_business": 45000},
    "ultra_long": {"max_km": 99999, "cap_economy": 20000, "cap_business": 60000},
}

# Denied boarding compensation (DGCA CAR Section 3, Series M, Part IV)
DENIED_BOARDING_COMPENSATION = {
    "within_1hr": {  # Alternate flight within 1 hour
        "up_to_1000km": 5000,
        "1000_to_2000km": 7500,
        "over_2000km": 10000,
    },
    "1hr_to_24hr": {  # Alternate within 1-24 hours
        "up_to_1000km": 10000,
        "1000_to_2000km": 15000,
        "over_2000km": 20000,
    },
    "over_24hr_or_none": {  # No alternate within 24 hours
        "up_to_1000km": 20000,
        "1000_to_2000km": 20000,
        "over_2000km": 20000,
    },
}

# Cancellation compensation
CANCELLATION_COMPENSATION = {
    "0_to_2_weeks": {"refund": "full", "compensation": 5000},
    "2_weeks_to_24hr": {"refund": "full", "compensation": 2500},
    "less_24hr": {"refund": "full", "compensation": 10000},
    "airline_fault": {"refund": "full", "compensation": "denied_boarding_equivalent"},
}

# Emergency pricing cap multiplier
EMERGENCY_PRICE_CAP_MULTIPLIER = 1.5  # Cannot exceed 1.5x normal fare during emergencies


class ComplianceEngine:
    """
    DGCA fare compliance and regulatory checks.

    Usage:
        compliance = ComplianceEngine()
        result = compliance.check_fare_compliance(
            price=12000, distance_km=800, fare_class="economy"
        )
    """

    def __init__(self):
        self._violations: list[dict] = []

    def check_fare_compliance(
        self,
        price: float,
        distance_km: float,
        fare_class: str = "economy",
        is_emergency: bool = False,
        base_fare: float = None,
    ) -> dict:
        """
        Check if a fare complies with DGCA regulations.

        Returns approval status and any violations.
        """
        violations = []
        warnings = []

        # Determine distance slab
        slab = None
        for slab_name, slab_data in FARE_CAPS.items():
            if distance_km <= slab_data["max_km"]:
                slab = slab_data
                slab_key = slab_name
                break

        if slab is None:
            slab = FARE_CAPS["ultra_long"]
            slab_key = "ultra_long"

        # Check fare cap
        cap_key = f"cap_{fare_class}" if f"cap_{fare_class}" in slab else "cap_economy"
        fare_cap = slab.get(cap_key, slab["cap_economy"])

        if is_emergency:
            fare_cap = fare_cap * EMERGENCY_PRICE_CAP_MULTIPLIER

        if price > fare_cap:
            violations.append({
                "rule": "DGCA_FARE_CAP",
                "description": f"Fare ₹{price:.0f} exceeds {slab_key} cap ₹{fare_cap:.0f}",
                "cap": fare_cap,
                "excess": round(price - fare_cap, 2),
                "excess_pct": round((price - fare_cap) / fare_cap * 100, 2),
                "severity": "critical",
            })

        # Check emergency pricing
        if is_emergency and base_fare:
            emergency_cap = base_fare * EMERGENCY_PRICE_CAP_MULTIPLIER
            if price > emergency_cap:
                violations.append({
                    "rule": "EMERGENCY_PRICE_CAP",
                    "description": f"Emergency fare ₹{price:.0f} exceeds 1.5x base fare ₹{base_fare:.0f}",
                    "cap": round(emergency_cap, 2),
                    "severity": "critical",
                })

        # Price reasonableness check
        per_km_rate = price / max(distance_km, 1)
        if per_km_rate > 20:
            warnings.append({
                "rule": "HIGH_PER_KM_RATE",
                "description": f"Per-km rate ₹{per_km_rate:.2f} is unusually high",
                "rate": round(per_km_rate, 2),
                "severity": "warning",
            })

        if price < 1000 and distance_km > 300:
            warnings.append({
                "rule": "BELOW_COST_PRICING",
                "description": f"Fare ₹{price:.0f} may be below operating cost for {distance_km:.0f}km",
                "severity": "warning",
            })

        compliant = len(violations) == 0
        result = {
            "compliant": compliant,
            "price": price,
            "distance_km": distance_km,
            "fare_class": fare_class,
            "distance_slab": slab_key,
            "fare_cap": fare_cap,
            "is_emergency": is_emergency,
            "violations": violations,
            "warnings": warnings,
            "checked_at": datetime.now().isoformat(),
        }

        if violations:
            self._violations.append(result)

        return result

    def calculate_denied_boarding_compensation(
        self,
        distance_km: float,
        alternate_hours: float = None,
    ) -> dict:
        """
        Calculate denied boarding compensation per DGCA rules.

        alternate_hours: hours until alternate flight (None = no alternate)
        """
        # Distance bucket
        if distance_km <= 1000:
            dist_key = "up_to_1000km"
        elif distance_km <= 2000:
            dist_key = "1000_to_2000km"
        else:
            dist_key = "over_2000km"

        # Time bucket
        if alternate_hours is not None and alternate_hours <= 1:
            time_key = "within_1hr"
        elif alternate_hours is not None and alternate_hours <= 24:
            time_key = "1hr_to_24hr"
        else:
            time_key = "over_24hr_or_none"

        compensation = DENIED_BOARDING_COMPENSATION[time_key][dist_key]

        return {
            "distance_km": distance_km,
            "alternate_hours": alternate_hours,
            "compensation_inr": compensation,
            "distance_category": dist_key,
            "time_category": time_key,
            "additional_rights": [
                "Full refund of ticket price",
                "Complimentary meals and refreshments during wait",
                "Free hotel accommodation if overnight delay",
                "Two free phone calls / emails",
            ],
            "dgca_reference": "CAR Section 3, Series M, Part IV",
        }

    def calculate_cancellation_compensation(
        self,
        notice_hours: float,
        ticket_price: float,
        airline_fault: bool = True,
        distance_km: float = 1000,
    ) -> dict:
        """Calculate cancellation compensation."""
        if not airline_fault:
            return {
                "compensation_inr": 0,
                "refund": "As per fare rules",
                "reason": "Cancellation not due to airline",
            }

        if notice_hours < 24:
            comp = CANCELLATION_COMPENSATION["less_24hr"]
            fixed_comp = comp["compensation"]
        elif notice_hours < 336:  # 2 weeks
            comp = CANCELLATION_COMPENSATION["2_weeks_to_24hr"]
            fixed_comp = comp["compensation"]
        else:
            comp = CANCELLATION_COMPENSATION["0_to_2_weeks"]
            fixed_comp = comp["compensation"]

        return {
            "distance_km": distance_km,
            "notice_hours": notice_hours,
            "ticket_price": ticket_price,
            "refund": "Full ticket price",
            "refund_amount": ticket_price,
            "additional_compensation": fixed_comp,
            "total_payout": ticket_price + fixed_comp,
            "airline_fault": airline_fault,
        }

    def overbooking_risk_check(
        self,
        booked_pax: int,
        capacity: int,
        historical_no_show_rate: float = 0.05,
    ) -> dict:
        """
        Check if overbooking level is within acceptable limits.
        DGCA allows reasonable overbooking (~5-8%) based on no-show rates.
        """
        overbooking_pct = (booked_pax - capacity) / capacity * 100 if booked_pax > capacity else 0
        expected_no_shows = int(booked_pax * historical_no_show_rate)
        expected_actual = booked_pax - expected_no_shows

        denied_boarding_risk = max(0, expected_actual - capacity)
        risk_level = "low"
        if denied_boarding_risk > 5:
            risk_level = "high"
        elif denied_boarding_risk > 2:
            risk_level = "medium"

        max_allowed_overbooking = capacity * (1 + historical_no_show_rate + 0.02)

        return {
            "booked_passengers": booked_pax,
            "capacity": capacity,
            "overbooking_pct": round(overbooking_pct, 2),
            "historical_no_show_rate": historical_no_show_rate,
            "expected_no_shows": expected_no_shows,
            "expected_actual_pax": expected_actual,
            "denied_boarding_risk": denied_boarding_risk,
            "risk_level": risk_level,
            "max_recommended_bookings": int(max_allowed_overbooking),
            "compliant": booked_pax <= max_allowed_overbooking,
            "est_compensation_exposure": denied_boarding_risk * 10000 if denied_boarding_risk > 0 else 0,
        }

    def batch_check(self, fares: list[dict]) -> dict:
        """
        Batch compliance check for multiple fares.

        fares: [{"price": ..., "distance_km": ..., "fare_class": ..., "route": ...}, ...]
        """
        results = []
        violations_count = 0
        for fare in fares:
            result = self.check_fare_compliance(
                price=fare.get("price", 0),
                distance_km=fare.get("distance_km", 0),
                fare_class=fare.get("fare_class", "economy"),
                is_emergency=fare.get("is_emergency", False),
            )
            result["route"] = fare.get("route", "unknown")
            results.append(result)
            if not result["compliant"]:
                violations_count += 1

        return {
            "total_checked": len(results),
            "compliant": len(results) - violations_count,
            "non_compliant": violations_count,
            "compliance_rate": round((len(results) - violations_count) / max(len(results), 1) * 100, 2),
            "details": results,
            "checked_at": datetime.now().isoformat(),
        }

    def get_violations(self, limit: int = 50) -> list[dict]:
        """Return recent violations."""
        return self._violations[-limit:]


# Singleton
compliance_engine = ComplianceEngine()
