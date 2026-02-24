"""
Pytest test suite for LNT Aviation Dashboard.
Covers: FastAPI endpoints, ML models, pricing approval gate, health check, alerts.
Run: pytest tests/ -v
"""
import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

client = TestClient(app)


# ================================================================== #
#  Health Check
# ================================================================== #
class TestHealthCheck:
    def test_health_returns_200(self):
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in ("healthy", "degraded")
        assert "models" in data
        assert "data_files" in data

    def test_health_has_model_count(self):
        r = client.get("/api/health")
        data = r.json()
        assert data["total_models_expected"] >= 10
        assert data["total_models_loaded"] >= 0


# ================================================================== #
#  Dashboard Summary
# ================================================================== #
class TestDashboardSummary:
    def test_dashboard_summary_200(self):
        r = client.get("/api/dashboard/summary")
        assert r.status_code == 200

    def test_dashboard_has_kpis(self):
        r = client.get("/api/dashboard/summary")
        data = r.json()
        assert "kpis" in data
        assert "total_revenue" in data["kpis"]
        assert "total_bookings" in data["kpis"]

    def test_dashboard_has_revenue_data(self):
        r = client.get("/api/dashboard/summary")
        data = r.json()
        assert "revenue_by_date" in data
        assert "revenue_by_route" in data
        assert "monthly_revenue" in data

    def test_dashboard_has_alerts(self):
        r = client.get("/api/dashboard/summary")
        data = r.json()
        assert "alerts" in data

    def test_dashboard_has_forecast(self):
        r = client.get("/api/dashboard/summary")
        data = r.json()
        assert "demand_forecast" in data

    def test_dashboard_has_rev_target(self):
        r = client.get("/api/dashboard/summary")
        data = r.json()
        assert "rev_target" in data

    def test_dashboard_has_cap_vs_demand(self):
        r = client.get("/api/dashboard/summary")
        data = r.json()
        assert "cap_vs_demand" in data


# ================================================================== #
#  ML Prediction Endpoints
# ================================================================== #
class TestMLDemand:
    def test_demand_prediction(self):
        r = client.post("/api/ml/predict/demand", json={
            "route": "DEL-BOM", "month": 6, "year": 2024,
            "historical_pax": 30000, "total_flights": 250,
            "cancellation_rate": 3, "delay_rate": 20,
            "weather_rate": 1, "seasonal_index": 1.1,
        })
        assert r.status_code == 200
        data = r.json()
        assert "predicted_demand" in data
        assert data["predicted_demand"] > 0


class TestMLPricing:
    def test_pricing_prediction(self):
        r = client.post("/api/ml/predict/pricing", json={
            "passenger_demand": 35000, "load_factor": 80,
            "operating_cost": 5000, "fuel_cost": 2000,
            "delay_rate": 20, "cancellation_rate": 3, "seasonal_index": 1.0,
        })
        assert r.status_code == 200
        assert "predicted_price" in r.json()


class TestMLOverbooking:
    def test_overbooking_simulation(self):
        r = client.post("/api/ml/predict/overbooking", json={
            "seat_capacity": 180, "no_show_rate": 6,
            "cancel_rate": 4, "ticket_price": 8000,
            "compensation_cost": 20000,
        })
        assert r.status_code == 200
        data = r.json()
        assert "optimal_extra_seats" in data
        assert "breakdown" in data
        assert len(data["breakdown"]) == 25


class TestMLProfitability:
    def test_profitability_prediction(self):
        r = client.post("/api/ml/predict/profitability", json={
            "passenger_count": 12000, "load_factor": 75,
            "operating_cost": 1200000, "flights_per_route": 50,
            "fuel_cost": 500000,
        })
        assert r.status_code == 200
        data = r.json()
        assert "predicted_profit" in data
        assert "margin" in data


class TestMLRisk:
    def test_risk_prediction(self):
        r = client.post("/api/ml/predict/risk", json={
            "route_type": "High-Traffic",
            "delay_rate": 20, "cancellation_rate": 5,
            "weather_rate": 3, "tech_fault_rate": 2,
        })
        assert r.status_code == 200
        data = r.json()
        assert "predicted_risk" in data
        assert "confidence" in data


class TestMLChurn:
    def test_churn_prediction(self):
        r = client.post("/api/ml/predict/churn", json={
            "age": 35, "gender": "M", "loyalty_tier": "Silver",
            "total_miles_flown": 25000, "lifetime_spend": 500000,
            "avg_ticket_price": 8000, "cancellation_rate": 10,
            "upgrade_history_count": 3, "ancillary_spend_avg": 1200,
        })
        assert r.status_code == 200
        data = r.json()
        assert "will_churn" in data
        assert "churn_probability" in data
        assert "risk_level" in data

    def test_churn_has_recommendation(self):
        r = client.post("/api/ml/predict/churn", json={
            "age": 25, "gender": "F", "loyalty_tier": "Blue",
            "total_miles_flown": 5000, "lifetime_spend": 50000,
            "avg_ticket_price": 4000, "cancellation_rate": 40,
            "upgrade_history_count": 0, "ancillary_spend_avg": 200,
        })
        assert r.status_code == 200
        assert "recommendation" in r.json()


class TestMLDelay:
    def test_delay_prediction(self):
        r = client.post("/api/ml/predict/delay", json={
            "origin_airport": "DEL", "departure_hour": 9,
            "turnaround_time": 45, "aircraft_utilization": 10,
            "crew_delay": False, "technical_issue": False,
            "load_factor": 0.85, "distance_km": 1400,
            "seat_capacity": 180, "maintenance_flag": False,
        })
        assert r.status_code == 200
        data = r.json()
        assert "will_be_delayed" in data
        assert "delay_probability" in data


class TestMLCancellation:
    def test_cancellation_prediction(self):
        r = client.post("/api/ml/predict/cancellation", json={
            "fare_class": "Economy", "base_fare": 8000, "taxes": 400,
            "ancillary_revenue": 500, "total_price_paid": 8900,
            "discount_applied": 0, "booking_channel": "Mobile App",
            "passenger_count": 1, "payment_method": "UPI",
            "days_before_departure": 14,
        })
        assert r.status_code == 200
        data = r.json()
        assert "will_cancel" in data
        assert "revenue_at_risk" in data


class TestMLLoadFactor:
    def test_load_factor_prediction(self):
        r = client.post("/api/ml/predict/load_factor", json={
            "seat_capacity": 180, "distance_km": 1200,
            "lead_time_days": 30, "booking_velocity": 8,
            "month": 10, "is_weekend": 0, "peak_season": 0,
            "comp_avg_price": 7500, "base_fare": 8000,
        })
        assert r.status_code == 200
        data = r.json()
        assert "predicted_load_factor" in data
        assert 0 <= data["predicted_load_factor"] <= 100
        assert "risk_level" in data


class TestMLNoShow:
    def test_noshow_prediction(self):
        r = client.post("/api/ml/predict/noshow", json={
            "fare_class": "Economy", "lead_time_days": 21,
            "base_fare": 7000, "discount_applied": 5,
            "passenger_count": 1, "is_weekend": 0,
            "peak_season": 0, "distance_km": 1200,
        })
        assert r.status_code == 200
        data = r.json()
        assert "will_noshow" in data
        assert "noshow_probability" in data


class TestMLCluster:
    def test_cluster_prediction(self):
        r = client.post("/api/ml/predict/cluster", json={
            "age": 35, "total_miles_flown": 30000,
            "lifetime_spend": 400000, "avg_ticket_price": 7500,
            "cancellation_rate": 8, "upgrade_history_count": 2,
            "ancillary_spend_avg": 1000,
        })
        assert r.status_code == 200
        data = r.json()
        assert "cluster_id" in data
        assert "cluster_name" in data
        assert "all_clusters" in data


# ================================================================== #
#  Explainability (G5)
# ================================================================== #
class TestExplainability:
    def test_list_explainable_models(self):
        r = client.get("/api/ml/explain")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data
        assert data["count"] > 0

    def test_explain_demand_model(self):
        r = client.get("/api/ml/explain/demand")
        assert r.status_code == 200
        data = r.json()
        assert "feature_importance" in data
        assert len(data["feature_importance"]) > 0
        assert "interpretation" in data

    def test_explain_clustering(self):
        r = client.get("/api/ml/explain/clustering")
        assert r.status_code == 200
        data = r.json()
        assert data["method"] == "cluster_analysis"
        assert "cluster_names" in data

    def test_explain_unknown_model(self):
        r = client.get("/api/ml/explain/nonexistent_model")
        assert r.status_code == 404


# ================================================================== #
#  Pricing Approval Gate (G28)
# ================================================================== #
class TestPricingApproval:
    def test_submit_proposal(self):
        r = client.post("/api/pricing/propose", json={
            "route": "DEL-BOM", "current_price": 8000,
            "proposed_price": 9000, "reason": "Demand surge",
            "change_type": "increase",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "pending"
        assert "proposal_id" in data

    def test_auto_reject_large_change(self):
        r = client.post("/api/pricing/propose", json={
            "route": "BOM-BLR", "current_price": 5000,
            "proposed_price": 10000, "reason": "Testing",
            "change_type": "increase",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "auto_rejected"

    def test_approve_proposal(self):
        r1 = client.post("/api/pricing/propose", json={
            "route": "MAA-CCU", "current_price": 7000,
            "proposed_price": 7500, "reason": "Market adjustment",
            "change_type": "increase",
        })
        pid = r1.json()["proposal_id"]
        r2 = client.post(f"/api/pricing/review/{pid}", json={
            "action": "approve", "reviewer": "admin",
            "comments": "Approved per market analysis",
        })
        assert r2.status_code == 200
        assert r2.json()["status"] == "approved"

    def test_reject_proposal(self):
        r1 = client.post("/api/pricing/propose", json={
            "route": "CCU-DEL", "current_price": 9000,
            "proposed_price": 8000, "reason": "Competitor match",
            "change_type": "decrease",
        })
        pid = r1.json()["proposal_id"]
        r2 = client.post(f"/api/pricing/review/{pid}", json={
            "action": "reject", "reviewer": "admin",
            "comments": "Not enough justification",
        })
        assert r2.status_code == 200
        assert r2.json()["status"] == "rejected"

    def test_queue_listing(self):
        r = client.get("/api/pricing/queue")
        assert r.status_code == 200
        data = r.json()
        assert "proposals" in data
        assert "pending" in data

    def test_queue_filter_by_status(self):
        r = client.get("/api/pricing/queue?status=pending")
        assert r.status_code == 200


# ================================================================== #
#  Alerts
# ================================================================== #
class TestAlerts:
    def test_alerts_endpoint(self):
        r = client.get("/api/alerts")
        assert r.status_code == 200
        data = r.json()
        assert "alerts" in data
        assert "count" in data


# ================================================================== #
#  Data Endpoints
# ================================================================== #
class TestDataEndpoints:
    def test_flights_data(self):
        r = client.get("/api/data/flights")
        assert r.status_code == 200

    def test_bookings_data(self):
        r = client.get("/api/data/bookings")
        assert r.status_code == 200

    def test_passengers_data(self):
        r = client.get("/api/data/passengers")
        assert r.status_code == 200


# ================================================================== #
#  Input Validation
# ================================================================== #
class TestValidation:
    def test_demand_invalid_month(self):
        r = client.post("/api/ml/predict/demand", json={
            "route": "DEL-BOM", "month": 13, "year": 2024,
            "historical_pax": 30000, "total_flights": 250,
            "cancellation_rate": 3, "delay_rate": 20,
            "weather_rate": 1, "seasonal_index": 1.0,
        })
        assert r.status_code == 422  # Pydantic validation error

    def test_churn_invalid_age(self):
        r = client.post("/api/ml/predict/churn", json={
            "age": 5, "gender": "M", "loyalty_tier": "Silver",
            "total_miles_flown": 25000, "lifetime_spend": 500000,
            "avg_ticket_price": 8000, "cancellation_rate": 10,
            "upgrade_history_count": 3, "ancillary_spend_avg": 1200,
        })
        assert r.status_code == 422

    def test_delay_missing_field(self):
        r = client.post("/api/ml/predict/delay", json={
            "origin_airport": "DEL",
            # Missing required fields
        })
        assert r.status_code == 422

    def test_pricing_review_not_found(self):
        r = client.post("/api/pricing/review/nonexistent", json={
            "action": "approve", "reviewer": "admin",
        })
        assert r.status_code == 404


# ================================================================== #
#  Dashboard HTML Pages
# ================================================================== #
class TestPages:
    def test_index_page(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_dashboard_page(self):
        r = client.get("/dashboard")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
