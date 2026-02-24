"""
ETL Scheduler – APScheduler-based batch jobs for LNT Aviation.

Jobs:
  - Daily demand aggregation
  - Hourly search trend refresh
  - Daily competitor pricing refresh
  - Weekly economic indicator refresh
  - Weather sync (every 30 min)
  - Social sentiment batch processing (every 6 hours)
  - Materialized view refresh (daily)
  - Model retraining check (daily)
"""

from __future__ import annotations

import os
import logging
from datetime import datetime
from typing import Callable

import pandas as pd

logger = logging.getLogger("etl.scheduler")


# ------------------------------------------------------------------ #
#  Job definitions
# ------------------------------------------------------------------ #

def job_daily_demand_aggregation():
    """Aggregate daily demand metrics per route into mv_daily_booking_pace."""
    from database.schema import get_connection
    logger.info("[ETL] Running daily demand aggregation...")
    try:
        with get_connection() as conn:
            conn.execute("DELETE FROM mv_daily_booking_pace")
            conn.execute("""
                INSERT INTO mv_daily_booking_pace (route_id, departure_date, days_out, cumulative_bkgs, pace_vs_hist)
                SELECT
                    route_id,
                    departure_date,
                    CAST(julianday(departure_date) - julianday(booking_date) AS INTEGER) as days_out,
                    COUNT(*) as cumulative_bkgs,
                    1.0 as pace_vs_hist
                FROM fact_bookings
                WHERE booking_status = 'Confirmed' AND route_id IS NOT NULL
                GROUP BY route_id, departure_date,
                    CAST(julianday(departure_date) - julianday(booking_date) AS INTEGER)
            """)
        logger.info("[ETL] Daily demand aggregation complete.")
    except Exception as e:
        logger.error(f"[ETL] Demand aggregation failed: {e}")


def job_refresh_route_revenue():
    """Refresh mv_route_revenue materialized view."""
    from database.schema import get_connection
    logger.info("[ETL] Refreshing route revenue view...")
    try:
        with get_connection() as conn:
            conn.execute("DELETE FROM mv_route_revenue")
            conn.execute("""
                INSERT INTO mv_route_revenue (route_id, route, period, total_revenue, ticket_revenue, ancillary_rev, avg_fare, pax_count)
                SELECT
                    fr.route_id,
                    dr.origin_iata || '-' || dr.dest_iata,
                    strftime('%Y-%m', fr.revenue_date),
                    SUM(fr.total_revenue),
                    SUM(fr.ticket_revenue),
                    SUM(fr.ancillary_rev),
                    AVG(fr.total_revenue / NULLIF(fr.pax_count, 0)),
                    SUM(fr.pax_count)
                FROM fact_revenue fr
                JOIN dim_route dr ON fr.route_id = dr.route_id
                GROUP BY fr.route_id, strftime('%Y-%m', fr.revenue_date)
            """)
        logger.info("[ETL] Route revenue view refreshed.")
    except Exception as e:
        logger.error(f"[ETL] Route revenue refresh failed: {e}")


def job_refresh_load_factor():
    """Refresh mv_load_factor_agg materialized view."""
    from database.schema import get_connection
    logger.info("[ETL] Refreshing load factor aggregation...")
    try:
        with get_connection() as conn:
            conn.execute("DELETE FROM mv_load_factor_agg")
            conn.execute("""
                INSERT INTO mv_load_factor_agg (route_id, route, period, avg_load_factor, min_load_factor, max_load_factor, flights_count, below_target)
                SELECT
                    ff.route_id,
                    dr.origin_iata || '-' || dr.dest_iata,
                    strftime('%Y-%m', ff.scheduled_dep),
                    AVG(ff.load_factor),
                    MIN(ff.load_factor),
                    MAX(ff.load_factor),
                    COUNT(*),
                    SUM(CASE WHEN ff.load_factor < 0.70 THEN 1 ELSE 0 END)
                FROM fact_flights ff
                JOIN dim_route dr ON ff.route_id = dr.route_id
                WHERE ff.load_factor IS NOT NULL
                GROUP BY ff.route_id, strftime('%Y-%m', ff.scheduled_dep)
            """)
        logger.info("[ETL] Load factor aggregation refreshed.")
    except Exception as e:
        logger.error(f"[ETL] Load factor refresh failed: {e}")


def job_competitor_price_refresh():
    """Refresh competitor prices (simulated – in production, scrape external sources)."""
    from database.schema import get_connection
    logger.info("[ETL] Refreshing competitor prices...")
    try:
        sim_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "simulated_data", "competitor_prices.csv")
        if os.path.exists(sim_path):
            df = pd.read_csv(sim_path)
            with get_connection() as conn:
                for _, r in df.iterrows():
                    conn.execute(
                        """INSERT INTO coll_competitor_prices (route, competitor, price, fare_class, source)
                           VALUES (?, ?, ?, ?, 'etl_refresh')""",
                        (r.get("route", ""), r.get("competitor_airline", ""),
                         r.get("competitor_price", 0), r.get("fare_class", "Economy")),
                    )
        logger.info("[ETL] Competitor prices refreshed.")
    except Exception as e:
        logger.error(f"[ETL] Competitor price refresh failed: {e}")


def job_weather_sync():
    """Sync weather snapshots for all major airports."""
    from database.schema import get_connection
    logger.info("[ETL] Syncing weather data...")
    try:
        import httpx
        airports = ["DEL", "BOM", "BLR", "MAA", "HYD", "CCU", "PNQ", "GOI"]
        # Simulated weather data (in production, call actual weather API)
        import random
        with get_connection() as conn:
            for apt in airports:
                conn.execute(
                    """INSERT INTO coll_weather_snapshots
                       (airport_iata, temperature, humidity, wind_speed, visibility_km, condition_text, severity_index)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (apt, round(random.uniform(18, 42), 1), round(random.uniform(30, 95), 1),
                     round(random.uniform(2, 25), 1), round(random.uniform(2, 15), 1),
                     random.choice(["Clear", "Cloudy", "Rain", "Haze", "Fog"]),
                     round(random.uniform(0, 0.5), 2)),
                )
        logger.info("[ETL] Weather sync complete.")
    except Exception as e:
        logger.error(f"[ETL] Weather sync failed: {e}")


def job_sentiment_processing():
    """Batch process social sentiment data."""
    from database.schema import get_connection
    logger.info("[ETL] Processing sentiment batch...")
    try:
        sim_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "simulated_data", "sentiment.csv")
        if os.path.exists(sim_path):
            df = pd.read_csv(sim_path)
            with get_connection() as conn:
                latest = conn.execute("SELECT MAX(id) FROM coll_social_sentiment").fetchone()[0] or 0
                if latest == 0:
                    for _, r in df.iterrows():
                        conn.execute(
                            "INSERT INTO coll_social_sentiment (airline,platform,post_text,sentiment_score,complaint_cat,engagement) VALUES (?,?,?,?,?,?)",
                            (r.get("airline", ""), r.get("platform", ""), "",
                             r.get("sentiment_score", 0), r.get("complaint_category", ""), r.get("engagement_score", 0)),
                        )
        logger.info("[ETL] Sentiment processing complete.")
    except Exception as e:
        logger.error(f"[ETL] Sentiment processing failed: {e}")


def job_model_retraining_check():
    """Check if models need retraining based on drift detection."""
    logger.info("[ETL] Checking model retraining requirements...")
    try:
        from ml_advanced.drift_detector import check_all_models_drift
        results = check_all_models_drift()
        for model_name, drift_info in results.items():
            if drift_info.get("drift_detected"):
                logger.warning(f"[ETL] Drift detected in {model_name} – retraining recommended")
        logger.info("[ETL] Model retraining check complete.")
    except Exception as e:
        logger.error(f"[ETL] Model retraining check failed: {e}")


# ------------------------------------------------------------------ #
#  Scheduler setup
# ------------------------------------------------------------------ #

_scheduler = None
_job_history: list[dict] = []


def _wrap_job(func: Callable, name: str) -> Callable:
    """Wrap an ETL job to record execution history."""
    def wrapper():
        start = datetime.now()
        status = "success"
        error = None
        try:
            func()
        except Exception as e:
            status = "failed"
            error = str(e)
            logger.error(f"[ETL] {name} failed: {e}")
        _job_history.append({
            "job": name,
            "started_at": start.isoformat(),
            "finished_at": datetime.now().isoformat(),
            "duration_sec": round((datetime.now() - start).total_seconds(), 2),
            "status": status,
            "error": error,
        })
        # Keep last 200 entries
        if len(_job_history) > 200:
            _job_history.pop(0)
    return wrapper


def init_scheduler():
    """Initialize and start the APScheduler with all ETL jobs."""
    global _scheduler
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger
    except ImportError:
        logger.warning("[ETL] APScheduler not installed – scheduler disabled. Run: pip install apscheduler")
        return None

    _scheduler = BackgroundScheduler(timezone="Asia/Kolkata")

    # Daily jobs (run at 2 AM IST)
    _scheduler.add_job(_wrap_job(job_daily_demand_aggregation, "demand_aggregation"),
                       CronTrigger(hour=2, minute=0), id="demand_agg", replace_existing=True)
    _scheduler.add_job(_wrap_job(job_refresh_route_revenue, "route_revenue"),
                       CronTrigger(hour=2, minute=15), id="route_rev", replace_existing=True)
    _scheduler.add_job(_wrap_job(job_refresh_load_factor, "load_factor_agg"),
                       CronTrigger(hour=2, minute=30), id="lf_agg", replace_existing=True)

    # Daily competitor refresh at 6 AM
    _scheduler.add_job(_wrap_job(job_competitor_price_refresh, "competitor_refresh"),
                       CronTrigger(hour=6, minute=0), id="comp_refresh", replace_existing=True)

    # Weather sync every 30 minutes
    _scheduler.add_job(_wrap_job(job_weather_sync, "weather_sync"),
                       IntervalTrigger(minutes=30), id="weather", replace_existing=True)

    # Sentiment processing every 6 hours
    _scheduler.add_job(_wrap_job(job_sentiment_processing, "sentiment_batch"),
                       IntervalTrigger(hours=6), id="sentiment", replace_existing=True)

    # Model retraining check daily at 3 AM
    _scheduler.add_job(_wrap_job(job_model_retraining_check, "model_retrain_check"),
                       CronTrigger(hour=3, minute=0), id="model_check", replace_existing=True)

    _scheduler.start()
    logger.info("[ETL] Scheduler started with all jobs.")
    return _scheduler


def get_scheduler_status() -> dict:
    """Return current scheduler status and job history."""
    jobs = []
    if _scheduler:
        for job in _scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": str(job.func.__name__) if hasattr(job.func, '__name__') else str(job.func),
                "next_run": str(job.next_run_time) if job.next_run_time else None,
                "trigger": str(job.trigger),
            })
    return {
        "scheduler_running": _scheduler is not None and _scheduler.running if _scheduler else False,
        "jobs": jobs,
        "recent_history": _job_history[-20:],
    }


def run_job_now(job_name: str) -> dict:
    """Manually trigger a specific ETL job."""
    job_map = {
        "demand_aggregation": job_daily_demand_aggregation,
        "route_revenue": job_refresh_route_revenue,
        "load_factor_agg": job_refresh_load_factor,
        "competitor_refresh": job_competitor_price_refresh,
        "weather_sync": job_weather_sync,
        "sentiment_batch": job_sentiment_processing,
        "model_retrain_check": job_model_retraining_check,
    }
    func = job_map.get(job_name)
    if not func:
        return {"error": f"Unknown job: {job_name}", "available": list(job_map.keys())}
    start = datetime.now()
    try:
        func()
        return {"job": job_name, "status": "success", "duration_sec": round((datetime.now() - start).total_seconds(), 2)}
    except Exception as e:
        return {"job": job_name, "status": "failed", "error": str(e)}
