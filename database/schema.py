"""
SQLite Star Schema for LNT Aviation Revenue Management.

Tables:
  Dimensions: dim_route, dim_time, dim_passenger, dim_aircraft, dim_price
  Facts:      fact_bookings, fact_flights, fact_revenue

Indexes and views included for performance.
"""

from __future__ import annotations

import sqlite3
import os
from contextlib import contextmanager
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "aviation.db")

# ------------------------------------------------------------------ #
#  Connection helper
# ------------------------------------------------------------------ #

@contextmanager
def get_connection(db_path: str = DB_PATH):
    """Yield a SQLite connection with WAL mode & foreign keys enabled."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ------------------------------------------------------------------ #
#  DDL – Star Schema
# ------------------------------------------------------------------ #

SCHEMA_DDL = """
-- ========================================================
-- DIMENSION TABLES
-- ========================================================

CREATE TABLE IF NOT EXISTS dim_route (
    route_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    origin_iata     TEXT    NOT NULL,
    dest_iata       TEXT    NOT NULL,
    origin_city     TEXT,
    dest_city       TEXT,
    distance_km     REAL,
    haul_category   TEXT    CHECK(haul_category IN ('Short','Medium','Long')),
    is_metro        INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(origin_iata, dest_iata)
);

CREATE TABLE IF NOT EXISTS dim_time (
    time_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    full_date       DATE    NOT NULL UNIQUE,
    year            INTEGER NOT NULL,
    quarter         INTEGER NOT NULL,
    month           INTEGER NOT NULL,
    month_name      TEXT    NOT NULL,
    week_of_year    INTEGER NOT NULL,
    day_of_month    INTEGER NOT NULL,
    day_of_week     INTEGER NOT NULL,
    day_name        TEXT    NOT NULL,
    is_weekend      INTEGER DEFAULT 0,
    is_holiday      INTEGER DEFAULT 0,
    holiday_name    TEXT,
    season          TEXT,
    is_peak         INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS dim_passenger (
    passenger_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name      TEXT,
    last_name       TEXT,
    age             INTEGER,
    gender          TEXT,
    nationality     TEXT,
    loyalty_tier    TEXT    CHECK(loyalty_tier IN ('Blue','Silver','Gold','Platinum')),
    total_miles     REAL    DEFAULT 0,
    lifetime_spend  REAL    DEFAULT 0,
    signup_date     DATE,
    churn_risk      REAL    DEFAULT 0,
    segment_cluster INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dim_aircraft (
    aircraft_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    registration    TEXT    UNIQUE,
    aircraft_type   TEXT,
    manufacturer    TEXT,
    model           TEXT,
    seat_capacity   INTEGER,
    range_km        REAL,
    age_years       REAL,
    status          TEXT    DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS dim_price (
    price_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    route_id        INTEGER REFERENCES dim_route(route_id),
    fare_class      TEXT    CHECK(fare_class IN ('Economy','Premium','Business')),
    base_fare       REAL    NOT NULL,
    taxes           REAL    DEFAULT 0,
    ancillary_avg   REAL    DEFAULT 0,
    effective_from  DATE    NOT NULL,
    effective_to    DATE,
    is_current      INTEGER DEFAULT 1,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ========================================================
-- FACT TABLES
-- ========================================================

CREATE TABLE IF NOT EXISTS fact_bookings (
    booking_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    route_id        INTEGER REFERENCES dim_route(route_id),
    time_id         INTEGER REFERENCES dim_time(time_id),
    passenger_id    INTEGER REFERENCES dim_passenger(passenger_id),
    price_id        INTEGER REFERENCES dim_price(price_id),
    booking_date    DATE    NOT NULL,
    departure_date  DATE    NOT NULL,
    fare_class      TEXT,
    booking_channel TEXT,
    payment_method  TEXT,
    total_price     REAL,
    discount_pct    REAL    DEFAULT 0,
    ancillary_rev   REAL    DEFAULT 0,
    passenger_count INTEGER DEFAULT 1,
    lead_time_days  INTEGER,
    booking_status  TEXT    DEFAULT 'Confirmed',
    cancellation_date DATE,
    is_no_show      INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_flights (
    flight_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    route_id        INTEGER REFERENCES dim_route(route_id),
    time_id         INTEGER REFERENCES dim_time(time_id),
    aircraft_id     INTEGER REFERENCES dim_aircraft(aircraft_id),
    flight_number   TEXT,
    scheduled_dep   TIMESTAMP,
    actual_dep      TIMESTAMP,
    scheduled_arr   TIMESTAMP,
    actual_arr      TIMESTAMP,
    delay_minutes   REAL    DEFAULT 0,
    is_cancelled    INTEGER DEFAULT 0,
    cancel_reason   TEXT,
    seat_capacity   INTEGER,
    pax_booked      INTEGER,
    pax_actual      INTEGER,
    load_factor     REAL,
    crew_delay      INTEGER DEFAULT 0,
    tech_issue      INTEGER DEFAULT 0,
    weather_issue   INTEGER DEFAULT 0,
    turnaround_min  REAL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_revenue (
    revenue_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    route_id        INTEGER REFERENCES dim_route(route_id),
    time_id         INTEGER REFERENCES dim_time(time_id),
    revenue_date    DATE    NOT NULL,
    ticket_revenue  REAL    DEFAULT 0,
    ancillary_rev   REAL    DEFAULT 0,
    total_revenue   REAL    DEFAULT 0,
    operating_cost  REAL    DEFAULT 0,
    fuel_cost       REAL    DEFAULT 0,
    profit          REAL    DEFAULT 0,
    rasm            REAL,
    yield_per_pax   REAL,
    pax_count       INTEGER DEFAULT 0,
    flights_count   INTEGER DEFAULT 0,
    avg_load_factor REAL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(route_id, revenue_date)
);

-- ========================================================
-- BOOKING VELOCITY TIME-SERIES (15 min buckets)
-- ========================================================

CREATE TABLE IF NOT EXISTS ts_booking_velocity (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    route_id        INTEGER REFERENCES dim_route(route_id),
    bucket_time     TIMESTAMP NOT NULL,
    departure_date  DATE      NOT NULL,
    bookings_count  INTEGER   DEFAULT 0,
    cancellations   INTEGER   DEFAULT 0,
    revenue         REAL      DEFAULT 0,
    cumulative_lf   REAL,
    UNIQUE(route_id, bucket_time, departure_date)
);

-- ========================================================
-- MONGODB-STYLE COLLECTIONS (JSON storage in SQLite)
-- ========================================================

CREATE TABLE IF NOT EXISTS coll_competitor_prices (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    route           TEXT    NOT NULL,
    competitor      TEXT    NOT NULL,
    price           REAL,
    fare_class      TEXT,
    scraped_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source          TEXT
);

CREATE TABLE IF NOT EXISTS coll_social_sentiment (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    airline         TEXT,
    platform        TEXT,
    post_text       TEXT,
    sentiment_score REAL,
    complaint_cat   TEXT,
    engagement      INTEGER DEFAULT 0,
    posted_at       TIMESTAMP,
    ingested_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS coll_events_calendar (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_name      TEXT    NOT NULL,
    event_type      TEXT,
    start_date      DATE,
    end_date        DATE,
    region          TEXT,
    demand_mult     REAL    DEFAULT 1.0,
    source          TEXT
);

CREATE TABLE IF NOT EXISTS coll_search_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    search_query    TEXT,
    origin          TEXT,
    destination     TEXT,
    travel_date     DATE,
    pax_count       INTEGER DEFAULT 1,
    fare_class      TEXT,
    channel         TEXT,
    session_id      TEXT,
    searched_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    converted       INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS coll_operational_notes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    flight_number   TEXT,
    route           TEXT,
    note_type       TEXT,
    note_text       TEXT,
    author          TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    severity        TEXT    DEFAULT 'info'
);

CREATE TABLE IF NOT EXISTS coll_weather_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    airport_iata    TEXT    NOT NULL,
    temperature     REAL,
    humidity        REAL,
    wind_speed      REAL,
    visibility_km   REAL,
    condition_text  TEXT,
    severity_index  REAL   DEFAULT 0,
    snapshot_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ========================================================
-- AUDIT LOG
-- ========================================================

CREATE TABLE IF NOT EXISTS audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    action          TEXT    NOT NULL,
    entity_type     TEXT,
    entity_id       TEXT,
    old_value       TEXT,
    new_value       TEXT,
    user_id         TEXT    DEFAULT 'system',
    ip_address      TEXT,
    details         TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ========================================================
-- MODEL REGISTRY
-- ========================================================

CREATE TABLE IF NOT EXISTS model_registry (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT    NOT NULL,
    version         TEXT    NOT NULL,
    algorithm       TEXT,
    metrics         TEXT,
    parameters      TEXT,
    features        TEXT,
    artifact_path   TEXT,
    status          TEXT    DEFAULT 'active',
    trained_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    promoted_at     TIMESTAMP,
    retired_at      TIMESTAMP,
    UNIQUE(model_name, version)
);

-- ========================================================
-- INDEXES (Performance)
-- ========================================================

CREATE INDEX IF NOT EXISTS idx_fb_route_dep   ON fact_bookings(route_id, departure_date);
CREATE INDEX IF NOT EXISTS idx_fb_booking_dt  ON fact_bookings(booking_date);
CREATE INDEX IF NOT EXISTS idx_fb_status      ON fact_bookings(booking_status);
CREATE INDEX IF NOT EXISTS idx_ff_route_time  ON fact_flights(route_id, time_id);
CREATE INDEX IF NOT EXISTS idx_ff_flight_no   ON fact_flights(flight_number);
CREATE INDEX IF NOT EXISTS idx_fr_route_date  ON fact_revenue(route_id, revenue_date);
CREATE INDEX IF NOT EXISTS idx_bv_route_dep   ON ts_booking_velocity(route_id, departure_date);
CREATE INDEX IF NOT EXISTS idx_bv_bucket      ON ts_booking_velocity(bucket_time);
CREATE INDEX IF NOT EXISTS idx_comp_route     ON coll_competitor_prices(route, scraped_at);
CREATE INDEX IF NOT EXISTS idx_search_date    ON coll_search_logs(searched_at);
CREATE INDEX IF NOT EXISTS idx_weather_apt    ON coll_weather_snapshots(airport_iata, snapshot_at);
CREATE INDEX IF NOT EXISTS idx_audit_entity   ON audit_log(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_model_name     ON model_registry(model_name, status);

-- ========================================================
-- MATERIALIZED VIEWS (as regular tables, refreshed by ETL)
-- ========================================================

CREATE TABLE IF NOT EXISTS mv_route_revenue (
    route_id        INTEGER,
    route           TEXT,
    period          TEXT,
    total_revenue   REAL,
    ticket_revenue  REAL,
    ancillary_rev   REAL,
    avg_fare        REAL,
    pax_count       INTEGER,
    refreshed_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mv_load_factor_agg (
    route_id        INTEGER,
    route           TEXT,
    period          TEXT,
    avg_load_factor REAL,
    min_load_factor REAL,
    max_load_factor REAL,
    flights_count   INTEGER,
    below_target    INTEGER DEFAULT 0,
    refreshed_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mv_daily_booking_pace (
    route_id        INTEGER,
    departure_date  DATE,
    days_out        INTEGER,
    cumulative_bkgs INTEGER,
    pace_vs_hist    REAL,
    forecast_bkgs   INTEGER,
    refreshed_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def init_database(db_path: str = DB_PATH) -> None:
    """Create all tables, indexes, and views."""
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_DDL)
    print(f"[DB] Star schema initialized at {db_path}")


def query(sql: str, params: tuple = (), db_path: str = DB_PATH) -> list[dict]:
    """Execute a SELECT and return list of dicts."""
    with get_connection(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def execute(sql: str, params: tuple = (), db_path: str = DB_PATH) -> int:
    """Execute INSERT/UPDATE/DELETE, return lastrowid."""
    with get_connection(db_path) as conn:
        cur = conn.execute(sql, params)
        return cur.lastrowid


def execute_many(sql: str, data: list[tuple], db_path: str = DB_PATH) -> int:
    """Execute batch INSERT."""
    with get_connection(db_path) as conn:
        conn.executemany(sql, data)
        return len(data)


if __name__ == "__main__":
    init_database()
    print("[DB] Schema created successfully.")
