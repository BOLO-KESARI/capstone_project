"""
Migrate CSV/JSON simulation data into the SQLite star schema.

Reads from:
  simulated_data/   → bookings, flights, passengers, operations, etc.
  db_simulation/    → fact_bookings, dim_flights, route_performance, passenger_intelligence
  new_models/       → ML datasets

Populates all dimension and fact tables.
"""

from __future__ import annotations

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from database.schema import get_connection, init_database, DB_PATH

BASE = os.path.dirname(os.path.dirname(__file__))
SIM = os.path.join(BASE, "simulated_data")
DB_SIM = os.path.join(BASE, "db_simulation")


def _safe_read(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ------------------------------------------------------------------ #
#  Dimension loaders
# ------------------------------------------------------------------ #

def migrate_dim_route(conn):
    """Populate dim_route from flights/bookings data."""
    routes = set()
    for fname in ["flights.csv", "bookings.csv", "advanced_features.csv"]:
        df = _safe_read(os.path.join(SIM, fname))
        if df is not None and "route" in df.columns:
            routes.update(df["route"].dropna().unique())

    CITY_MAP = {
        "DEL": "Delhi", "BOM": "Mumbai", "BLR": "Bengaluru", "MAA": "Chennai",
        "HYD": "Hyderabad", "CCU": "Kolkata", "AMD": "Ahmedabad", "COK": "Kochi",
        "PNQ": "Pune", "GOI": "Goa", "JAI": "Jaipur", "LKO": "Lucknow",
        "GAU": "Guwahati", "NAG": "Nagpur", "PAT": "Patna", "TRV": "Thiruvananthapuram",
        "BBI": "Bhubaneswar", "IXC": "Chandigarh", "ATQ": "Amritsar", "IDR": "Indore",
        "VTZ": "Vishakhapatnam", "SXR": "Srinagar", "IXJ": "Jammu", "GOX": "Goa Mopa",
    }
    DIST = {
        "DEL-BOM": 1150, "DEL-BLR": 1740, "DEL-CCU": 1300, "DEL-MAA": 1760,
        "DEL-AMD": 780, "DEL-JAI": 240, "DEL-NAG": 850, "DEL-GAU": 1450,
        "BOM-BLR": 840, "BOM-MAA": 1030, "BOM-HYD": 620, "BOM-CCU": 1960,
        "BOM-GOI": 440, "BOM-AMD": 440, "BOM-NAG": 700,
        "BLR-MAA": 290, "BLR-HYD": 500, "BLR-COK": 365, "BLR-CCU": 1560,
        "MAA-HYD": 520, "PNQ-NAG": 620, "PNQ-DEL": 1170, "PNQ-BLR": 735,
        "PNQ-HYD": 500, "NAG-BLR": 1080, "GAU-CCU": 500,
    }

    for route_str in sorted(routes):
        parts = route_str.split("-")
        if len(parts) != 2:
            continue
        orig, dest = parts
        dist = DIST.get(route_str, DIST.get(f"{dest}-{orig}", 600))
        haul = "Short" if dist < 500 else ("Medium" if dist < 1200 else "Long")
        metros = {"DEL", "BOM", "BLR", "MAA", "HYD", "CCU"}
        is_metro = 1 if (orig in metros and dest in metros) else 0
        try:
            conn.execute(
                """INSERT OR IGNORE INTO dim_route
                   (origin_iata, dest_iata, origin_city, dest_city, distance_km, haul_category, is_metro)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (orig, dest, CITY_MAP.get(orig, orig), CITY_MAP.get(dest, dest), dist, haul, is_metro),
            )
        except Exception:
            pass
    print(f"  [dim_route] loaded {len(routes)} routes")


def migrate_dim_time(conn):
    """Populate dim_time with 3 years of dates (2023-2026)."""
    start = datetime(2023, 1, 1)
    end = datetime(2026, 12, 31)
    holidays = {
        "01-26": "Republic Day", "03-25": "Holi", "04-14": "Ambedkar Jayanti",
        "08-15": "Independence Day", "10-02": "Gandhi Jayanti", "10-24": "Diwali",
        "11-01": "Diwali", "12-25": "Christmas", "01-01": "New Year",
    }
    d = start
    rows = []
    while d <= end:
        md = d.strftime("%m-%d")
        h_name = holidays.get(md)
        season = {1: "Winter", 2: "Winter", 3: "Spring", 4: "Summer", 5: "Summer",
                  6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
                  10: "Autumn", 11: "Autumn", 12: "Winter"}[d.month]
        is_peak = 1 if d.month in (3, 4, 5, 10, 11, 12) else 0
        rows.append((
            d.date().isoformat(), d.year, (d.month - 1) // 3 + 1, d.month,
            d.strftime("%B"), d.isocalendar()[1], d.day, d.weekday(),
            d.strftime("%A"), 1 if d.weekday() >= 5 else 0,
            1 if h_name else 0, h_name, season, is_peak,
        ))
        d += timedelta(days=1)
    conn.executemany(
        """INSERT OR IGNORE INTO dim_time
           (full_date,year,quarter,month,month_name,week_of_year,day_of_month,
            day_of_week,day_name,is_weekend,is_holiday,holiday_name,season,is_peak)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows
    )
    print(f"  [dim_time] loaded {len(rows)} date records")


def migrate_dim_passenger(conn):
    """Populate dim_passenger from passengers.csv."""
    df = _safe_read(os.path.join(SIM, "passengers.csv"))
    if df is None:
        print("  [dim_passenger] skipped – file not found")
        return
    cols_map = {
        "first_name": "first_name", "last_name": "last_name", "age": "age",
        "gender": "gender", "nationality": "nationality", "loyalty_tier": "loyalty_tier",
        "total_miles_flown": "total_miles", "lifetime_spend": "lifetime_spend",
        "churn_risk_score": "churn_risk",
    }
    for _, row in df.iterrows():
        vals = {v: row.get(k) for k, v in cols_map.items()}
        try:
            conn.execute(
                """INSERT INTO dim_passenger
                   (first_name,last_name,age,gender,nationality,loyalty_tier,total_miles,lifetime_spend,churn_risk)
                   VALUES (:first_name,:last_name,:age,:gender,:nationality,:loyalty_tier,:total_miles,:lifetime_spend,:churn_risk)""",
                vals,
            )
        except Exception:
            pass
    print(f"  [dim_passenger] loaded {len(df)} passengers")


def migrate_dim_aircraft(conn):
    """Generate aircraft dimension."""
    types = [
        ("VT-LNT001", "Airbus", "A320neo", 180, 6300),
        ("VT-LNT002", "Airbus", "A321neo", 220, 7400),
        ("VT-LNT003", "Boeing", "737 MAX 8", 189, 6570),
        ("VT-LNT004", "Boeing", "787-8", 296, 13620),
        ("VT-LNT005", "Airbus", "A320neo", 180, 6300),
        ("VT-LNT006", "ATR", "72-600", 72, 1528),
        ("VT-LNT007", "Airbus", "A321neo", 220, 7400),
        ("VT-LNT008", "Boeing", "737 MAX 8", 189, 6570),
    ]
    for reg, mfr, model, seats, rng in types:
        conn.execute(
            """INSERT OR IGNORE INTO dim_aircraft
               (registration, aircraft_type, manufacturer, model, seat_capacity, range_km, age_years)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (reg, f"{mfr} {model}", mfr, model, seats, rng, round(np.random.uniform(1, 12), 1)),
        )
    print(f"  [dim_aircraft] loaded {len(types)} aircraft")


def migrate_dim_price(conn):
    """Generate price dimension from route + fare classes."""
    routes = conn.execute("SELECT route_id, origin_iata, dest_iata, distance_km FROM dim_route").fetchall()
    count = 0
    for r in routes:
        dist = r[3] or 600
        base_eco = max(2000, dist * 4.5)
        for fc, mult in [("Economy", 1.0), ("Premium", 1.6), ("Business", 2.8)]:
            fare = round(base_eco * mult)
            tax = round(fare * 0.05)
            anc = round(fare * 0.08) if fc == "Economy" else round(fare * 0.12)
            conn.execute(
                """INSERT OR IGNORE INTO dim_price
                   (route_id, fare_class, base_fare, taxes, ancillary_avg, effective_from, is_current)
                   VALUES (?, ?, ?, ?, ?, '2023-01-01', 1)""",
                (r[0], fc, fare, tax, anc),
            )
            count += 1
    print(f"  [dim_price] loaded {count} price records")


# ------------------------------------------------------------------ #
#  Fact loaders
# ------------------------------------------------------------------ #

def migrate_fact_bookings(conn):
    """Load fact_bookings from simulated bookings.csv + db_simulation."""
    df = _safe_read(os.path.join(SIM, "bookings.csv"))
    if df is None:
        df = _safe_read(os.path.join(DB_SIM, "rdbms", "fact_bookings.csv"))
    if df is None:
        print("  [fact_bookings] skipped – no data")
        return

    # Build route lookup
    route_rows = conn.execute("SELECT route_id, origin_iata || '-' || dest_iata as route FROM dim_route").fetchall()
    route_map = {r[1]: r[0] for r in route_rows}

    # Build time lookup
    time_rows = conn.execute("SELECT time_id, full_date FROM dim_time").fetchall()
    time_map = {r[1]: r[0] for r in time_rows}

    count = 0
    for _, row in df.iterrows():
        route_str = row.get("route", "")
        rid = route_map.get(route_str)
        bdate = str(row.get("booking_date", ""))[:10]
        ddate = str(row.get("departure_date", ""))[:10]
        tid = time_map.get(ddate)
        try:
            conn.execute(
                """INSERT INTO fact_bookings
                   (route_id, time_id, booking_date, departure_date, fare_class,
                    booking_channel, payment_method, total_price, discount_pct,
                    ancillary_rev, passenger_count, lead_time_days, booking_status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (rid, tid, bdate, ddate,
                 row.get("fare_class", "Economy"),
                 row.get("booking_channel", "Web"),
                 row.get("payment_method", "UPI"),
                 row.get("total_price_paid") or row.get("total_price", 0),
                 row.get("discount_applied", 0),
                 row.get("ancillary_revenue", 0),
                 row.get("passenger_count", 1),
                 row.get("lead_time_days") or row.get("days_before_departure", 0),
                 row.get("booking_status", "Confirmed")),
            )
            count += 1
        except Exception:
            pass
    print(f"  [fact_bookings] loaded {count} bookings")


def migrate_fact_flights(conn):
    """Load fact_flights from flights.csv + operations.csv."""
    df = _safe_read(os.path.join(SIM, "flights.csv"))
    if df is None:
        print("  [fact_flights] skipped – no data")
        return

    ops = _safe_read(os.path.join(SIM, "operations.csv"))
    ops_map = {}
    if ops is not None and "route" in ops.columns:
        for _, r in ops.iterrows():
            ops_map[r.get("route", "")] = r

    route_rows = conn.execute("SELECT route_id, origin_iata || '-' || dest_iata as route FROM dim_route").fetchall()
    route_map = {r[1]: r[0] for r in route_rows}

    count = 0
    for _, row in df.iterrows():
        route_str = row.get("route", "")
        rid = route_map.get(route_str)
        op = ops_map.get(route_str, {})
        try:
            conn.execute(
                """INSERT INTO fact_flights
                   (route_id, flight_number, scheduled_dep, actual_dep,
                    delay_minutes, is_cancelled, seat_capacity, pax_booked,
                    load_factor, crew_delay, tech_issue, weather_issue, turnaround_min)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (rid,
                 row.get("flight_number", ""),
                 str(row.get("scheduled_departure", ""))[:19],
                 str(row.get("actual_departure", ""))[:19],
                 row.get("delay_minutes", 0),
                 1 if row.get("flight_status") == "Cancelled" else 0,
                 row.get("seat_capacity", 180),
                 row.get("passengers_booked", 0),
                 row.get("load_factor", 0),
                 1 if op.get("crew_availability", 1) < 0.8 else 0 if isinstance(op, dict) and op else 0,
                 0, 0,
                 row.get("turnaround_time", 45)),
            )
            count += 1
        except Exception:
            pass
    print(f"  [fact_flights] loaded {count} flights")


def migrate_fact_revenue(conn):
    """Aggregate revenue from bookings + flights."""
    conn.execute("""
        INSERT OR IGNORE INTO fact_revenue
            (route_id, revenue_date, ticket_revenue, ancillary_rev, total_revenue, pax_count, flights_count, avg_load_factor)
        SELECT
            fb.route_id,
            fb.departure_date,
            SUM(fb.total_price),
            SUM(fb.ancillary_rev),
            SUM(fb.total_price) + SUM(fb.ancillary_rev),
            SUM(fb.passenger_count),
            COUNT(DISTINCT fb.booking_id),
            0
        FROM fact_bookings fb
        WHERE fb.route_id IS NOT NULL AND fb.departure_date IS NOT NULL
        GROUP BY fb.route_id, fb.departure_date
    """)
    cnt = conn.execute("SELECT COUNT(*) FROM fact_revenue").fetchone()[0]
    print(f"  [fact_revenue] aggregated {cnt} revenue records")


def migrate_collections(conn):
    """Load MongoDB-style collections from CSVs."""
    # Competitor prices
    df = _safe_read(os.path.join(SIM, "competitor_prices.csv"))
    if df is not None:
        for _, r in df.iterrows():
            conn.execute(
                "INSERT INTO coll_competitor_prices (route, competitor, price, fare_class, source) VALUES (?,?,?,?,?)",
                (r.get("route", ""), r.get("competitor_airline", ""), r.get("competitor_price", 0),
                 r.get("fare_class", "Economy"), "csv_import"),
            )
        print(f"  [coll_competitor_prices] loaded {len(df)}")

    # Social sentiment
    df = _safe_read(os.path.join(SIM, "sentiment.csv"))
    if df is not None:
        for _, r in df.iterrows():
            conn.execute(
                "INSERT INTO coll_social_sentiment (airline,platform,post_text,sentiment_score,complaint_cat,engagement) VALUES (?,?,?,?,?,?)",
                (r.get("airline", ""), r.get("platform", ""), r.get("post_text", ""),
                 r.get("sentiment_score", 0), r.get("complaint_category", ""), r.get("engagement_score", 0)),
            )
        print(f"  [coll_social_sentiment] loaded {len(df)}")

    # Events
    df = _safe_read(os.path.join(SIM, "events.csv"))
    if df is not None:
        for _, r in df.iterrows():
            conn.execute(
                "INSERT INTO coll_events_calendar (event_name, event_type, start_date, region, demand_mult) VALUES (?,?,?,?,?)",
                (r.get("event_name", ""), r.get("event_type", ""), r.get("event_date", ""),
                 r.get("region", ""), r.get("travel_demand_multiplier", 1.0)),
            )
        print(f"  [coll_events_calendar] loaded {len(df)}")

    print("  [collections] migration complete")


# ------------------------------------------------------------------ #
#  Master migration
# ------------------------------------------------------------------ #

def run_full_migration(db_path: str = DB_PATH):
    """Run complete migration from CSV → SQLite."""
    print("=" * 60)
    print("  LNT Aviation – Data Migration")
    print("=" * 60)
    init_database(db_path)

    with get_connection(db_path) as conn:
        migrate_dim_route(conn)
        migrate_dim_time(conn)
        migrate_dim_passenger(conn)
        migrate_dim_aircraft(conn)
        migrate_dim_price(conn)
        migrate_fact_bookings(conn)
        migrate_fact_flights(conn)
        migrate_fact_revenue(conn)
        migrate_collections(conn)

    print("=" * 60)
    print("  Migration complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_full_migration()
