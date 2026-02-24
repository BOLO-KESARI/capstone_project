"""
Data Quality Rules Engine for LNT Aviation.

Validates, cleanses, and standardises incoming data before it enters
the star schema or feeds ML models.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
#  IATA Code Reference
# ------------------------------------------------------------------ #

VALID_IATA = {
    "DEL", "BOM", "BLR", "MAA", "HYD", "CCU", "AMD", "COK", "PNQ", "GOI",
    "GOX", "LKO", "GAU", "JAI", "TRV", "BBI", "PAT", "IXC", "IXJ", "SXR",
    "ATQ", "IDR", "NAG", "VTZ",
}

VALID_FARE_CLASSES = {"Economy", "Premium", "Business"}
VALID_BOOKING_CHANNELS = {"Web", "Mobile App", "OTA", "Call Center", "Corporate", "Travel Agent"}
VALID_PAYMENT_METHODS = {"UPI", "Credit Card", "Debit Card", "Wallet", "Net Banking", "Cash"}
VALID_BOOKING_STATUS = {"Confirmed", "Cancelled", "No-Show", "Waitlisted"}


class DQReport:
    """Holds results of a data quality check run."""

    def __init__(self):
        self.checks: list[dict] = []
        self.rows_cleaned = 0
        self.rows_rejected = 0

    def add(self, rule: str, field: str, severity: str, count: int, action: str):
        self.checks.append({
            "rule": rule, "field": field, "severity": severity,
            "affected_rows": count, "action": action,
            "timestamp": datetime.now().isoformat(),
        })

    def summary(self) -> dict:
        return {
            "total_checks": len(self.checks),
            "rows_cleaned": self.rows_cleaned,
            "rows_rejected": self.rows_rejected,
            "issues": self.checks,
        }


# ------------------------------------------------------------------ #
#  Rule Functions
# ------------------------------------------------------------------ #

def standardise_iata_codes(df: pd.DataFrame, columns: list[str], report: DQReport) -> pd.DataFrame:
    """Uppercase IATA codes and flag invalid ones."""
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).str.strip().str.upper()
        invalid = ~df[col].isin(VALID_IATA) & (df[col] != "NAN") & (df[col] != "")
        n_invalid = invalid.sum()
        if n_invalid:
            report.add("IATA_STANDARDISE", col, "warning", int(n_invalid),
                       f"Flagged {n_invalid} non-standard IATA codes")
    return df


def remove_negative_prices(df: pd.DataFrame, price_cols: list[str], report: DQReport) -> pd.DataFrame:
    """Remove rows with negative ticket prices."""
    for col in price_cols:
        if col not in df.columns:
            continue
        mask = pd.to_numeric(df[col], errors="coerce") < 0
        n_neg = mask.sum()
        if n_neg:
            df = df[~mask].copy()
            report.add("NEGATIVE_PRICE", col, "critical", int(n_neg),
                       f"Removed {n_neg} rows with negative {col}")
            report.rows_rejected += int(n_neg)
    return df


def remove_impossible_dates(df: pd.DataFrame, report: DQReport) -> pd.DataFrame:
    """Remove bookings where booking_date > departure_date."""
    if "booking_date" in df.columns and "departure_date" in df.columns:
        bd = pd.to_datetime(df["booking_date"], errors="coerce")
        dd = pd.to_datetime(df["departure_date"], errors="coerce")
        impossible = bd > dd
        n = impossible.sum()
        if n:
            df = df[~impossible].copy()
            report.add("IMPOSSIBLE_DATE", "booking_date > departure_date", "critical",
                       int(n), f"Removed {n} rows with booking after departure")
            report.rows_rejected += int(n)
    return df


def impute_missing_competitor_prices(df: pd.DataFrame, report: DQReport) -> pd.DataFrame:
    """Fill missing competitor prices with last known value per route."""
    col = "competitor_price"
    if col not in df.columns:
        return df
    missing_before = df[col].isna().sum()
    if "route" in df.columns:
        df[col] = df.groupby("route")[col].transform(lambda s: s.fillna(method="ffill").fillna(method="bfill"))
    else:
        df[col] = df[col].fillna(method="ffill").fillna(method="bfill")
    df[col] = df[col].fillna(df[col].median())
    filled = missing_before - df[col].isna().sum()
    if filled:
        report.add("IMPUTE_COMP_PRICE", col, "info", int(filled),
                    f"Forward-filled {filled} missing competitor prices")
        report.rows_cleaned += int(filled)
    return df


def detect_booking_spikes(df: pd.DataFrame, report: DQReport, threshold: float = 3.0) -> pd.DataFrame:
    """Flag outlier days with booking counts > threshold * std above mean."""
    if "booking_date" not in df.columns:
        return df
    daily = df.groupby("booking_date").size()
    mean, std = daily.mean(), daily.std()
    if std == 0:
        return df
    outlier_dates = daily[daily > mean + threshold * std].index.tolist()
    if outlier_dates:
        report.add("BOOKING_SPIKE", "booking_date", "warning", len(outlier_dates),
                    f"Detected {len(outlier_dates)} spike days (>{threshold}σ): {outlier_dates[:5]}")
    return df


def reconcile_timezones(df: pd.DataFrame, time_cols: list[str], target_tz: str = "Asia/Kolkata",
                        report: DQReport = None) -> pd.DataFrame:
    """Convert datetime columns to IST (Asia/Kolkata)."""
    for col in time_cols:
        if col not in df.columns:
            continue
        ts = pd.to_datetime(df[col], errors="coerce", utc=True)
        n_converted = ts.notna().sum()
        df[col] = ts.dt.tz_convert(target_tz).dt.tz_localize(None)
        if report and n_converted:
            report.add("TZ_RECONCILE", col, "info", int(n_converted),
                       f"Converted {n_converted} timestamps to {target_tz}")
            report.rows_cleaned += int(n_converted)
    return df


def handle_delayed_feeds(df: pd.DataFrame, max_delay_hours: int = 48, report: DQReport = None) -> pd.DataFrame:
    """Flag records where ingested_at - event_time > max_delay_hours."""
    if "ingested_at" in df.columns and "event_time" in df.columns:
        ing = pd.to_datetime(df["ingested_at"], errors="coerce")
        evt = pd.to_datetime(df["event_time"], errors="coerce")
        delay = (ing - evt).dt.total_seconds() / 3600
        late = delay > max_delay_hours
        n = late.sum()
        if n and report:
            report.add("DELAYED_FEED", "ingested_at", "warning", int(n),
                       f"{n} records arrived >{max_delay_hours}h late")
    return df


def backfill_weather(df: pd.DataFrame, nearby_map: dict[str, str], report: DQReport) -> pd.DataFrame:
    """Backfill missing weather from a nearby airport."""
    if "airport_iata" not in df.columns or "temperature" not in df.columns:
        return df
    missing = df["temperature"].isna()
    filled = 0
    for idx in df[missing].index:
        apt = df.at[idx, "airport_iata"]
        nearby = nearby_map.get(apt)
        if nearby:
            donor = df[(df["airport_iata"] == nearby) & df["temperature"].notna()]
            if not donor.empty:
                df.at[idx, "temperature"] = donor["temperature"].iloc[-1]
                filled += 1
    if filled:
        report.add("WEATHER_BACKFILL", "temperature", "info", filled,
                    f"Backfilled {filled} weather records from nearby airports")
        report.rows_cleaned += filled
    return df


def validate_fare_class(df: pd.DataFrame, report: DQReport) -> pd.DataFrame:
    """Standardise fare_class values."""
    if "fare_class" not in df.columns:
        return df
    mapping = {"eco": "Economy", "economy": "Economy", "prem": "Premium",
               "premium": "Premium", "biz": "Business", "business": "Business"}
    df["fare_class"] = df["fare_class"].str.strip().str.title()
    df["fare_class"] = df["fare_class"].replace(mapping)
    invalid = ~df["fare_class"].isin(VALID_FARE_CLASSES)
    n = invalid.sum()
    if n:
        df.loc[invalid, "fare_class"] = "Economy"
        report.add("FARE_CLASS_CLEAN", "fare_class", "info", int(n),
                    f"Corrected {n} invalid fare classes to 'Economy'")
        report.rows_cleaned += int(n)
    return df


# ------------------------------------------------------------------ #
#  Master pipeline
# ------------------------------------------------------------------ #

def run_quality_checks(df: pd.DataFrame, dataset_type: str = "bookings") -> tuple[pd.DataFrame, DQReport]:
    """
    Run full data quality pipeline on a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    dataset_type : 'bookings' | 'flights' | 'competitor' | 'weather'

    Returns
    -------
    (cleaned_df, DQReport)
    """
    report = DQReport()
    original_len = len(df)

    if dataset_type == "bookings":
        df = standardise_iata_codes(df, ["dep_iata", "arr_iata", "origin", "destination"], report)
        df = remove_negative_prices(df, ["total_price_paid", "total_price", "base_fare"], report)
        df = remove_impossible_dates(df, report)
        df = validate_fare_class(df, report)
        df = detect_booking_spikes(df, report)

    elif dataset_type == "competitor":
        df = standardise_iata_codes(df, ["dep_iata", "arr_iata"], report)
        df = remove_negative_prices(df, ["competitor_price", "price"], report)
        df = impute_missing_competitor_prices(df, report)

    elif dataset_type == "flights":
        df = standardise_iata_codes(df, ["dep_iata", "arr_iata"], report)

    elif dataset_type == "weather":
        df = standardise_iata_codes(df, ["airport_iata"], report)

    report.add("SUMMARY", "*", "info", original_len - len(df),
               f"Pipeline complete: {original_len} → {len(df)} rows")
    return df, report
