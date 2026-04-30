from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from flightrisk.config import DATE_COLUMN, TARGET_NAME, WEATHER_COLUMNS


@dataclass
class AggregateStats:
    airline_cancel_rate: dict[str, float]
    airline_long_delay_rate: dict[str, float]
    route_cancel_rate: dict[str, float]
    route_long_delay_rate: dict[str, float]
    origin_disruption_rate: dict[str, float]
    dest_disruption_rate: dict[str, float]
    route_month_delay_rate: dict[str, float]
    route_month_cancel_rate: dict[str, float]
    defaults: dict[str, float]


def _route_key(origin: pd.Series, dest: pd.Series) -> pd.Series:
    return origin.fillna("UNK").astype(str) + "->" + dest.fillna("UNK").astype(str)


def _route_month_key(route_key: pd.Series, month: pd.Series) -> pd.Series:
    return route_key.astype(str) + "|m" + month.astype(str)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    frame["month"] = frame[DATE_COLUMN].dt.month.astype("Int64").astype(str)
    frame["day_of_week"] = frame[DATE_COLUMN].dt.dayofweek.astype("Int64").astype(str)
    frame["is_weekend"] = frame[DATE_COLUMN].dt.dayofweek.isin([5, 6]).astype(int).astype(str)
    frame["is_holiday_window"] = (
        ((frame[DATE_COLUMN].dt.month == 11) & (frame[DATE_COLUMN].dt.day >= 20))
        | ((frame[DATE_COLUMN].dt.month == 12) & (frame[DATE_COLUMN].dt.day >= 15))
        | ((frame[DATE_COLUMN].dt.month == 1) & (frame[DATE_COLUMN].dt.day <= 5))
        | ((frame[DATE_COLUMN].dt.month == 7) & (frame[DATE_COLUMN].dt.day <= 7))
    ).astype(int).astype(str)

    crs_dep = pd.to_numeric(frame["CRS_DEP_TIME"], errors="coerce").fillna(0)
    hhmm = crs_dep.astype(int).clip(lower=0)
    frame["dep_hour"] = (hhmm // 100).clip(lower=0, upper=23)
    frame["dep_minute"] = (hhmm % 100).clip(lower=0, upper=59)
    frame["dep_time_bucket"] = pd.cut(
        frame["dep_hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["overnight", "morning", "afternoon", "evening"],
    ).astype(str)

    frame["route_key"] = _route_key(frame["ORIGIN"], frame["DEST"])
    frame["route_month_key"] = _route_month_key(frame["route_key"], frame["month"])
    return frame


def build_aggregate_stats(train_df: pd.DataFrame) -> AggregateStats:
    frame = add_time_features(train_df)
    disruption = frame[TARGET_NAME].isin(["short_delay", "long_delay", "cancelled"]).astype(float)
    cancelled = (frame[TARGET_NAME] == "cancelled").astype(float)
    long_delay = (frame[TARGET_NAME] == "long_delay").astype(float)

    airline_cancel_rate = cancelled.groupby(frame["AIRLINE_CODE"]).mean().to_dict()
    airline_long_delay_rate = long_delay.groupby(frame["AIRLINE_CODE"]).mean().to_dict()
    route_cancel_rate = cancelled.groupby(frame["route_key"]).mean().to_dict()
    route_long_delay_rate = long_delay.groupby(frame["route_key"]).mean().to_dict()
    origin_disruption_rate = disruption.groupby(frame["ORIGIN"]).mean().to_dict()
    dest_disruption_rate = disruption.groupby(frame["DEST"]).mean().to_dict()
    route_month_delay_rate = long_delay.groupby(frame["route_month_key"]).mean().to_dict()
    route_month_cancel_rate = cancelled.groupby(frame["route_month_key"]).mean().to_dict()

    defaults = {
        "airline_cancel_rate": float(cancelled.mean()),
        "airline_long_delay_rate": float(long_delay.mean()),
        "route_cancel_rate": float(cancelled.mean()),
        "route_long_delay_rate": float(long_delay.mean()),
        "origin_disruption_rate": float(disruption.mean()),
        "dest_disruption_rate": float(disruption.mean()),
        "route_month_delay_rate": float(long_delay.mean()),
        "route_month_cancel_rate": float(cancelled.mean()),
    }

    return AggregateStats(
        airline_cancel_rate=airline_cancel_rate,
        airline_long_delay_rate=airline_long_delay_rate,
        route_cancel_rate=route_cancel_rate,
        route_long_delay_rate=route_long_delay_rate,
        origin_disruption_rate=origin_disruption_rate,
        dest_disruption_rate=dest_disruption_rate,
        route_month_delay_rate=route_month_delay_rate,
        route_month_cancel_rate=route_month_cancel_rate,
        defaults=defaults,
    )


def apply_aggregate_stats(df: pd.DataFrame, stats: AggregateStats) -> pd.DataFrame:
    frame = add_time_features(df)
    frame["airline_cancel_rate"] = frame["AIRLINE_CODE"].map(stats.airline_cancel_rate).fillna(
        stats.defaults["airline_cancel_rate"]
    )
    frame["airline_long_delay_rate"] = frame["AIRLINE_CODE"].map(stats.airline_long_delay_rate).fillna(
        stats.defaults["airline_long_delay_rate"]
    )
    frame["route_cancel_rate"] = frame["route_key"].map(stats.route_cancel_rate).fillna(
        stats.defaults["route_cancel_rate"]
    )
    frame["route_long_delay_rate"] = frame["route_key"].map(stats.route_long_delay_rate).fillna(
        stats.defaults["route_long_delay_rate"]
    )
    frame["origin_disruption_rate"] = frame["ORIGIN"].map(stats.origin_disruption_rate).fillna(
        stats.defaults["origin_disruption_rate"]
    )
    frame["dest_disruption_rate"] = frame["DEST"].map(stats.dest_disruption_rate).fillna(
        stats.defaults["dest_disruption_rate"]
    )
    frame["route_month_delay_rate"] = frame["route_month_key"].map(stats.route_month_delay_rate).fillna(
        stats.defaults["route_month_delay_rate"]
    )
    frame["route_month_cancel_rate"] = frame["route_month_key"].map(stats.route_month_cancel_rate).fillna(
        stats.defaults["route_month_cancel_rate"]
    )
    return frame


def build_route_options(train_df: pd.DataFrame, min_support: int = 15) -> pd.DataFrame:
    frame = add_time_features(train_df)
    weather_columns = [column for column in WEATHER_COLUMNS if column in frame.columns]

    group_columns = ["ORIGIN", "DEST", "AIRLINE_CODE", "CRS_DEP_TIME"]
    aggregations = {
        "DISTANCE": "median",
        "CRS_ELAPSED_TIME": "median",
        DATE_COLUMN: "count",
    }
    for column in weather_columns:
        aggregations[column] = "median"

    options = (
        frame.groupby(group_columns, dropna=False)
        .agg(aggregations)
        .rename(columns={DATE_COLUMN: "historical_option_support"})
        .reset_index()
    )

    flight_number_lookup = (
        frame.groupby(group_columns, dropna=False)["FL_NUMBER"]
        .agg(lambda series: series.mode().iloc[0] if not series.mode().empty else series.dropna().iloc[0])
        .reset_index()
        .rename(columns={"FL_NUMBER": "FL_NUMBER_REP"})
    )

    options = options.merge(flight_number_lookup, on=group_columns, how="left")

    options = options[options["historical_option_support"] >= min_support].copy()
    options["scheduled_departure"] = options["CRS_DEP_TIME"].map(format_hhmm)
    options["scheduled_departure_ampm"] = options["CRS_DEP_TIME"].map(format_hhmm_ampm)
    options["flight_code"] = (
        options["AIRLINE_CODE"].fillna("").astype(str)
        + options["FL_NUMBER_REP"].fillna("").astype(str).str.replace(".0", "", regex=False)
    )
    return options.sort_values(
        ["ORIGIN", "DEST", "historical_option_support"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def build_feature_frame(df: pd.DataFrame, stats: AggregateStats) -> pd.DataFrame:
    frame = apply_aggregate_stats(df, stats)
    for column in WEATHER_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    if "historical_option_support" not in frame.columns:
        frame["historical_option_support"] = 0
    frame["historical_option_support"] = pd.to_numeric(
        frame["historical_option_support"], errors="coerce"
    ).fillna(0)
    return frame


def format_hhmm(value: float | int | str) -> str:
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return "Unknown"
    hours = max(0, min(23, numeric // 100))
    minutes = max(0, min(59, numeric % 100))
    return f"{hours:02d}:{minutes:02d}"


def format_hhmm_ampm(value: float | int | str) -> str:
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return "Unknown"
    hours = max(0, min(23, numeric // 100))
    minutes = max(0, min(59, numeric % 100))
    suffix = "AM" if hours < 12 else "PM"
    display_hour = hours % 12
    if display_hour == 0:
        display_hour = 12
    return f"{display_hour}:{minutes:02d} {suffix}"
