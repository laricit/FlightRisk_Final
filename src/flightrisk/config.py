from __future__ import annotations

TARGET_NAME = "risk_label"
DATE_COLUMN = "FL_DATE"

WEATHER_COLUMNS = [
    "origin_temp_mean_c",
    "origin_wind_speed_max_kmh",
    "origin_pressure_msl_hpa",
    "origin_humidity_pct",
    "origin_precipitation_mm",
    "origin_snowfall_cm",
    "dest_temp_mean_c",
    "dest_wind_speed_max_kmh",
    "dest_pressure_msl_hpa",
    "dest_humidity_pct",
    "dest_precipitation_mm",
    "dest_snowfall_cm",
]

BASE_COLUMNS = [
    DATE_COLUMN,
    "AIRLINE",
    "AIRLINE_CODE",
    "FL_NUMBER",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
    "CANCELLED",
    "ARR_DELAY",
]

LEAKAGE_COLUMNS = [
    "DEP_TIME",
    "DEP_DELAY",
    "TAXI_OUT",
    "WHEELS_OFF",
    "WHEELS_ON",
    "TAXI_IN",
    "ARR_TIME",
    "ARR_DELAY",
    "ELAPSED_TIME",
    "AIR_TIME",
    "DELAY_DUE_CARRIER",
    "DELAY_DUE_WEATHER",
    "DELAY_DUE_NAS",
    "DELAY_DUE_SECURITY",
    "DELAY_DUE_LATE_AIRCRAFT",
    "CANCELLATION_CODE",
    "DIVERTED",
]

CATEGORICAL_FEATURES = [
    "AIRLINE_CODE",
    "ORIGIN",
    "DEST",
    "dep_time_bucket",
    "month",
    "day_of_week",
    "is_weekend",
    "is_holiday_window",
]

NUMERIC_FEATURES = [
    "dep_hour",
    "dep_minute",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
    "airline_cancel_rate",
    "airline_long_delay_rate",
    "route_cancel_rate",
    "route_long_delay_rate",
    "origin_disruption_rate",
    "dest_disruption_rate",
    "route_month_delay_rate",
    "route_month_cancel_rate",
    "historical_option_support",
] + WEATHER_COLUMNS

TARGET_ORDER = ["on_time", "short_delay", "long_delay", "cancelled"]
