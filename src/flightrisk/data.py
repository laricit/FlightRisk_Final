from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from flightrisk.config import BASE_COLUMNS, DATE_COLUMN, TARGET_NAME, WEATHER_COLUMNS


@dataclass
class DataSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_dataset(path: str | Path, sample_size: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        usecols = lambda column: column in set(BASE_COLUMNS + WEATHER_COLUMNS)
        df = pd.read_csv(path, usecols=usecols)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=42).sort_index()

    return df


def create_target_labels(df: pd.DataFrame) -> pd.Series:
    cancelled = df["CANCELLED"].fillna(0).astype(float) >= 1
    arr_delay = pd.to_numeric(df["ARR_DELAY"], errors="coerce")

    labels = pd.Series("on_time", index=df.index, dtype="object")
    labels = labels.mask((~cancelled) & (arr_delay > 0) & (arr_delay < 30), "short_delay")
    labels = labels.mask((~cancelled) & (arr_delay >= 30), "long_delay")
    labels = labels.mask(cancelled, "cancelled")
    return labels


def prepare_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    modeling_df = df.copy()
    modeling_df[DATE_COLUMN] = pd.to_datetime(modeling_df[DATE_COLUMN], errors="coerce")
    modeling_df["ARR_DELAY"] = pd.to_numeric(modeling_df["ARR_DELAY"], errors="coerce")
    modeling_df["CANCELLED"] = pd.to_numeric(modeling_df["CANCELLED"], errors="coerce").fillna(0.0)
    modeling_df["CRS_DEP_TIME"] = pd.to_numeric(modeling_df["CRS_DEP_TIME"], errors="coerce")
    modeling_df["CRS_ELAPSED_TIME"] = pd.to_numeric(modeling_df["CRS_ELAPSED_TIME"], errors="coerce")
    modeling_df["DISTANCE"] = pd.to_numeric(modeling_df["DISTANCE"], errors="coerce")

    weather_columns = [column for column in WEATHER_COLUMNS if column in modeling_df.columns]
    for column in weather_columns:
        modeling_df[column] = pd.to_numeric(modeling_df[column], errors="coerce")

    modeling_df = modeling_df.dropna(subset=[DATE_COLUMN, "AIRLINE_CODE", "ORIGIN", "DEST", "CRS_DEP_TIME"])
    modeling_df[TARGET_NAME] = create_target_labels(modeling_df)
    modeling_df = modeling_df[modeling_df[TARGET_NAME].notna()].copy()
    return modeling_df


def time_split_dataset(df: pd.DataFrame) -> DataSplit:
    years = df[DATE_COLUMN].dt.year
    max_year = int(years.max())
    unique_years = sorted(years.dropna().unique())

    if len(unique_years) >= 3:
        train_cutoff = unique_years[-3]
        validation_year = unique_years[-2]
        test_year = unique_years[-1]

        train = df[years <= train_cutoff].copy()
        validation = df[years == validation_year].copy()
        test = df[years == test_year].copy()
    else:
        sorted_df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
        train_end = int(len(sorted_df) * 0.7)
        validation_end = int(len(sorted_df) * 0.85)
        train = sorted_df.iloc[:train_end].copy()
        validation = sorted_df.iloc[train_end:validation_end].copy()
        test = sorted_df.iloc[validation_end:].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("Time split produced an empty train, validation, or test set.")

    return DataSplit(train=train, validation=validation, test=test)
