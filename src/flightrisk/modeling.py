from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from flightrisk.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_NAME, TARGET_ORDER
from flightrisk.data import DataSplit
from flightrisk.features import AggregateStats, build_aggregate_stats, build_feature_frame, build_route_options


@dataclass
class FlightRiskBundle:
    model: Pipeline
    label_encoder: LabelEncoder
    aggregate_stats: AggregateStats
    route_options: pd.DataFrame
    numeric_fill_values: dict[str, float]
    feature_columns: list[str]

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)


def _make_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ]
    )
    classifier = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def _prepare_xy(frame: pd.DataFrame, stats: AggregateStats) -> tuple[pd.DataFrame, pd.Series]:
    features = build_feature_frame(frame, stats)
    x = features[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
    y = frame[TARGET_NAME].copy()
    return x, y


def _evaluate_model(
    model: Pipeline,
    label_encoder: LabelEncoder,
    frame: pd.DataFrame,
    stats: AggregateStats,
) -> dict[str, object]:
    x, y_true_labels = _prepare_xy(frame, stats)
    y_true = label_encoder.transform(y_true_labels)
    probabilities = model.predict_proba(x)
    predictions = probabilities.argmax(axis=1)

    return {
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "log_loss": float(log_loss(y_true, probabilities, labels=np.arange(len(label_encoder.classes_)))),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            target_names=label_encoder.classes_.tolist(),
            output_dict=True,
            zero_division=0,
        ),
    }


def train_bundle(split: DataSplit, min_route_support: int = 15) -> tuple[FlightRiskBundle, dict[str, object]]:
    stats = build_aggregate_stats(split.train)
    x_train, y_train_labels = _prepare_xy(split.train, stats)

    observed_classes = set(y_train_labels.astype(str).unique().tolist())
    missing_classes = [label for label in TARGET_ORDER if label not in observed_classes]
    if missing_classes:
        raise ValueError(
            "Training data is missing required target classes: "
            + ", ".join(missing_classes)
            + ". Use a larger or less filtered dataset."
        )

    label_encoder = LabelEncoder()
    label_encoder.fit(TARGET_ORDER)
    y_train = label_encoder.transform(y_train_labels)

    model = _make_pipeline()
    model.fit(x_train, y_train)

    route_options = build_route_options(split.train, min_support=min_route_support)
    numeric_fill_values = {
        column: float(x_train[column].median()) if column in x_train.columns else 0.0 for column in NUMERIC_FEATURES
    }

    bundle = FlightRiskBundle(
        model=model,
        label_encoder=label_encoder,
        aggregate_stats=stats,
        route_options=route_options,
        numeric_fill_values=numeric_fill_values,
        feature_columns=CATEGORICAL_FEATURES + NUMERIC_FEATURES,
    )

    metrics = {
        "train_rows": int(len(split.train)),
        "validation_rows": int(len(split.validation)),
        "test_rows": int(len(split.test)),
        "class_order": TARGET_ORDER,
        "validation": _evaluate_model(model, label_encoder, split.validation, stats),
        "test": _evaluate_model(model, label_encoder, split.test, stats),
    }
    return bundle, metrics
