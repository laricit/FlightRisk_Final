from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from flightrisk.config import WEATHER_COLUMNS
from flightrisk.features import build_feature_frame
from flightrisk.modeling import FlightRiskBundle


@dataclass
class FlightRiskPredictor:
    bundle: FlightRiskBundle

    @classmethod
    def load(cls, path: str | Path) -> "FlightRiskPredictor":
        bundle = joblib.load(path)
        return cls(bundle=bundle)

    def available_airlines_for_route(self, origin: str, destination: str) -> list[str]:
        options = self._route_subset(origin, destination)
        airlines = sorted(options["AIRLINE_CODE"].dropna().astype(str).unique().tolist())
        return airlines

    def rank_route_options(
        self,
        origin: str,
        destination: str,
        flight_date: pd.Timestamp,
        airline_code: str | None = None,
        top_n: int = 8,
    ) -> pd.DataFrame:
        candidates = self._route_subset(origin, destination)
        if airline_code:
            candidates = candidates[candidates["AIRLINE_CODE"] == airline_code].copy()
        if candidates.empty:
            return pd.DataFrame()

        inference_frame = self._build_inference_frame(candidates, pd.Timestamp(flight_date))
        feature_frame = build_feature_frame(inference_frame, self.bundle.aggregate_stats)
        x = feature_frame[self.bundle.feature_columns].copy()
        probabilities = self.bundle.model.predict_proba(x)

        result = candidates.copy().reset_index(drop=True)
        for index, label in enumerate(self.bundle.label_encoder.classes_):
            result[label] = probabilities[:, index]

        result["risk_score"] = (
            result["cancelled"] + (0.6 * result["long_delay"]) + (0.2 * result["short_delay"])
        )
        result["airline_code"] = result["AIRLINE_CODE"]
        result["flight_code"] = result["flight_code"]
        result["distance"] = result["DISTANCE"]
        result["scheduled_elapsed_time"] = result["CRS_ELAPSED_TIME"]
        result["historical_support"] = result["historical_option_support"]
        result["departure_time"] = result["scheduled_departure_ampm"]
        result["explanation"] = result.apply(self._explain_row, axis=1)

        return result.sort_values(
            ["risk_score", "cancelled", "long_delay", "short_delay", "historical_support"],
            ascending=[True, True, True, True, False],
        ).head(top_n)

    def _route_subset(self, origin: str, destination: str) -> pd.DataFrame:
        route_options = self.bundle.route_options
        return route_options[
            (route_options["ORIGIN"].astype(str).str.upper() == origin.upper())
            & (route_options["DEST"].astype(str).str.upper() == destination.upper())
        ].copy()

    def _build_inference_frame(self, candidates: pd.DataFrame, flight_date: pd.Timestamp) -> pd.DataFrame:
        frame = candidates.copy()
        frame["FL_DATE"] = pd.Timestamp(flight_date)
        frame["CANCELLED"] = 0
        frame["ARR_DELAY"] = 0
        frame["AIRLINE"] = frame["AIRLINE_CODE"]
        frame["FL_NUMBER"] = 0
        for column in WEATHER_COLUMNS:
            if column not in frame.columns:
                frame[column] = self.bundle.numeric_fill_values.get(column, 0.0)
        return frame

    def _explain_row(self, row: pd.Series) -> str:
        reasons: list[str] = []
        if row.get("cancelled", 0.0) >= 0.15:
            reasons.append("elevated cancellation probability")
        if row.get("long_delay", 0.0) >= 0.25:
            reasons.append("high long-delay risk")
        if row.get("historical_support", 0.0) >= 50:
            reasons.append("well-supported by route history")
        if row.get("origin_precipitation_mm", 0.0) > 2 or row.get("dest_precipitation_mm", 0.0) > 2:
            reasons.append("wetter-than-usual conditions in history")
        if not reasons:
            reasons.append("historically lower disruption profile")
        return "; ".join(reasons[:3])
