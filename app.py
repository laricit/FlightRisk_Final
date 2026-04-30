from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flightrisk.predictor import FlightRiskPredictor


st.set_page_config(page_title="Flight Risk", layout="wide")

BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "flight_risk_bundle.joblib"
PRICE_DATA_PATH = PROJECT_ROOT / "flight_dataset.csv"
PRICE_FEATURE_COLUMNS = [
    "Total_Stops",
    "Date",
    "Month",
    "Dep_hours",
    "Dep_min",
    "Duration_hours",
    "Duration_min",
]
PRICE_DATA_TO_DOLLARS = 0.012
SHORT_DELAY_COST_RATE = 0.12
LONG_DELAY_COST_RATE = 0.45
CANCELLATION_COST_RATE = 1.2


def estimate_ticket_prices_from_route(results: pd.DataFrame) -> pd.Series:
    distance = pd.to_numeric(results["distance"], errors="coerce").fillna(500)
    elapsed = pd.to_numeric(results["scheduled_elapsed_time"], errors="coerce").fillna(120)
    departure_time = pd.to_numeric(results["CRS_DEP_TIME"], errors="coerce").fillna(1200)
    peak_departure_premium = departure_time.between(600, 900) | departure_time.between(1600, 1900)

    estimate = 65 + (distance * 0.14) + (elapsed * 0.08) + peak_departure_premium.astype(int) * 25
    return estimate.round(0).clip(lower=79)


@st.cache_resource
def load_price_model(dataset_path: Path) -> dict[str, object]:
    if not dataset_path.exists():
        return {"available": False, "error": "flight_dataset.csv was not found."}

    try:
        raw = pd.read_csv(dataset_path)
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    required_columns = set(PRICE_FEATURE_COLUMNS + ["Price"])
    missing_columns = sorted(required_columns - set(raw.columns))
    if missing_columns:
        return {
            "available": False,
            "error": f"flight_dataset.csv is missing: {', '.join(missing_columns)}",
        }

    price_frame = raw[list(required_columns)].copy()
    for column in required_columns:
        price_frame[column] = pd.to_numeric(price_frame[column], errors="coerce")
    price_frame = price_frame.dropna()
    price_frame = price_frame[
        (price_frame["Price"] > 0)
        & ((price_frame["Duration_hours"] * 60 + price_frame["Duration_min"]) > 0)
    ]
    if price_frame.empty:
        return {"available": False, "error": "flight_dataset.csv did not contain usable prices."}

    model = HistGradientBoostingRegressor(
        learning_rate=0.06,
        l2_regularization=0.05,
        max_iter=140,
        max_leaf_nodes=15,
        random_state=42,
    )
    model.fit(price_frame[PRICE_FEATURE_COLUMNS], np.log1p(price_frame["Price"]))

    elapsed_minutes = price_frame["Duration_hours"] * 60 + price_frame["Duration_min"]
    return {
        "available": True,
        "model": model,
        "lower_price": float(price_frame["Price"].quantile(0.05)),
        "upper_price": float(price_frame["Price"].quantile(0.95)),
        "median_elapsed": float(elapsed_minutes.median()),
    }


def split_departure_time(results: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    departure_time = pd.to_numeric(results["CRS_DEP_TIME"], errors="coerce").fillna(1200).astype(int)
    departure_hours = (departure_time // 100).clip(0, 23)
    departure_minutes = (departure_time % 100).clip(0, 59)
    return departure_hours, departure_minutes


def estimate_ticket_prices(
    results: pd.DataFrame,
    price_model: dict[str, object],
    travel_date: pd.Timestamp,
) -> pd.Series:
    if not price_model.get("available"):
        return estimate_ticket_prices_from_route(results)

    elapsed = pd.to_numeric(results["scheduled_elapsed_time"], errors="coerce").fillna(
        float(price_model.get("median_elapsed", 120))
    )
    elapsed = elapsed.clip(lower=30, upper=48 * 60)
    departure_hours, departure_minutes = split_departure_time(results)
    travel_timestamp = pd.Timestamp(travel_date)

    features = pd.DataFrame(index=results.index)
    features["Total_Stops"] = 0
    features["Date"] = travel_timestamp.day
    features["Month"] = travel_timestamp.month
    features["Dep_hours"] = departure_hours
    features["Dep_min"] = departure_minutes
    features["Duration_hours"] = np.floor(elapsed / 60).astype(int).clip(1, 47)
    features["Duration_min"] = elapsed.round().astype(int).mod(60).clip(0, 59)

    model = price_model["model"]
    predicted_price = np.expm1(model.predict(features[PRICE_FEATURE_COLUMNS]))
    predicted_price = np.clip(
        predicted_price,
        float(price_model["lower_price"]),
        float(price_model["upper_price"]),
    )
    return pd.Series(predicted_price * PRICE_DATA_TO_DOLLARS, index=results.index).round(0).clip(lower=49)


def add_budget_analysis(
    results: pd.DataFrame,
    ticket_prices: pd.Series,
    budget_amount: float,
) -> pd.DataFrame:
    enriched = results.copy()
    enriched["predicted_ticket_price"] = pd.to_numeric(ticket_prices, errors="coerce").fillna(0).clip(lower=0)
    short_delay_cost = enriched["predicted_ticket_price"] * SHORT_DELAY_COST_RATE
    long_delay_cost = enriched["predicted_ticket_price"] * LONG_DELAY_COST_RATE
    cancellation_cost = enriched["predicted_ticket_price"] * CANCELLATION_COST_RATE
    enriched["expected_disruption_cost"] = (
        (enriched["short_delay"] * short_delay_cost)
        + (enriched["long_delay"] * long_delay_cost)
        + (enriched["cancelled"] * cancellation_cost)
    )
    enriched["risk_adjusted_cost"] = enriched["predicted_ticket_price"] + enriched["expected_disruption_cost"]
    enriched["budget_gap"] = budget_amount - enriched["risk_adjusted_cost"]
    enriched["within_budget"] = enriched["risk_adjusted_cost"] <= budget_amount
    return enriched


def sort_budget_results(results: pd.DataFrame, sort_choice: str) -> pd.DataFrame:
    if sort_choice == "Lowest risk":
        sort_columns = ["risk_score", "cancelled", "long_delay", "risk_adjusted_cost"]
        ascending = [True, True, True, True]
    elif sort_choice == "Lowest predicted fare":
        sort_columns = ["predicted_ticket_price", "risk_score", "risk_adjusted_cost"]
        ascending = [True, True, True]
    elif sort_choice == "Lowest expected disruption cost":
        sort_columns = ["expected_disruption_cost", "risk_score", "predicted_ticket_price"]
        ascending = [True, True, True]
    else:
        sort_columns = ["within_budget", "risk_adjusted_cost", "risk_score", "historical_support"]
        ascending = [False, True, True, False]
    return results.sort_values(sort_columns, ascending=ascending)


def format_money(value: float) -> str:
    return f"${value:,.0f}"


def format_gap(value: float) -> str:
    if value >= 0:
        return f"${value:,.0f}"
    return f"-${abs(value):,.0f}"


@st.cache_resource
def load_predictor(bundle_path: Path) -> FlightRiskPredictor:
    return FlightRiskPredictor.load(bundle_path)


st.title("Flight Risk")
st.caption("Prototype flight disruption ranking tool built from historical flight and weather data.")

if not BUNDLE_PATH.exists():
    st.error(
        "No trained model bundle was found. Train the model first with "
        "`python train_model.py --data-path flights_with_weather.parquet --output-dir artifacts`."
    )
    st.stop()

predictor = load_predictor(BUNDLE_PATH)
price_model = load_price_model(PRICE_DATA_PATH)

col1, col2, col3 = st.columns(3)
with col1:
    origin = st.text_input("Origin airport", value="LAX").strip().upper()
with col2:
    destination = st.text_input("Destination airport", value="SLC").strip().upper()
with col3:
    travel_date = st.date_input("Travel date", value=date.today())

airlines = predictor.available_airlines_for_route(origin, destination)
selected_airline = st.selectbox("Airline filter", options=["Any"] + airlines, index=0)
top_n = st.slider("How many options to show", min_value=3, max_value=15, value=8)

with st.expander("Budget settings", expanded=True):
    budget_col1, budget_col2 = st.columns(2)
    with budget_col1:
        budget_amount = st.number_input("Trip budget", min_value=0.0, value=500.0, step=25.0, format="%.0f")
    with budget_col2:
        sort_choice = st.selectbox(
            "Rank by",
            ["Best value", "Lowest risk", "Lowest predicted fare", "Lowest expected disruption cost"],
        )
    if price_model.get("available"):
        st.caption("Ticket prices are predicted from flight_dataset.csv; disruption costs are estimated automatically.")
    else:
        st.caption("Ticket prices are estimated from route distance because flight_dataset.csv could not be loaded.")

if st.button("Rank Flight Options", type="primary"):
    airline_filter = None if selected_airline == "Any" else selected_airline
    results = predictor.rank_route_options(
        origin=origin,
        destination=destination,
        flight_date=pd.Timestamp(travel_date),
        airline_code=airline_filter,
        top_n=top_n,
    )
    st.session_state["ranked_results"] = results
    st.session_state["result_travel_date"] = pd.Timestamp(travel_date)

results = st.session_state.get("ranked_results")
if results is not None:
    if results.empty:
        st.warning("No historical route options were found for that route and airline selection.")
    else:
        results = results.reset_index(drop=True)
        result_travel_date = st.session_state.get("result_travel_date", pd.Timestamp(travel_date))
        estimated_prices = estimate_ticket_prices(
            results=results,
            price_model=price_model,
            travel_date=result_travel_date,
        )

        budget_results = add_budget_analysis(
            results=results,
            ticket_prices=estimated_prices,
            budget_amount=budget_amount,
        )
        budget_results = sort_budget_results(budget_results, sort_choice).reset_index(drop=True)

        best_value = budget_results.iloc[0]
        safest = budget_results.sort_values(["risk_score", "risk_adjusted_cost"]).iloc[0]
        within_budget_count = int(budget_results["within_budget"].sum())

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric(
                "Best value",
                f"{best_value['flight_code']} at {best_value['departure_time']}",
                f"{format_money(best_value['risk_adjusted_cost'])} adjusted",
            )
        with metric_col2:
            st.metric(
                "Safest option",
                f"{safest['flight_code']} at {safest['departure_time']}",
                f"{safest['risk_score']:.1%} risk",
            )
        with metric_col3:
            st.metric("Within budget", f"{within_budget_count}/{len(budget_results)}")

        display_cols = [
            "flight_code",
            "airline_code",
            "departure_time",
            "predicted_ticket_price",
            "expected_disruption_cost",
            "risk_adjusted_cost",
            "budget_gap",
            "within_budget",
            "distance",
            "scheduled_elapsed_time",
            "risk_score",
            "on_time",
            "short_delay",
            "long_delay",
            "cancelled",
            "historical_support",
            "explanation",
        ]
        formatted = budget_results[display_cols].copy()
        money_cols = ["predicted_ticket_price", "expected_disruption_cost", "risk_adjusted_cost"]
        for col in money_cols:
            formatted[col] = formatted[col].map(format_money)
        formatted["budget_gap"] = formatted["budget_gap"].map(format_gap)
        formatted["within_budget"] = formatted["within_budget"].map(lambda value: "Yes" if value else "No")
        probability_cols = ["risk_score", "on_time", "short_delay", "long_delay", "cancelled"]
        for col in probability_cols:
            formatted[col] = formatted[col].map(lambda value: f"{value:.1%}")
        st.dataframe(formatted, use_container_width=True, hide_index=True)

        best = budget_results.iloc[0]
        st.success(
            f"Recommended option: {best['airline_code']} at {best['scheduled_departure']} "
            f"with {format_money(best['risk_adjusted_cost'])} risk-adjusted cost."
        )
