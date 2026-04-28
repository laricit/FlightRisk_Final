from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flightrisk.predictor import FlightRiskPredictor


st.set_page_config(page_title="Flight Risk", layout="wide")

BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "flight_risk_bundle.joblib"


def estimate_ticket_prices(results: pd.DataFrame, price_mode: str, default_ticket_price: float) -> pd.Series:
    if price_mode == "Use same ticket price":
        return pd.Series(default_ticket_price, index=results.index)

    distance = pd.to_numeric(results["distance"], errors="coerce").fillna(500)
    elapsed = pd.to_numeric(results["scheduled_elapsed_time"], errors="coerce").fillna(120)
    departure_time = pd.to_numeric(results["CRS_DEP_TIME"], errors="coerce").fillna(1200)
    peak_departure_premium = departure_time.between(600, 900) | departure_time.between(1600, 1900)

    estimate = 65 + (distance * 0.14) + (elapsed * 0.08) + peak_departure_premium.astype(int) * 25
    return estimate.round(0).clip(lower=79)


def add_budget_analysis(
    results: pd.DataFrame,
    ticket_prices: pd.Series,
    budget_amount: float,
    short_delay_cost: float,
    long_delay_cost: float,
    cancellation_cost: float,
) -> pd.DataFrame:
    enriched = results.copy()
    enriched["ticket_price"] = pd.to_numeric(ticket_prices, errors="coerce").fillna(0).clip(lower=0)
    enriched["expected_disruption_cost"] = (
        (enriched["short_delay"] * short_delay_cost)
        + (enriched["long_delay"] * long_delay_cost)
        + (enriched["cancelled"] * cancellation_cost)
    )
    enriched["risk_adjusted_cost"] = enriched["ticket_price"] + enriched["expected_disruption_cost"]
    enriched["budget_gap"] = budget_amount - enriched["risk_adjusted_cost"]
    enriched["within_budget"] = enriched["risk_adjusted_cost"] <= budget_amount
    return enriched


def sort_budget_results(results: pd.DataFrame, sort_choice: str) -> pd.DataFrame:
    if sort_choice == "Lowest risk":
        sort_columns = ["risk_score", "cancelled", "long_delay", "risk_adjusted_cost"]
        ascending = [True, True, True, True]
    elif sort_choice == "Lowest ticket price":
        sort_columns = ["ticket_price", "risk_score", "risk_adjusted_cost"]
        ascending = [True, True, True]
    elif sort_choice == "Lowest disruption cost":
        sort_columns = ["expected_disruption_cost", "risk_score", "ticket_price"]
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
    budget_col1, budget_col2, budget_col3 = st.columns(3)
    with budget_col1:
        budget_amount = st.number_input("Trip budget", min_value=0.0, value=500.0, step=25.0, format="%.0f")
    with budget_col2:
        price_mode = st.selectbox("Ticket price source", ["Estimate from distance", "Use same ticket price"])
    with budget_col3:
        default_ticket_price = st.number_input(
            "Default ticket price",
            min_value=0.0,
            value=250.0,
            step=10.0,
            format="%.0f",
            disabled=price_mode != "Use same ticket price",
        )

    cost_col1, cost_col2, cost_col3, cost_col4 = st.columns(4)
    with cost_col1:
        short_delay_cost = st.number_input("Short delay cost", min_value=0.0, value=50.0, step=10.0, format="%.0f")
    with cost_col2:
        long_delay_cost = st.number_input("Long delay cost", min_value=0.0, value=175.0, step=25.0, format="%.0f")
    with cost_col3:
        cancellation_cost = st.number_input("Cancellation cost", min_value=0.0, value=450.0, step=25.0, format="%.0f")
    with cost_col4:
        sort_choice = st.selectbox(
            "Rank by",
            ["Best value", "Lowest risk", "Lowest ticket price", "Lowest disruption cost"],
        )

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
    st.session_state["result_key"] = (
        f"{origin}_{destination}_{pd.Timestamp(travel_date).date().isoformat()}_"
        f"{selected_airline}_{top_n}"
    )

results = st.session_state.get("ranked_results")
if results is not None:
    if results.empty:
        st.warning("No historical route options were found for that route and airline selection.")
    else:
        results = results.reset_index(drop=True)
        estimated_prices = estimate_ticket_prices(results, price_mode, default_ticket_price)
        price_editor = results[["flight_code", "airline_code", "departure_time"]].copy()
        price_editor["ticket_price"] = estimated_prices
        edited_prices = st.data_editor(
            price_editor,
            column_config={
                "flight_code": st.column_config.TextColumn("Flight"),
                "airline_code": st.column_config.TextColumn("Airline"),
                "departure_time": st.column_config.TextColumn("Departure"),
                "ticket_price": st.column_config.NumberColumn(
                    "Ticket price",
                    min_value=0,
                    step=10,
                    format="$%.0f",
                ),
            },
            disabled=["flight_code", "airline_code", "departure_time"],
            hide_index=True,
            use_container_width=True,
            key=(
                f"ticket_price_editor_{st.session_state.get('result_key', 'current')}_"
                f"{price_mode}_{int(default_ticket_price)}"
            ),
        )

        budget_results = add_budget_analysis(
            results=results,
            ticket_prices=edited_prices["ticket_price"],
            budget_amount=budget_amount,
            short_delay_cost=short_delay_cost,
            long_delay_cost=long_delay_cost,
            cancellation_cost=cancellation_cost,
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
            "ticket_price",
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
        money_cols = ["ticket_price", "expected_disruption_cost", "risk_adjusted_cost"]
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
