# FlightRisk With Budget

FlightRisk With Budget is a Streamlit app that ranks historical flight options by disruption risk and adds a budget-aware comparison layer.

The app predicts four outcomes:

- `on_time`
- `short_delay`
- `long_delay`
- `cancelled`

It then lets users enter a trip budget, while ticket prices are predicted from `flight_dataset.csv` and delay/cancellation costs are estimated automatically before comparing options by risk-adjusted cost.

## What Is Included

- `app.py`: Streamlit app entry point
- `src/flightrisk/`: model loading, feature engineering, and prediction code
- `artifacts/flight_risk_bundle.joblib`: trained model bundle used by the app
- `artifacts/metrics.json`: model metrics from training
- `flight_dataset.csv`: fare prediction reference data used by the budget layer
- `requirements.txt`: Python dependencies for Streamlit Cloud
- `.streamlit/config.toml`: basic Streamlit display settings

Large local datasets are intentionally not included in this deployable folder.

## Run Locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy On Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload the contents of this `FlightRisk_WithBudget` folder to that repository.
3. In Streamlit Community Cloud, create a new app from the GitHub repo.
4. Set the main file path to:

```text
app.py
```

5. Deploy the app.

The trained model is already included at `artifacts/flight_risk_bundle.joblib`, so the app should not need the original CSV or parquet files to run.

## Retraining

`train_model.py` is included for future retraining, but retraining requires flight data files that are not committed to this deployment repo.
