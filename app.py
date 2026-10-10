import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Store-Item Demand Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Store-Item Demand Forecasting")
st.markdown(
    "Forecast future product demand using a trained "
    "Random Forest regression model."
)

ARTIFACT_DIR = Path(__file__).parent / "artifacts"


# --------------------------------------------------
# 2. LOAD THE SAVED MODEL AND HISTORICAL DATA
# --------------------------------------------------
@st.cache_resource
def load_model():
    bundle = joblib.load(
        ARTIFACT_DIR / "forecast_model.joblib"
    )
    return bundle["model"], bundle["feature_cols"]


@st.cache_data
def load_history():
    history = pd.read_csv(
        ARTIFACT_DIR / "historical_sales.csv",
        parse_dates=["date"]
    )

    history = history.sort_values(
        ["store", "item", "date"]
    ).reset_index(drop=True)

    return history


try:
    rf_model, feature_cols = load_model()
    history = load_history()
except Exception as e:
    st.error(f"Could not load project files: {e}")
    st.stop()


# --------------------------------------------------
# 3. RECURSIVE FORECASTING FUNCTION
# --------------------------------------------------
def forecast_sales(
    history,
    store_id,
    item_id,
    horizon,
    model,
    feature_cols
):
    series = history[
        (history["store"] == store_id)
        & (history["item"] == item_id)
    ].sort_values("date")

    if series.empty:
        raise ValueError(
            "No historical sales found for this combination."
        )

    if series["date"].duplicated().any():
        raise ValueError(
            "Duplicate dates found for this store-item series."
        )

    # Ensure we have sufficient history
    if len(series) < 28:
        raise ValueError(
            "At least 28 historical observations are required."
        )

    # Verify that historical observations are daily and continuous
    dates = pd.DatetimeIndex(series["date"])

    if not dates.equals(
        pd.date_range(dates.min(), dates.max(), freq="D")
    ):
        raise ValueError(
            "Historical dates contain gaps. "
            "Check the sales data before forecasting."
        )

    sales_values = series["sales"].astype(float).tolist()
    last_date = pd.Timestamp(series["date"].max())

    forecast_rows = []

    for step in range(1, horizon + 1):
        current_date = last_date + pd.Timedelta(days=step)

        # Calendar features
        row = {
            "store": store_id,
            "item": item_id,
            "year": current_date.year,
            "month": current_date.month,
            "day_of_week": current_date.dayofweek,
            "day_of_month": current_date.day,
            "is_weekend": int(
                current_date.dayofweek >= 5
            ),

            # Lag features
            "lag_7": sales_values[-7],
            "lag_14": sales_values[-14],
            "lag_28": sales_values[-28],

            # Rolling means use previous observations only
            "rolling_mean_7": np.mean(
                sales_values[-7:]
            ),
            "rolling_mean_28": np.mean(
                sales_values[-28:]
            )
        }

        # Preserve the exact feature order used during training
        X_next = pd.DataFrame(
            [row],
            columns=feature_cols
        )

        prediction = float(
            model.predict(X_next)[0]
        )

        # Demand cannot be negative
        prediction = max(0.0, prediction)

        forecast_rows.append({
            "date": current_date,
            "store": store_id,
            "item": item_id,
            "predicted_sales": prediction
        })

        # Recursive forecasting:
        # feed today's prediction into future lag features
        sales_values.append(prediction)

    return pd.DataFrame(forecast_rows), series


# --------------------------------------------------
# 4. SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("Forecast Settings")

store_ids = sorted(
    history["store"].unique().tolist()
)
item_ids = sorted(
    history["item"].unique().tolist()
)

store_id = st.sidebar.selectbox(
    "Select Store",
    store_ids
)

item_id = st.sidebar.selectbox(
    "Select Item",
    item_ids
)

horizon = st.sidebar.slider(
    "Forecast Horizon (days)",
    min_value=7,
    max_value=90,
    value=30,
    step=1
)

st.sidebar.caption(
    "Forecasts begin on the day after the selected "
    "store-item's last historical observation."
)


# --------------------------------------------------
# 5. GENERATE THE FORECAST
# --------------------------------------------------
try:
    forecast_df, selected_history = forecast_sales(
        history=history,
        store_id=store_id,
        item_id=item_id,
        horizon=horizon,
        model=rf_model,
        feature_cols=feature_cols
    )
except Exception as e:
    st.error(f"Forecast generation failed: {e}")
    st.stop()


# --------------------------------------------------
# 6. DISPLAY SUMMARY METRICS
# --------------------------------------------------
total_forecast = forecast_df["predicted_sales"].sum()
average_forecast = forecast_df["predicted_sales"].mean()
peak_forecast = forecast_df["predicted_sales"].max()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Forecast Demand",
    f"{total_forecast:,.0f} units"
)

col2.metric(
    "Average Daily Demand",
    f"{average_forecast:,.2f} units"
)

col3.metric(
    "Peak Daily Demand",
    f"{peak_forecast:,.2f} units"
)

st.caption(
    f"Store {store_id} · Item {item_id} · "
    f"{forecast_df['date'].min():%d %b %Y} to "
    f"{forecast_df['date'].max():%d %b %Y}"
)


# --------------------------------------------------
# 7. PLOT HISTORICAL AND FORECAST SALES
# --------------------------------------------------
recent_history = selected_history.tail(90)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=recent_history["date"],
        y=recent_history["sales"],
        mode="lines",
        name="Historical Sales",
        line=dict(color="#2563eb", width=2)
    )
)

fig.add_trace(
    go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["predicted_sales"],
        mode="lines+markers",
        name="Forecast Sales",
        line=dict(color="#f97316", width=2),
        marker=dict(size=4)
    )
)

# Mark the beginning of the forecast
forecast_start_date = pd.to_datetime(
    forecast_df["date"]
).min()

fig.add_shape(
    type="line",
    x0=forecast_start_date,
    x1=forecast_start_date,
    y0=0,
    y1=1,
    xref="x",
    yref="paper",
    line=dict(
        color="red",
        dash="dash",
        width=2
    )
)

fig.update_layout(
    title="Historical Sales vs Future Demand Forecast",
    xaxis_title="Date",
    yaxis_title="Units Sold",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", y=1.1)
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 8. FORECAST TABLE
# --------------------------------------------------
st.subheader("Daily Forecast")

display_forecast = forecast_df.copy()
display_forecast["date"] = (
    display_forecast["date"].dt.strftime("%Y-%m-%d")
)
display_forecast["predicted_sales"] = (
    display_forecast["predicted_sales"].round(2)
)

st.dataframe(
    display_forecast,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 9. DOWNLOAD FORECAST
# --------------------------------------------------
csv_data = forecast_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Forecast CSV",
    data=csv_data,
    file_name=f"store_{store_id}_item_{item_id}_forecast.csv",
    mime="text/csv"
)


# --------------------------------------------------
# 10. MODEL INFORMATION
# --------------------------------------------------
with st.expander("About this forecasting model"):
    st.write(
        "This application uses a trained Random Forest "
        "regressor with calendar, lag and rolling-mean features."
    )

    st.write(
        "Forecasts are generated recursively: each predicted "
        "day is used to construct the features for subsequent days."
    )

    st.write(
        "The model achieved a validation MAE of 6.1511, "
        "RMSE of 8.0095 and MAPE of 13.43% on the project's "
        "historical validation period. These are validation "
        "metrics, not a guarantee of future accuracy."
    )

    st.warning(
        "Longer forecast horizons may accumulate errors because "
        "future predictions become inputs to later predictions."
    )
