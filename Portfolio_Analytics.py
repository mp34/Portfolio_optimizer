import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import GradientBoostingRegressor
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_percentage_error
import requests
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(page_title="📊 Portfolio Optimizer", layout="wide")
st.title("💹 AI-Powered Stock Portfolio Forecast")

# 1. Inputs
investment_amount = st.slider("💰 Total Investment", 100_000, 10_000_000, 1_000_000, step=100_000)
forecast_period = st.selectbox("📆 Forecast Horizon", ["1 Month", "3 Months", "6 Months", "1 Year", "3 Years", "5 Years"])
model_choice = st.multiselect("🧠 Models", ["Prophet", "ARIMA", "Ensemble"], default=["Ensemble"])
uploaded_files = st.file_uploader("📁 Upload CSVs", type="csv", accept_multiple_files=True)

# Forecast mapping
forecast_days = {
    "1 Month": 30, "3 Months": 90, "6 Months": 180,
    "1 Year": 365, "3 Years": 1095, "5 Years": 1825
}[forecast_period]

# 2. Forecast function
def forecast_stock(file, forecast_days, model_choice, per_stock_amt):
    try:
        df = pd.read_csv(file)
        df.columns = [c.lower().strip() for c in df.columns]
        date_col = next(c for c in df.columns if "date" in c)
        price_col = next(c for c in df.columns if "close" in c or "price" in c)
        df = df[[date_col, price_col]].rename(columns={date_col: "ds", price_col: "y"})
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df["y"] = df["y"].astype(str).str.replace(",", "").astype(float)
        df = df.dropna().sort_values("ds")
        df = df[df["ds"] >= datetime.now() - timedelta(days=5*365)]

        current_price = df["y"].iloc[-1]
        units = per_stock_amt // current_price
        invested = units * current_price

        prophet_price = arima_price = None
        if "Prophet" in model_choice or "Ensemble" in model_choice:
            m = Prophet()
            m.fit(df)
            future = m.make_future_dataframe(periods=forecast_days)
            forecast = m.predict(future)
            prophet_price = forecast.iloc[-1]["yhat"]
            mape = mean_absolute_percentage_error(df["y"][-forecast_days:], forecast["yhat"][-forecast_days:])
        else:
            mape = None

        if "ARIMA" in model_choice or "Ensemble" in model_choice:
            model = ARIMA(df["y"], order=(3,1,2)).fit()
            arima_price = model.forecast(steps=forecast_days).iloc[-1]

        forecast_price = prophet_price or arima_price
        if "Ensemble" in model_choice and prophet_price and arima_price:
            forecast_price = (prophet_price + arima_price) / 2

        forecast_price = max(forecast_price, 0.001)
        roi = ((forecast_price - current_price) / current_price) * 100
        roi = max(roi, -100)

        expected = units * forecast_price
        volatility = df["y"].pct_change().std() * np.sqrt(252)

        growth = pd.DataFrame({
            "ds": pd.date_range(df["ds"].iloc[-1] + timedelta(days=1), periods=forecast_days),
            "yhat": np.linspace(current_price, forecast_price, forecast_days),
        })
        growth["Total"] = growth["yhat"] * units
        growth["Stock"] = os.path.splitext(file.name)[0]

        row = {
            "Stock": os.path.splitext(file.name)[0],
            "Current": round(current_price, 2),
            "Forecast": round(forecast_price, 2),
            "Units": int(units),
            "Invested": round(invested, 2),
            "Expected": round(expected, 2),
            "Return %": round(roi, 2),
            "Volatility": round(volatility, 4),
            "MAPE": round(mape*100, 2) if mape else "-"
        }

        return (row, growth), None
    except Exception as e:
        return None, f"❌ {file.name}: {e}"

# 3. Run Forecast
if uploaded_files and len(uploaded_files) >= 5:
    st.subheader("📈 Forecasting...")
    per_stock_amt = investment_amount // len(uploaded_files)
    results, full_forecast = [], pd.DataFrame()

    with ThreadPoolExecutor() as ex:
        futures = [ex.submit(forecast_stock, f, forecast_days, model_choice, per_stock_amt) for f in uploaded_files]
        for f in futures:
            result, err = f.result()
            if err:
                st.error(err)
            else:
                row, chart = result
                results.append(row)
                full_forecast = pd.concat([full_forecast, chart])

    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

    # 4. Growth Line Charts
    portfolio = full_forecast.groupby("ds")["Total"].sum().reset_index()
    portfolio["Indexed"] = portfolio["Total"] / portfolio["Total"].iloc[0] * 100
    final_val = portfolio["Total"].iloc[-1]
    roi_orig = (final_val - investment_amount) / investment_amount * 100

    # 5. Optimization
    model = GradientBoostingRegressor().fit(df[["Volatility", "Return %"]], df["Expected"])
    df["Predicted"] = model.predict(df[["Volatility", "Return %"]])
    top5 = df.sort_values("Predicted", ascending=False).head(5)
    opt_amt = investment_amount // 5
    top5["OptUnits"] = opt_amt // top5["Current"]
    top5["OptInvested"] = top5["OptUnits"] * top5["Current"]
    top5["OptExpected"] = (top5["OptUnits"] * top5["Forecast"]).clip(lower=top5["OptInvested"] * 0.8)

    opt_chart = full_forecast[full_forecast["Stock"].isin(top5["Stock"])]
    opt_group = opt_chart.groupby("ds")["Total"].sum().reset_index()
    opt_group["Indexed"] = opt_group["Total"] / opt_group["Total"].iloc[0] * 100
    roi_opt = (opt_group["Total"].iloc[-1] - investment_amount) / investment_amount * 100

    st.subheader("📊 Portfolio Growth Comparison")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio["ds"], y=portfolio["Indexed"], name="Original"))
    fig.add_trace(go.Scatter(x=opt_group["ds"], y=opt_group["Indexed"], name="Optimized"))
    fig.update_layout(title="Indexed Growth", xaxis_title="Date", yaxis_title="Index (Base 100)")
    st.plotly_chart(fig, use_container_width=True)

    # 6. Summary Table
    st.markdown(f"""
    | Metric              | Original Portfolio | Optimized Portfolio |
    |---------------------|--------------------|---------------------|
    | Investment          | ₹{investment_amount:,}      | ₹{investment_amount:,}         |
    | Expected Value      | ₹{final_val:,.0f}           | ₹{top5['OptExpected'].sum():,.0f}     |
    | Projected ROI       | {roi_orig:.2f}%             | {roi_opt:.2f}%              |
    """)

    # 7. Top Optimized Stocks
    st.subheader("✅ Top 5 Optimized Stocks")
    st.dataframe(top5[["Stock", "Return %", "Expected", "Volatility", "MAPE"]])

# 8. 🔍 Fundamental Screener (via API)
st.subheader("📚 Fundamental Screener via API")
@st.cache_data
def fetch_screener():
    url = "https://api.tickertape.in/stocks/screener?limit=100&offset=0"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        df = pd.json_normalize(data["data"])
        return df
    except Exception as e:
        st.warning(f"Failed to fetch screener: {e}")
        return pd.DataFrame()

fundamentals = fetch_screener()
if not fundamentals.empty:
    fundamentals.rename(columns=lambda x: x.split(".")[-1], inplace=True)
    st.write("Screener preview:", fundamentals.head(3))

    if "sector" in fundamentals.columns:
        selected_sectors = st.multiselect("Filter by Sector", fundamentals["sector"].dropna().unique())
    else:
        selected_sectors = []

    roce_col = "roce"
    roe_col = "roe"
    eps_col = "eps"
    esg_col = "esg"

    min_roce = st.slider("Min ROCE (%)", 0, 50, 10)
    min_roe = st.slider("Min ROE (%)", 0, 50, 10)
    min_eps = st.slider("Min EPS", 0, 500, 10)
    esg_min = st.slider("Min ESG Score", 0, 100, 40)

    filtered = fundamentals.copy()
    if selected_sectors:
        filtered = filtered[filtered["sector"].isin(selected_sectors)]
    filtered = filtered[
        (filtered[roce_col] >= min_roce) &
        (filtered[roe_col] >= min_roe) &
        (filtered[eps_col] >= min_eps) &
        (filtered[esg_col] >= esg_min)
    ]
    st.dataframe(filtered, use_container_width=True)
else:
    st.info("Could not fetch screener data. Try again later.")
