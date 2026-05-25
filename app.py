import streamlit as st
from streamlit_autorefresh import st_autorefresh

import pandas as pd
import numpy as np
import yfinance as yf
import ta
import pickle



# =====================================
# AUTO REFRESH
# =====================================

st_autorefresh(
    interval=300000,
    key="btc_refresh"
)



# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="BTC Regression AI",
    layout="centered"
)



# =====================================
# TITLE
# =====================================

st.title(
    "BTC Price Regression Dashboard"
)



# =====================================
# LOAD MODEL
# =====================================

model = pickle.load(
    open(
        "btc_regression_model.pkl",
        "rb"
    )
)



# =====================================
# DOWNLOAD BTC DATA
# =====================================

df = yf.download(
    "BTC-USD",
    period="300d",
    interval="1d"
)



# =====================================
# RESET INDEX
# =====================================

df.reset_index(inplace=True)



# =====================================
# KEEP COLUMNS
# =====================================

df = df[[
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]]



# =====================================
# FLOAT CONVERSION
# =====================================

for col in [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]:

    df[col] = df[col].astype(float)



# =====================================
# INDICATORS
# =====================================

# RSI
df["RSI"] = ta.momentum.RSIIndicator(
    close=df["Close"],
    window=14
).rsi()



# MACD
macd = ta.trend.MACD(
    close=df["Close"]
)

df["MACD"] = macd.macd()

df["MACD_SIGNAL"] = macd.macd_signal()



# EMA 20
df["EMA_20"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=20
).ema_indicator()



# EMA 50
df["EMA_50"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=50
).ema_indicator()



# SMA 20
df["SMA_20"] = ta.trend.SMAIndicator(
    close=df["Close"],
    window=20
).sma_indicator()



# RETURNS
df["Returns"] = (
    df["Close"].pct_change()
)



# VOLATILITY
df["Volatility"] = (
    df["High"] - df["Low"]
) / df["Close"]



# =====================================
# CLEAN DATA
# =====================================

df.dropna(inplace=True)



# =====================================
# FEATURES
# =====================================

latest = df[[
    "Close",
    "Volume",
    "RSI",
    "MACD",
    "MACD_SIGNAL",
    "EMA_20",
    "EMA_50",
    "SMA_20",
    "Returns",
    "Volatility"
]].tail(1)



# =====================================
# PREDICTION
# =====================================

prediction = model.predict(
    latest
)



# =====================================
# CURRENT PRICE
# =====================================

current_price = (
    df["Close"].iloc[-1]
)



# =====================================
# PREDICTED PRICE
# =====================================

predicted_price = (
    prediction[0]
)



# =====================================
# DIFFERENCE
# =====================================

difference = (
    predicted_price - current_price
)



# =====================================
# METRICS
# =====================================

st.metric(
    "Current BTC Price",
    f"${current_price:,.2f}"
)

st.metric(
    "Predicted Next Price",
    f"${predicted_price:,.2f}"
)

st.metric(
    "Prediction Difference",
    f"${difference:,.2f}"
)



# =====================================
# SIGNAL
# =====================================

if predicted_price > current_price:

    st.success(
        "AI Prediction: BTC may RISE 📈"
    )

else:

    st.error(
        "AI Prediction: BTC may FALL 📉"
    )



# =====================================
# CLOSE PRICE CHART
# =====================================

st.subheader(
    "BTC Close Price"
)

st.line_chart(
    df["Close"]
)



# =====================================
# RSI CHART
# =====================================

st.subheader(
    "RSI"
)

st.line_chart(
    df["RSI"]
)



# =====================================
# MACD CHART
# =====================================

st.subheader(
    "MACD"
)

st.line_chart(
    df[[
        "MACD",
        "MACD_SIGNAL"
    ]]
)



# =====================================
# LATEST DATA
# =====================================

st.subheader(
    "Latest BTC Data"
)

st.dataframe(
    df.tail()
)
