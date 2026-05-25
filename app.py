import streamlit as st
from streamlit_autorefresh import st_autorefresh

import pandas as pd
import ta
import pickle
import yfinance as yf



# =====================================
# AUTO REFRESH
# =====================================

st_autorefresh(
    interval=300000,
    key="btc_refresh"
)



# =====================================
# PAGE TITLE
# =====================================

st.title("BTC Price Regression Dashboard")



# =====================================
# LOAD MODEL
# =====================================

model = pickle.load(
    open("btc_regression_model.pkl", "rb")
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
# FIX MULTI INDEX
# =====================================

if isinstance(df.columns, pd.MultiIndex):

    df.columns = df.columns.get_level_values(0)



# =====================================
# RESET INDEX
# =====================================

df.reset_index(inplace=True)



# =====================================
# KEEP COLUMNS
# =====================================

df = df[[
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]]



# =====================================
# FLOAT CONVERSION
# =====================================

for col in df.columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )



# =====================================
# TECHNICAL INDICATORS
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
# DROP NAN
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

predicted_price = model.predict(
    latest
)[0]



# =====================================
# CURRENT PRICE
# =====================================

current_price = float(
    df["Close"].iloc[-1]
)



# =====================================
# PRICE DIFFERENCE
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
    "Predicted Next BTC Price",
    f"${predicted_price:,.2f}"
)

st.metric(
    "Expected Change",
    f"${difference:,.2f}"
)



# =====================================
# PREDICTION RESULT
# =====================================

if predicted_price > current_price:

    st.success(
        "AI predicts BTC may rise"
    )

else:

    st.error(
        "AI predicts BTC may fall"
    )



# =====================================
# CHARTS
# =====================================

st.subheader("BTC Close Price")

st.line_chart(
    df["Close"]
)



st.subheader("RSI")

st.line_chart(
    df["RSI"]
)



st.subheader("MACD")

st.line_chart(
    df[[
        "MACD",
        "MACD_SIGNAL"
    ]]
)



# =====================================
# TABLE
# =====================================

st.subheader(
    "Latest BTC Data"
)

st.dataframe(
    df.tail()
)
