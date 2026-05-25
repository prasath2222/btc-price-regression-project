import pandas as pd
import yfinance as yf
import ta
import pickle



# =========================================
# LOAD MODEL
# =========================================

model = pickle.load(
    open(
        "btc_regression_model.pkl",
        "rb"
    )
)



# =========================================
# DOWNLOAD BTC DATA
# =========================================

print("\nDownloading latest BTC data...\n")

df = yf.download(
    "BTC-USD",
    period="300d",
    interval="1d"
)



# =========================================
# RESET INDEX
# =========================================

df.reset_index(inplace=True)



# =========================================
# KEEP COLUMNS
# =========================================

df = df[[
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]]



# =========================================
# FLOAT CONVERSION
# =========================================

for col in [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]:

    df[col] = df[col].astype(float)



# =========================================
# TECHNICAL INDICATORS
# =========================================

print("Creating indicators...\n")



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



# =========================================
# CLEAN DATA
# =========================================

df.dropna(inplace=True)



# =========================================
# FEATURES
# =========================================

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



# =========================================
# PREDICT
# =========================================

prediction = model.predict(
    latest
)



# =========================================
# CURRENT PRICE
# =========================================

current_price = (
    df["Close"].iloc[-1]
)



# =========================================
# PRICE DIFFERENCE
# =========================================

difference = (
    prediction[0] - current_price
)



# =========================================
# RESULTS
# =========================================

print("\n========== BTC PRICE PREDICTION ==========\n")

print(
    f"Current BTC Price : "
    f"${current_price:,.2f}"
)

print(
    f"Predicted Price   : "
    f"${prediction[0]:,.2f}"
)

print(
    f"Difference        : "
    f"${difference:,.2f}"
)



# =========================================
# MARKET SIGNAL
# =========================================

if prediction[0] > current_price:

    print("\nPrediction: BTC may RISE 📈\n")

else:

    print("\nPrediction: BTC may FALL 📉\n")
