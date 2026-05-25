import pandas as pd
import numpy as np
import yfinance as yf
import ta
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)



# =========================================
# DOWNLOAD BTC DATA
# =========================================

print("\nDownloading BTC data...\n")

df = yf.download(
    "BTC-USD",
    period="1000d",
    interval="1d"
)



# =========================================
# RESET INDEX
# =========================================

df.reset_index(inplace=True)



# =========================================
# KEEP NEEDED COLUMNS
# =========================================

df = df[[
    "Date",
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

print("Creating technical indicators...\n")



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
# TARGET COLUMN
# =========================================

print("Creating target column...\n")

# Tomorrow BTC price
df["Target"] = (
    df["Close"].shift(-1)
)



# =========================================
# DROP NAN VALUES
# =========================================

df.dropna(inplace=True)



# =========================================
# FEATURES
# =========================================

X = df[[
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
]]



# =========================================
# TARGET
# =========================================

y = df["Target"]



# =========================================
# TRAIN TEST SPLIT
# =========================================

print("Splitting dataset...\n")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)



# =========================================
# MODEL
# =========================================

print("Training Linear Regression model...\n")

model = LinearRegression()



# =========================================
# TRAIN MODEL
# =========================================

model.fit(
    X_train,
    y_train
)



# =========================================
# PREDICTIONS
# =========================================

predictions = model.predict(
    X_test
)



# =========================================
# EVALUATION
# =========================================

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    predictions
)



# =========================================
# PRINT RESULTS
# =========================================

print("\n========== MODEL RESULTS ==========\n")

print(f"MAE  : {mae:.2f}")

print(f"MSE  : {mse:.2f}")

print(f"RMSE : {rmse:.2f}")

print(f"R2   : {r2:.4f}")



# =========================================
# SAVE MODEL
# =========================================

pickle.dump(
    model,
    open(
        "btc_regression_model.pkl",
        "wb"
    )
)



print("\nModel saved successfully!\n")



# =========================================
# FUTURE PREDICTION
# =========================================

latest = X.tail(1)

future_price = model.predict(
    latest
)



print(
    f"Predicted Next BTC Price: "
    f"${future_price[0]:,.2f}"
)



# =========================================
# FEATURE IMPORTANCE
# =========================================

importance = pd.DataFrame({

    "Feature": X.columns,

    "Coefficient": model.coef_

})



print("\n========== FEATURE IMPORTANCE ==========\n")

print(
    importance.sort_values(
        by="Coefficient",
        ascending=False
    )
)
