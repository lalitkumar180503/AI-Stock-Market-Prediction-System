import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title("AI-Based Stock Market Prediction System")

stock = st.text_input("Enter Stock Symbol", "AAPL")

df = yf.download(stock, start="2023-01-01", end="2025-01-01")

st.subheader("Stock Data")

st.write(df.head())

fig, ax = plt.subplots(figsize=(12,6))

ax.plot(df['Close'])

ax.set_title("Closing Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price")

st.pyplot(fig)
# Moving Average
df['MA_10'] = df['Close'].rolling(window=10).mean()
df['MA_50'] = df['Close'].rolling(window=50).mean()

st.subheader("Moving Averages")

fig2, ax2 = plt.subplots(figsize=(12,6))

ax2.plot(df['Close'], label='Close Price')
ax2.plot(df['MA_10'], label='MA 10')
ax2.plot(df['MA_50'], label='MA 50')

ax2.legend()

st.pyplot(fig2)

# Prediction Column
# Prediction Column
df['Prediction'] = df['Close'].shift(-1)

df.dropna(inplace=True)

# Features and Target
X = df[['Close']].to_numpy().reshape(-1, 1)

y = df['Prediction'].to_numpy()

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()

model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

st.subheader("Model Accuracy")

st.write(accuracy)
current_price = float(df['Close'].iloc[-1])
st.write(f"Current Price: {current_price:.2f}")

# Future Prediction
latest_price = np.array([[float(df['Close'].iloc[-1])]])

future_price = model.predict(latest_price)

predicted_price = float(future_price[0])

current_price = float(df['Close'].iloc[-1])

st.subheader("Predicted Future Price")

st.write(f"Predicted Next Day Price: {predicted_price:.2f}")

# Buy/Sell Signal
if predicted_price > current_price:
    st.success("BUY SIGNAL")
else:
    st.error("SELL SIGNAL")
returns = df['Close'].pct_change()
risk = returns.std()

st.subheader("Risk Analysis")

st.write("Risk (Standard Deviation):", risk)