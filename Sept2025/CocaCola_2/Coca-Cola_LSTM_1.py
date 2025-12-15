import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ======================
# DATA FETCH
# ======================
ticker = "KO"
data = yf.download(ticker, start="2015-01-01", end="2023-12-31")

# 🔑 FIX MULTIINDEX
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

close_prices = data[['Close']].values

# ======================
# SCALING
# ======================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# ======================
# CREATE SEQUENCES
# ======================
def create_sequences(data, window=60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)
X = X.reshape(X.shape[0], X.shape[1], 1)

# ======================
# TRAIN / TEST SPLIT
# ======================
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ======================
# LSTM MODEL
# ======================
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32)

# ======================
# PREDICTION
# ======================
predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted)

print("Next Day Predicted Price (LSTM):", predicted_prices[-1][0])
