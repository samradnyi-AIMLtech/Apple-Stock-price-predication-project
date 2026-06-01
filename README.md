# Apple-Stock-price-predication-project

# Imports
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Conv1D, Dense, Flatten, Input, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Parameters
SEQ_LEN = 60       # Past days
FORECAST_HORIZON = 5  # Predict next 5 days

#Download OHLCV Data
df = yf.download("AAPL", start="2015-01-01", end="2024-12-31")
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

#Normalize
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

#Create Multi-step Sequences
def create_multistep_sequences(data, seq_len=60, horizon=5):
    X, y = [], []
    for i in range(len(data) - seq_len - horizon):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+horizon, 3])  # Close column index=3
    return np.array(X), np.array(y)

X, y = create_multistep_sequences(scaled, SEQ_LEN, FORECAST_HORIZON)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

#Model Builders
def build_lstm():
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
        LSTM(64),
        Dense(FORECAST_HORIZON)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru():
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
        GRU(64),
        Dense(FORECAST_HORIZON)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_cnn():
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(SEQ_LEN, X.shape[2])),
        Conv1D(64, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(FORECAST_HORIZON)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_transformer():
    inputs = Input(shape=(SEQ_LEN, X.shape[2]))
    x = LayerNormalization()(inputs)
    x = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(FORECAST_HORIZON)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

#Train and Evaluate
def train_and_evaluate(model_fn, name):
    print(f"\nTraining {name}...")
    model = model_fn()
    es = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=30, batch_size=32,
              validation_split=0.1, callbacks=[es], verbose=0)

    preds = model.predict(X_test)

    # Inverse transform Close only
    dummy_preds = np.zeros((preds.shape[0]*FORECAST_HORIZON, scaled.shape[1]))
    dummy_preds[:, 3] = preds.flatten()
    pred_prices = scaler.inverse_transform(dummy_preds)[:, 3].reshape(preds.shape)

    dummy_true = np.zeros((y_test.shape[0]*FORECAST_HORIZON, scaled.shape[1]))
    dummy_true[:, 3] = y_test.flatten()
    true_prices = scaler.inverse_transform(dummy_true)[:, 3].reshape(y_test.shape)

    # Plot
    plt.figure(figsize=(10, 4))
    for i in range(5):
        plt.plot(true_prices[i], label=f"True {i+1}d")
        plt.plot(pred_prices[i], linestyle='--', label=f"Pred {i+1}d")
    plt.title(f"{name} - Multi-step Forecast (5 Days)")
    plt.xlabel("Forecast Day")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.show()

#Run All Models
train_and_evaluate(build_lstm, "LSTM")
train_and_evaluate(build_gru, "GRU")
train_and_evaluate(build_cnn, "CNN")
train_and_evaluate(build_transformer, "Transformer")
