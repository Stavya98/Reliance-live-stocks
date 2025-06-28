import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import joblib

# Load data
df = pd.read_csv('reliance_stock_data.csv')

df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna(subset=['Close'])

df['High'] = pd.to_numeric(df['High'], errors='coerce')
df = df.dropna(subset=['High'])

df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
df = df.dropna(subset=['Low'])

df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
df = df.dropna(subset=['Open'])

df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
df = df.dropna(subset=['Volume'])

data = df['Close'].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Save scaler for later
joblib.dump(scaler, 'scaler.save')

# Prepare data for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)

# Reshape to 3D [samples, time_steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train
history = model.fit(X, y, epochs=25, batch_size=32)

# Save model
#model.save('stock_model.h5')
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Predict on training data
predicted = model.predict(X)

# Inverse transform to get actual prices
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(y.reshape(-1, 1))

# Calculate metrics
mse = mean_squared_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_prices, predicted_prices)
r2 = r2_score(actual_prices, predicted_prices)

print(f"📈 RMSE: {rmse:.4f}")
print(f"📉 MAE: {mae:.4f}")
print(f"📊 R² Score: {r2:.4f}")
print("✅ Model trained and saved as 'stock_model.h5'")

plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
