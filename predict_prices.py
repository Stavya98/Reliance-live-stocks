import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load data, model, and scaler
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

model = load_model('stock_model.h5')
scaler = joblib.load('scaler.save')

# Normalize data
scaled_data = scaler.transform(data)

# Take last 60 days for prediction
last_60_days = scaled_data[-60:]

predictions = []
current_input = last_60_days.copy()

for _ in range(5):
    pred = model.predict(current_input.reshape(1, 60, 1), verbose=0)
    predictions.append(pred[0, 0])
    current_input = np.append(current_input[1:], pred, axis=0)

# Inverse scale predictions
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Last 5 actual prices
last_5_days = data[-5:]

# Save both to CSV
result_df = pd.DataFrame({
    'Day': ['Day-5', 'Day-4', 'Day-3', 'Day-2', 'Day-1', 
            'Day+1', 'Day+2', 'Day+3', 'Day+4', 'Day+5'],
    'Price': np.concatenate((last_5_days.flatten(), predictions.flatten()))
})

result_df.to_csv('predicted_prices.csv', index=False)
print("✅ Predictions saved to 'predicted_prices.csv'")
