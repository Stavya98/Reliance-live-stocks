import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# Fetch today's stock price
ticker = "RELIANCE.NS"
today = datetime.today().strftime('%Y-%m-%d')

# Download only today's data
data = yf.download(ticker, start=today, end=today)

if not data.empty:
    # Load existing CSV
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
    # Prepare new row
    new_row = {
        'Date': today,
        'Open': data['Open'].values[0],
        'High': data['High'].values[0],
        'Low': data['Low'].values[0],
        'Close': data['Close'].values[0],
        'Volume': data['Volume'].values[0]
    }

    # Append to CSV
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv('reliance_stock_data.csv', index=False)

    print(f"✅ Today's data ({today}) appended successfully.")

    # Re-run prediction
    os.system('python predict_prices.py')
    print("✅ Predictions updated successfully.")

else:
    print("No trading data available for today (maybe market closed).")
