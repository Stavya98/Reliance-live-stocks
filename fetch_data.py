import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date, file_name):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.to_csv(file_name)
    print(f"✅ Data for {ticker} saved to {file_name}")

if __name__ == "__main__":
    ticker = "RELIANCE.NS"
    start_date = "2010-01-01"
    end_date = "2025-06-28"
    file_name = "reliance_stock_data.csv"

    fetch_stock_data(ticker, start_date, end_date, file_name)
