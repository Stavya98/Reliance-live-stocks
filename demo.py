import pandas as pd

# Load the CSV
df = pd.read_csv('reliance_stock_data.csv')

# List of columns to convert
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

# Convert only these columns to numeric
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows where any of these columns are NaN
df_cleaned = df.dropna(subset=numeric_cols)

# Save cleaned data

print(df.describe())
print("✅ Cleaned CSV saved as 'reliance_stock_data_cleaned.csv'")