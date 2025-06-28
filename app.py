from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def index():
    # Load prediction data
    df = pd.read_csv('predicted_prices.csv')

    # Plot price trend graph
    plt.figure(figsize=(10, 5))
    plt.plot(df['Day'], df['Price'], marker='o')
    plt.title('Reliance Stock Price Trend')
    plt.xlabel('Day')
    plt.ylabel('Price (₹)')
    plt.grid(True)
    plt.savefig('static/graph.png')
    plt.close()

    # Send data to frontend
    price_data = df.to_dict(orient='records')
    return render_template('index.html', price_data=price_data)

if __name__ == '__main__':
    app.run(debug=True)
