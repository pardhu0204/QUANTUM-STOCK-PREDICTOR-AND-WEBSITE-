from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

# Ensure chart directories exist
if not os.path.exists("static/charts"):
    os.makedirs("static/charts")

def generate_charts(stock, hours):
    # Download historical data
    data = yf.download(stock, period="1mo", interval="1h", auto_adjust=True)
    if data.empty:
        return None

    # Historical chart
    plt.figure(figsize=(10,5))
    plt.plot(data['Close'], label='Historical Price', color='cyan')
    plt.title(f'{stock} Historical Prices', color='white')
    plt.xlabel('Time', color='white')
    plt.ylabel('Price', color='white')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/charts/history.png")
    plt.close()

    # Forecast with Linear Regression
    data_reset = data.reset_index()
    X = np.arange(len(data_reset)).reshape(-1, 1)
    y = data_reset['Close'].values
    model = LinearRegression()
    model.fit(X, y)

    # Predict future prices
    future_index = np.arange(len(data_reset), len(data_reset) + hours).reshape(-1,1)
    future_prices = model.predict(future_index)

    # Forecast chart
    plt.figure(figsize=(10,5))
    plt.plot(data_reset.index, y, label='Historical', color='cyan')
    plt.plot(range(len(data_reset), len(data_reset)+hours), future_prices, label='Forecast', linestyle='--', color='yellow')
    plt.title(f'{stock} Forecast for {hours} hours', color='white')
    plt.xlabel('Time', color='white')
    plt.ylabel('Price', color='white')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/charts/forecast.png")
    plt.close()

    # Current price
    current_price = y[-1]

    # Calculate expected profit/loss and risk ratio
    expected_profit = float(np.max(future_prices) - current_price)
    expected_loss = float(current_price - np.min(future_prices))
    risk_ratio = round(expected_profit / expected_loss, 2) if expected_loss != 0 else float('inf')

    # Future hourly data
    future_data = []
    for i in range(hours):
        future_data.append({
            'Hour': i+1,
            'Price': round(float(future_prices[i]), 2)
        })

    result = {
        'direction': 'Up' if expected_profit > 0 else 'Down',
        'risk': round(expected_loss, 2),
        'profit': round(expected_profit, 2),
        'ratio': risk_ratio,
        'hours': hours,
        'future_data': future_data
    }

    return result

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    stock = None
    hours = 1
    if request.method == "POST":
        stock = request.form.get("stock").upper()
        hours = int(request.form.get("hours"))
        result = generate_charts(stock, hours)
    return render_template("index.html", result=result, stock=stock)

if __name__ == "__main__":
    app.run(debug=True)
