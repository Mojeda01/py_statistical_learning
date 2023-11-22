import pandas as pd
import numpy as np

def simulate_stock_prices(days, start_price=100, volatility=0.01):
    # Randomly choose daily returns
    daily_returns = np.random.normal(0, volatility, days)

    # Calculate the price data
    price_data = start_price * (1 + daily_returns).cumprod()

    # Create a pandas DataFrame
    dates = pd.date_range(start=pd.to_datetime("today"), periods=days)
    stock_prices = pd.DataFrame(price_data, index=dates, columns=["Price"])

    return stock_prices

