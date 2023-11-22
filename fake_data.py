import pandas as pd
import numpy as np
import statsmodels.api as sm

def simulate_stock_prices(days, start_price=100, volatility=0.01):
    # Randomly choose daily returns
    daily_returns = np.random.normal(0, volatility, days)

    # Calculate the price data
    price_data = start_price * (1 + daily_returns).cumprod()

    # Create a pandas DataFrame
    dates = pd.date_range(start=pd.to_datetime("today"), periods=days)
    stock_prices = pd.DataFrame(price_data, index=dates, columns=["Price"])

    return stock_prices

# Simulate data
data = simulate_stock_prices(365)

# Create a lagged price column
data['Lagged_Price'] = data["Price"].shift(1)

# Drop the NaN values created by the shift
data = data.dropna()

# Define the dependent (y) and independent (X) variables
y = data["Price"]
X = data["Lagged_Price"]

# Add a constant to the independent variables matrix (for intercept)
X = sm.add_constant(X)

# Create the model and fit in
model = sm.OLS(y, X).fit()

# Print the summary of the regression
print(model.summary())
