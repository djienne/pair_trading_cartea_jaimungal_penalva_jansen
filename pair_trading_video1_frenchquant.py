import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. Data Loading and Preparation
# ---------------------------------------------------------

# Define the tickers for the pair
tickers = ["GDX", "GOAU"]

# Download data using yfinance
# Note: The video starts data from 2018-01-01
prices = yf.download(tickers, start="2018-01-01")['Close']
prices = prices.dropna()

# Visualize the two assets
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(prices['GDX'], label='VanEck Gold Miners', color='blue')
ax2 = ax.twinx()
ax2.plot(prices['GOAU'], label='US Global GO GOLD', color='orange')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title("GDX vs GOAU Prices")
plt.show()

# ---------------------------------------------------------
# 2. Cointegration Test (OLS Regression)
# ---------------------------------------------------------

# Define dependent (Y) and independent (X) variables
# We try to express GDX as a linear combination of GOAU
Y = prices['GDX']
X = prices['GOAU']

# Add a constant to the independent variable (intercept)
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(Y, X)
res = model.fit()

# Print regression summary
print(res.summary())

# Calculate the residuals (epsilon)
# This represents the "spread" or the error term
# epsilon = Y - (alpha + beta * X)
epsilon = Y - (res.params['const'] + res.params['GOAU'] * prices['GOAU'])

# ---------------------------------------------------------
# 3. Stationarity Test (ADF Test)
# ---------------------------------------------------------

# Perform Augmented Dickey-Fuller test on residuals
adf = adfuller(epsilon)

print(f"ADF Statistic: {adf[0]}")
print(f"p-value: {adf[1]}")

# ---------------------------------------------------------
# 4. Strategy Signal Generation
# ---------------------------------------------------------

# Calculate statistics for the spread (epsilon)
mean_epsilon = np.mean(epsilon)
std_epsilon = np.std(epsilon)

# Plot epsilon with standard deviation bands
plt.figure(figsize=(12, 6))
plt.plot(epsilon, label='Epsilon')
plt.axhline(mean_epsilon, color='red', linestyle='--')
plt.axhline(mean_epsilon + std_epsilon, color='green', linestyle='--')
plt.axhline(mean_epsilon - std_epsilon, color='green', linestyle='--')
plt.legend()
plt.title("Spread (Epsilon) with +/- 1 Std Dev")
plt.show()

# Define Z-score (normalized spread)
# This helps in determining entry points
z_score = (epsilon - mean_epsilon) / std_epsilon

# Determine positions
# If spread is too low (<-1 std), we Long the spread (Buy GDX, Sell GOAU)
# If spread is too high (>1 std), we Short the spread (Sell GDX, Buy GOAU)
long_entry = z_score < -1
short_entry = z_score > 1

# Initialize positions DataFrame
portfolio_pos = pd.DataFrame(index=prices.index)
portfolio_pos['pos'] = np.nan

# Set signals
portfolio_pos.loc[long_entry, 'pos'] = 1
portfolio_pos.loc[short_entry, 'pos'] = -1

# Forward fill positions (hold until signal changes)
portfolio_pos['pos'] = portfolio_pos['pos'].fillna(method='ffill')

# ---------------------------------------------------------
# 5. Backtesting
# ---------------------------------------------------------

# Calculate the hedge ratio (beta from regression)
hedge_ratio = res.params['GOAU']

# Calculate simple returns for the portfolio components
# Portfolio Return approx = Return(Y) - HedgeRatio * Return(X)
portfolio_returns = prices['GDX'].pct_change() - hedge_ratio * prices['GOAU'].pct_change()

# Calculate Strategy Returns
# We shift positions by 1 day to avoid look-ahead bias (entering at Close based on Close signal)
backtest_ret = portfolio_pos['pos'].shift(1) * portfolio_returns

# Calculate Cumulative Returns
backtest_cum_ret = (1 + backtest_ret).cumprod() - 1

# ---------------------------------------------------------
# 6. Visualization of Results
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))
backtest_cum_ret.plot(label='Strategy Cumulative Returns', color='orange')
plt.legend()
plt.title("Backtest Cumulative Returns")
plt.show()

# Visualize Returns alongside Positions
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(backtest_cum_ret, label='Cumulative Returns', color='blue')
ax1.set_ylabel('Cumulative Returns')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(portfolio_pos['pos'], label='Position', color='orange', alpha=0.3)
ax2.set_ylabel('Position (1=Long, -1=Short)')
ax2.set_yticks([-1, 0, 1])
ax2.legend(loc='upper right')

plt.title("Portfolio Cumulative Returns and Positions")
plt.show()