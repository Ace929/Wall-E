import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
import pytz

# Fetch historical stock market data (S&P 500)
ticker = "AAPL"  # S&P 500 Index
data = yf.download(ticker, start="2000-01-01", end="2025-01-01")
data.index = data.index.tz_localize(pytz.utc)  # Set timezone

# Compute daily log returns
data["Log_Returns"] = np.log(data["Adj Close"] / data["Adj Close"].shift(1))
data.dropna(inplace=True)

# Train Hidden Markov Model (HMM)
hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
X = data[["Log_Returns"]].values  # HMM needs a 2D array
hmm_model.fit(X)

# Predict the hidden market regimes
data["Regime"] = hmm_model.predict(X)

# Map regimes to labels
regime_labels = {0: "Bear Market", 1: "Bull Market"}
data["Regime_Label"] = data["Regime"].map(regime_labels)

# Plot Market Regimes
plt.figure(figsize=(14, 6))
sns.scatterplot(x=data.index, y=data["Adj Close"], hue=data["Regime_Label"], palette={"Bear Market": "red", "Bull Market": "green"}, alpha=0.6)
plt.title("Market Regime Detection using HMM")
plt.xlabel("Date")
plt.ylabel("S&P 500 Adjusted Close Price")
plt.legend(title="Market Regime")
plt.show()
