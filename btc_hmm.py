import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import MultinomialHMM
import pandas_ta as ta

# Load the dataset
data_path = "btc_data.csv"  # Replace with your actual file path
data = pd.read_csv(data_path)

# Ensure date column is datetime
data["date"] = pd.to_datetime(data["date"], format="%d/%m/%Y")

# Sort data by date
data.sort_values("date", inplace=True)

# Select a specific date range
start_date = "2018-02-01"  # Define your start date  #"2018-01-01"
end_date = "2024-12-31"    # Define your end date

# Filter data within the date range
data = data[(data["date"] >= start_date) & (data["date"] <= end_date)]

# Calculate features
# Log returns
data["log_return"] = np.log(data["close"] / data["close"].shift(1))

# RSI
data["RSI"] = ta.rsi(data["close"], length=14)

# MACD
macd = ta.macd(data["close"], fast=12, slow=26, signal=9)
data["MACD"] = macd["MACD_12_26_9"]
data["Signal_Line"] = macd["MACDs_12_26_9"]

# Moving averages
data["sma_50"] = data["close"].rolling(window=50).mean()
data["sma_100"] = data["close"].rolling(window=100).mean()

# Discretize features
def discretize_log_return(log_return):
    if log_return <= -0.05:
        return 1  # Extreme Negative
    elif -0.05 < log_return <= -0.005:
        return 2  # Negative
    elif -0.005 < log_return <= 0.005:
        return 3  # Low/Neutral
    elif 0.005 < log_return <= 0.05:
        return 4  # Positive
    else:
        return 5  # Extreme Positive

def discretize_rsi(rsi):
    if rsi < 30:
        return 1  # Oversold
    elif 30 <= rsi < 50:
        return 2  # Neutral
    elif 50 <= rsi <= 65:
        return 3  # Bullish
    else:
        return 4  # Overbought

def discretize_macd(macd, signal_line):
    if macd < signal_line and macd < -0.01:
        return 1  # Strong Bearish
    elif macd < signal_line and -0.01 <= macd < 0:
        return 2  # Weak Bearish
    elif abs(macd - signal_line) < 0.01:
        return 3  # Neutral
    elif macd > signal_line and 0 < macd <= 0.01:
        return 4  # Weak Bullish
    elif macd > signal_line and macd > 0.01:
        return 5  # Strong Bullish

def discretize_sma_crossover(sma_50, sma_100):
    gap = sma_50 - sma_100
    if sma_50 < sma_100 and gap < -0.01:
        return 1  # Strong Bearish
    elif sma_50 < sma_100 and -0.01 <= gap < 0:
        return 2  # Weak Bearish
    elif abs(gap) < 0.01:
        return 3  # Neutral
    elif sma_50 > sma_100 and 0 < gap <= 0.01:
        return 4  # Weak Bullish
    elif sma_50 > sma_100 and gap > 0.01:
        return 5  # Strong Bullish

def discretize_volume(volume, historical_volumes):
    quantiles = np.percentile(historical_volumes, [25, 50, 75, 90])
    if volume < quantiles[0]:
        return 1  # Low Volume
    elif quantiles[0] <= volume < quantiles[1]:
        return 2  # Below Average Volume
    elif quantiles[1] <= volume < quantiles[2]:
        return 3  # Average Volume
    elif quantiles[2] <= volume < quantiles[3]:
        return 4  # Above Average Volume
    else:
        return 5  # High Volume

def discretize_fear_greed(fng_value):
    if 0 <= fng_value <= 25:
        return 1  # Extreme Fear
    elif 26 <= fng_value <= 46:
        return 2  # Fear
    elif 47 <= fng_value <= 54:
        return 3  # Neutral
    elif 55 <= fng_value <= 75:
        return 4  # Greed
    elif 76 <= fng_value <= 100:
        return 5  # Extreme Greed

# Apply discretization
historical_volumes = data["volume"].dropna()
data["discretized_log_return"] = data["log_return"].apply(discretize_log_return)
data["discretized_rsi"] = data["RSI"].apply(discretize_rsi)
data["discretized_macd"] = data.apply(lambda row: discretize_macd(row["MACD"], row["Signal_Line"]), axis=1)
data["discretized_sma_crossover"] = data.apply(lambda row: discretize_sma_crossover(row["sma_50"], row["sma_100"]), axis=1)
data["discretized_volume"] = data["volume"].apply(lambda v: discretize_volume(v, historical_volumes))
data["discretized_fear_greed"] = data["fng_value"].apply(discretize_fear_greed)

# Prepare data for HMM
features = [
    "discretized_log_return",
    "discretized_rsi",
    "discretized_macd",
    "discretized_sma_crossover",
    "discretized_volume",
    "discretized_fear_greed"
]
model_data = data[features].dropna().astype(int)

# Fit the HMM
hmm_model = MultinomialHMM(n_components=8, n_iter=500, random_state=42)
hmm_model.fit(model_data)

# Predict hidden states
hidden_states = hmm_model.predict(model_data)

# Map hidden states back to the original dataset
data["hidden_state"] = np.nan
data.loc[model_data.index, "hidden_state"] = hidden_states

# Plot BTC/USD
plt.figure(figsize=(12, 6))
plt.plot(data["date"], data["close"], label="BTC/USD Price", alpha=0.7)
plt.legend()
plt.title("BTC/USD Close Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data["date"], data["close"], label="BTC/USD Price", alpha=0.7)
plt.plot(data["date"], data["sma_50"], label="50 SMA", alpha=0.7)
plt.plot(data["date"], data["sma_100"], label="100 SMA", alpha=0.7)
plt.scatter(
    data["date"],
    data["close"],
    c=data["hidden_state"],
    cmap="RdYlGn",
    label="Hidden States",
    s=10,
)
plt.colorbar(label="Hidden State")
plt.legend()
plt.title("BTC/USD Price with Hidden States")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

# Print transition and emission probabilities
print("Transition Matrix:")
print(hmm_model.transmat_)

print("\nEmission Matrix:")
print(hmm_model.emissionprob_)

# Ensure hidden states are integers for grouping
data["hidden_state"] = data["hidden_state"].astype("Int64")

# Group data by hidden state and calculate statistics
state_statistics = data.groupby("hidden_state").agg({
    "log_return": ["mean", "std", "min", "max"],
    "RSI": ["mean", "std", "min", "max"],
    "MACD": ["mean", "std", "min", "max"],
    "Signal_Line": ["mean", "std", "min", "max"],
    "sma_50": ["mean", "std", "min", "max"],
    "sma_100": ["mean", "std", "min", "max"],
    "volume": ["mean", "std", "min", "max"],
    "fng_value": ["mean", "std", "min", "max"]
})

# Display statistics
print("Hidden State Statistics:")
print(state_statistics)
