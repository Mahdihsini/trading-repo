# trading-repo
trading
"""
Trading Prompts & Simple Backtester
-----------------------------------

GitHub Repository: trading-prompts-and-strategy
Author: Your Name
License: MIT
This repository includes:
1. A collection of English prompts about financial markets & trading.
2. A simple Python script for backtesting a trading strategy (SMA crossover + RSI).
"""
trade
# ================================
# Section 1: Prompts
# ================================

financial_prompts = [
    # General Financial Markets
    "Explain how global financial markets are interconnected and how a crisis in one country can impact others.",
    "Discuss the role of central banks in stabilizing financial markets.",
    "What are the key differences between developed and emerging financial markets?",

    # Stock Market
    "Analyze the factors that influence stock price movements in the short term vs. long term.",
    "Compare active trading vs. passive investing strategies in the stock market.",
    "Explain how market sentiment and psychology affect stock market trends.",

    # Forex & Commodities
    "Describe the main factors that drive currency exchange rates in the forex market.",
    "How do geopolitical events influence commodity prices such as oil and gold?",
    "Discuss the risks and opportunities of trading in the forex market.",

    # Risk & Investment
    "What role does diversification play in reducing risk in financial markets?",
    "Evaluate the impact of financial derivatives on market stability.",
    "Discuss ethical investing and the rise of ESG funds in modern markets.",

    # Trading-specific
    "Explain the difference between day trading, swing trading, and position trading.",
    "What are the psychological challenges traders face, and how can they overcome them?",
    "Discuss the importance of risk management in trading and provide practical strategies.",
    "Compare technical analysis and fundamental analysis in trading. Which one is more effective and why?",
    "Explain how candlestick patterns can be used to predict price movements.",
    "Describe how economic news releases impact short-term trading decisions.",
    "What are the pros and cons of algorithmic trading compared to manual trading?",
    "Discuss how traders use moving averages and RSI to identify entry and exit points.",
    "Explain the concept of leverage in trading and its risks and rewards.",
    "How do fear and greed drive market trends, and how can traders avoid emotional decisions?",
    "Discuss the role of stop-loss and take-profit orders in professional trading.",
    "What are the most common mistakes beginner traders make, and how can they avoid them?",
]

# ================================
# Section 2: Simple Backtest Script
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_fake_data(n=300):
    """Generate random walk stock prices for demo."""
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, n)
    price = 100 + np.cumsum(returns)
    return pd.DataFrame({"Close": price})

def compute_indicators(df, short=20, long=50, rsi_period=14):
    """Compute SMA and RSI indicators."""
    df["SMA_short"] = df["Close"].rolling(short).mean()
    df["SMA_long"] = df["Close"].rolling(long).mean()

    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(rsi_period).mean()
    avg_loss = pd.Series(loss).rolling(rsi_period).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def backtest(df):
    """Simple SMA crossover + RSI filter strategy."""
    df["Signal"] = 0
    df.loc[(df["SMA_short"] > df["SMA_long"]) & (df["RSI"] < 70), "Signal"] = 1
    df.loc[(df["SMA_short"] < df["SMA_long"]) & (df["RSI"] > 30), "Signal"] = -1

    df["Position"] = df["Signal"].shift(1).fillna(0)
    df["Returns"] = df["Close"].pct_change()
    df["Strategy"] = df["Position"] * df["Returns"]

    total_return = (1 + df["Strategy"]).prod() - 1
    print(f"Total Strategy Return: {total_return:.2%}")
    return df

def plot_results(df):
    """Plot price, SMA, and signals."""
    plt.figure(figsize=(12,6))
    plt.plot(df["Close"], label="Price", alpha=0.7)
    plt.plot(df["SMA_short"], label="SMA Short", alpha=0.7)
    plt.plot(df["SMA_long"], label="SMA Long", alpha=0.7)

    buy_signals = df[df["Signal"] == 1]
    sell_signals = df[df["Signal"] == -1]
    plt.scatter(buy_signals.index, buy_signals["Close"], marker="^", color="g", label="Buy Signal", alpha=0.9)
    plt.scatter(sell_signals.index, sell_signals["Close"], marker="v", color="r", label="Sell Signal", alpha=0.9)

    plt.title("SMA Crossover + RSI Strategy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    data = generate_fake_data()
    data = compute_indicators(data)
    data = backtest(data)
    plot_results(data)
if ---- mahdi
plt.figur
if i were you ... i will ...
hello
have a great day
kebfwbel
