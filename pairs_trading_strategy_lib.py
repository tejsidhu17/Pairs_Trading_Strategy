import yfinance as yf
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.stattools import adfuller


def get_historical_data(tickers, period):
    data = pd.DataFrame()
    names = list()

    for tick in tickers:
        ticker_data =  yf.download(tick, period=period)
        data = pd.concat([data, ticker_data["Adj Close"]], axis=1) 
        names.append(tick)

    data.columns = names
    return data

def create_correlation_heatmap(data):
    corr_matrix = data.corr()
    plt.figure(figsize=(10, 8), dpi=200)
    sb.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()

def plot_spreads(data, pair):
    plt.figure(figsize=(8,6), dpi=200)
    spread = data[pair[0]] - data[pair[1]]
    plt.plot(spread, label=f"Price Spread {pair[0]} vs {pair[1]}")
    plt.axhline(spread.mean(), color="red")
    plt.legend()
    plt.title(f"Price Spread between {pair[0]} and {pair[1]}")
    return spread  

def plot_ratio(data, pair):
    plt.figure(figsize=(8,6), dpi=200)
    ratio = data[pair[0]]/data[pair[1]]
    plt.plot(ratio, label=f"Price Ratio {pair[0]} vs {pair[1]}")
    plt.axhline(ratio.mean(), color="red")
    plt.legend()
    plt.title(f"Price Ratio between {pair[0]} and {pair[1]}")
    return ratio

def cointegration_test(data, tuple):
    engle_granger_p_val = (ts.coint(data[tuple[0]], data[tuple[1]]))[1]
    spread_adf_p_val = (adfuller(data[tuple[0]] - data[tuple[1]]))[1]
    ratio_adf_p_val = (adfuller(data[tuple[0]]/data[tuple[1]]))[1]
    print(f"Engle Granger Test: {engle_granger_p_val}")
    print(f"AD Fuller Test Spread: {spread_adf_p_val}")
    print(f"AD Fuller Test Ratio: {ratio_adf_p_val}")

def plot_zscore(comp_method, critical_zscores, pair):
    plt.figure(figsize=(10, 6), dpi=200)
    z_score = (comp_method - comp_method.mean())/comp_method.std()
    plt.plot(z_score, label="Z-Scores")
    for critical in critical_zscores:
        if(critical > 0.0):
            plt.axhline(critical, color="red")
        else:
            plt.axhline(critical, color="green")
    plt.legend(loc="best")
    plt.title(f"Z-score ratio between {pair[0]} and {pair[1]}")
    return z_score

def develop_strategy(comp_method, comp_method_type, z_score, critical_buy, critical_sell, pair):
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(comp_method)
    buy=comp_method.copy()
    sell=comp_method.copy()

    buy[z_score > critical_buy] = 0
    sell[z_score < critical_sell] = 0

    plt.plot(buy, color="green", linestyle="None", marker="^")
    plt.plot(sell, color="red", linestyle="None", marker="^")

    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, comp_method.min(), comp_method.max()))
    plt.legend([f"{comp_method_type}", "Buy Signal", "Sell Signal"])
    plt.title(f"Relationship {pair[0]} and {pair[1]}")
    plt.show()