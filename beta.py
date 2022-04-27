import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf

# beta_calculation(stock_ticker_symbol) calculates the beta, a measure of relative risk, of stock_ticker_symbol
def beta_calculation(stock_ticker_symbol):
    market_index = '^IXIC'
    market_symbol = yf.Ticker(market_index)
    df_market = market_symbol.history(period="2y")
    weekly_returns_market = df_market.resample('W').ffill().pct_change()
    weekly_returns_market = weekly_returns_market['Close']
    market_variance = weekly_returns_market.var()

    stock_ticker = yf.Ticker(stock_ticker_symbol)
    df_stock = stock_ticker.history(period="2y")
    weekly_returns_stock = df_stock.resample('W').ffill().pct_change()
    weekly_returns_stock = weekly_returns_stock['Close']

    covariance_table = pd.concat([weekly_returns_market, weekly_returns_stock], axis = 1)
    covariance = covariance_table.cov()
    covariance = covariance.iloc[0, 1].round(7)
    beta = covariance / market_variance
    beta = beta.round(3)
    return beta