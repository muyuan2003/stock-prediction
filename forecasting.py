import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
from pmdarima.arima import auto_arima

# price_prediction(stock_ticker_symbol) predicts stock prices of stock_ticker_symbol for the next 2 weeks
# using ARIMA, a time-series forecasting model
def price_prediction(stock_ticker_symbol) :
    stock_ticker = yf.Ticker(stock_ticker_symbol)
    df_stock = stock_ticker.history(period="2y")
    df_stock = df_stock.to_period('D')
    past_data = df_stock['Close']
    model = auto_arima(past_data, d = 1, start_p = 1, start_q = 1, max_p = 5, max_q = 5, seasonal=True, m = 5,
                       D = 1, start_P = 0, start_Q = 0, max_P = 5, max_Q= 5)
    prediction = pd.DataFrame(model.predict(n_periods = 10))
    prediction = prediction.round(2)
    time_series = pd.date_range(start = dt.datetime.today().strftime('%Y-%m-%d'), periods = 11, freq = 'B')
    time_series = time_series[1:]
    prediction['time'] = time_series
    prediction.columns = ['Price', 'time']
    prediction = prediction.set_index('time')
    return prediction

