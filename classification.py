import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import yfinance as yf

# Download data
aapl = yf.Ticker("AAPL")
aapl_5y = aapl.history(period="5y")
print(aapl_5y.head(10))

# Visualize Data
plt.plot(aapl_5y.index, aapl_5y['Close'])
plt.title("AAPL Closing Price over Time")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.show()

# Add target variable
today = {'Close': 0}
aapl_5y = aapl_5y.append(today, ignore_index = True)
df_stock = aapl_5y[['Close']]
df_stock['Up/Down'] = df_stock['Close'].rolling(2).apply(lambda x: 1 if x.iloc[1] > x.iloc[0] else 0)

# Add feature variables
df_stock = df_stock.rename(columns = {'Close': 'Real Close'})
aapl_hist_shift = aapl_5y.copy()
aapl_hist_shift1 = aapl_hist_shift.shift(1)
aapl_hist_shift2 = aapl_hist_shift.shift(2)
variables = ['Open', 'High', 'Low', 'Close', 'Volume']
df_stock = df_stock.join(aapl_hist_shift1[variables])
df_stock = df_stock.iloc[1:]
df_stock = df_stock.rename(columns = {'Open': 'Previous Open', 'High': 'Previous High', 'Low': 'Previous Low',
                                      'Close': 'Previous Close', 'Volume': 'Previous Volume'})
df_stock['Previous Close to Open Ratio'] = df_stock['Previous Close'] / df_stock['Previous Open']
df_stock['Previous High to Close Ratio'] = df_stock['Previous High'] / df_stock['Previous Close']
df_stock['Previous Low to Close Ratio'] = df_stock['Previous Low'] / df_stock['Previous Close']
weekly_mean = df_stock.rolling(5).mean()
weekly_mean = weekly_mean.dropna()
monthly_mean = df_stock.rolling(22).mean()
monthly_mean = monthly_mean.dropna()
weekly_trend = df_stock['Up/Down'].shift(1).rolling(5).mean()
df_stock['Previous Weekly Mean to Close Ratio'] = weekly_mean['Previous Close'] / df_stock['Previous Close']
df_stock['Previous Monthly Mean to Close Ratio'] = monthly_mean['Previous Close'] / df_stock['Previous Close']
df_stock['Previous Weekly Trend'] = weekly_trend
df_stock = df_stock.dropna()
print(df_stock)

# Assign predictor variables
predictor_variables = ['Previous Close to Open Ratio',
                       'Previous High to Close Ratio', 'Previous Low to Close Ratio',
                       'Previous Weekly Mean to Close Ratio', 'Previous Monthly Mean to Close Ratio',
                       'Previous Weekly Trend']
print(df_stock['Up/Down'].value_counts())

# Create and train model and predict prices using model
model = RandomForestClassifier(n_estimators = 150, min_samples_split = 50, random_state = 0)
train = df_stock.iloc[:1000]
test = df_stock.iloc[1000:]
model.fit(train[predictor_variables], train['Up/Down'])

# Precision without backtesting algorithm
predictions = model.predict(test[predictor_variables])
predictions = pd.Series(predictions, index=test.index)
print(predictions)
precision = precision_score(test['Up/Down'], predictions)
print(precision)

# backtest(df, model, predictor_variables) backtests model on df with predictor_variables
def backtest(df, model, predictor_variables):
    start = 200
    step = 25
    predictions_accumulator = []
    test_accumulator = []
    total_tests = df[start:]

    for i in range(start, df.shape[0], step):
        train = df.iloc[0:i].copy()
        test = df.iloc[i:(i + step)].copy()
        model.fit(train[predictor_variables], train['Up/Down'])
        partial_predictions = model.predict_proba(test[predictor_variables])[:, 1]
        partial_predictions[partial_predictions > 0.6] = 1
        partial_predictions[partial_predictions <= 0.6] = 0
        partial_predictions = partial_predictions.tolist()
        for j in partial_predictions:
            predictions_accumulator.append(j)
        for j in test['Up/Down']:
            test_accumulator.append(j)

    predictions_accumulator = pd.Series(predictions_accumulator, index = total_tests.index)
    test_accumulator = pd.Series(test_accumulator, index = total_tests.index)
    return pd.concat({'Up/Down': test_accumulator, 'Predictions': predictions_accumulator}, axis=1)

# Precision with backtesting algorithm
total_predictions = backtest(df_stock, model, predictor_variables)
print(total_predictions)
print(total_predictions['Predictions'].value_counts())
total_precision = precision_score(total_predictions['Up/Down'].iloc[:-1], total_predictions['Predictions'].iloc[:-1])
print(total_precision)

