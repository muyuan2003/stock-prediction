import yfinance as yf
import beta
import forecasting

stocks_list = []

# Input the ticker symbols of stocks, and the output will display each inputted stock's beta and price prediction
# for the next 2 weeks
while True:
    stock_input = input("Which stock would you like to know more about? Write 'done' if you have no more to add.")
    if stock_input.lower() == "done":
        break
    else:
        stocks_list.append(stock_input)

for stock in stocks_list:
    try:
        stock_ticker = yf.Ticker(stock)
        stock_hist = stock_ticker.history(period="5d")

        if stock_hist.empty:
            raise Exception
        else:
            print(f"Beta for {stock.upper()}:")
            print(beta.beta_calculation(stock))
            print(f"Stock price predictions for {stock.upper()} for the next 2 weeks:")
            print(forecasting.price_prediction(stock))
            print(" ")

    except:
        print(f"{stock.upper()} does not exist.")


