import yfinance as yf
from datetime import datetime

def get_option_data(symbol, option_type, expiration_date):
    # Convert expiration date to datetime object
    expiration_datetime = datetime.strptime(expiration_date, "%Y-%m-%d")

    # Create a Ticker object for the underlying symbol
    ticker = yf.Ticker(symbol)

    # Get the option chain for the specified expiration date
    option_chain = ticker.option_chain(expiration_date)

    # Get the data for the specified option type (call or put)
    if option_type == "call":
        option_data = option_chain.calls
    elif option_type == "put":
        option_data = option_chain.puts
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

    return option_data

if __name__ == "__main__": 

   option_data = get_option_data("AAPL", "call", "2023-04-21")
   print(option_data)