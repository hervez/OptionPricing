import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dfply import *
import yfinance as yf
from ordered_set import OrderedSet
import random
from copy import copy
from yahoo_fin.stock_info import *
from yahoo_fin.options import *
import math
from datetime import date
from datetime import datetime

#Generating S&P 500 List of Stocks
sp_list = tickers_sp500()

get_expiration_dates(sp_list[0])

#Creating options dataset
expiration = get_expiration_dates(sp_list[0])[1]   #September date
options_df = pd.DataFrame()
for stock in sp_list:
   try:
      price = get_live_price(stock)           
      option = get_calls(stock, expiration)
      option['CurrentPrice'] = price
      option['Ticker'] = stock
      option['DateExp'] = expiration
      options_df = options_df.append(option)
      print(price)
      print(stock)
   except:
      print('Option contract not available on '+ str(expiration))
      pass

options_df = options_df[['Contract Name', 'Strike','Bid', 'Ask', 'CurrentPrice', 'Volume', 'Open Interest', 'Ticker', 'DateExp']]

options_df = options_df[options_df.Bid != '-']
options_df = options_df[options_df.Ask != '-']

options_df['AvgPrice'] = (pd.to_numeric(options_df.Bid) + pd.to_numeric(options_df.Ask)) / 2
options_df['OptionCost'] = options_df['AvgPrice'] * 100
options_df['BE_percent'] = ((options_df.Strike + options_df.AvgPrice)/(options_df.CurrentPrice)) - 1
     
print(options_df)