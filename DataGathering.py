import yfinance as yf
from datetime import datetime
import os.path
import re 
import pandas as pd
from Utils import *
from datetime import datetime
import numpy as np

class OptionDataGathering(): 
   """ Class to gather the informations relative to an option type, at a given date for a given underlying in a dataframe. 
   If save_data is true, the the data is automatically saved as a CSV file in the appropriate file structure. """

   def __init__(self, 
                underlying: str, 
                option_type: str,
                expiration_date: str, 
                save_data: bool = True,
                verbose: bool = True): 

      self.underlying = underlying 
      if option_type == 'call' or option_type == 'put':
         self.option_type = option_type 
      else:
         raise ValueError('Invalid option type. Must be "call" or "put".')
      self.expiration_date = expiration_date 
      self.save_data = save_data 
      self.verbose = verbose
      self.file_path = './results/{}/{}_{}'.format(self.underlying, self.option_type, self.expiration_date)

      if self.save_data: 
         folder_creation('results', self.verbose)

      self.data = self.get_option_data(self.underlying, self.option_type, self.expiration_date)
      self.underlying_data = self.get_underlying_data(self.underlying, '10y')
      self.risk_free_rate = self.get_risk_free_rate()

   def download_underlying_data(self, symbol: str, period: str): 
      """ Download a dataframe containing the historical data of a given stock """ 

      ticker = yf.Ticker(symbol)

      underlying_data = ticker.history(period)

      return underlying_data
   
   def get_underlying_data(self, symbol:str, period: str): 
      """Get the underlying data, either from the web or the locale memory if they have already been queried."""
      
      # TODO: check that underlying symbol is valid!
      # if self.underlying not in underlying_symbols:
      #       error_message = 'ERROR: '{}' is not in the list underlying_symbols. '.format(asset)
      #       error_message += 'Please enter an asset from the list.'
      #       sys.exit(error_message)

      if self.save_data: 
         folder_creation('results/{}'.format(symbol), self.verbose)

      path = './results/{}/{}_data.csv'.format(symbol, symbol)
      try: 
         data = pd.read_csv(path) 
         if self.verbose: 
            print('{} data recovered from: '.format(symbol) + path)
      except FileNotFoundError:
         data = self.download_underlying_data(symbol, period)
         data['Return'] = data['Close'].pct_change()
         data['Log_return'] = np.log(data['Close']) - np.log(data['Close'].shift(1))
         if self.verbose: 
            print('{} data downloaded from the web.'.format(symbol))
         if self.save_data: 
            data.to_csv(path)
            if self.verbose: 
               print('Data saved under {}'.format(self.file_path))

      return data
   
   def download_option_data(self, symbol: str, option_type: str, expiration_date: str):
      """Download the option data using the yfinance library."""
           
      expiration_datetime = datetime.strptime(expiration_date, '%Y-%m-%d')
      ticker = yf.Ticker(symbol)
      option_chain = ticker.option_chain(expiration_date)

      if option_type == 'call':
         option_data = option_chain.calls
      elif option_type == 'put':
         option_data = option_chain.puts

      return option_data

   def get_option_data(self, symbol, option_type, expiration_date): 
      """Get the option data, either from the web or the locale memory if they have already been queried."""

      if self.save_data:
         folder_creation('results/{}'.format(symbol), self.verbose)

      try: 
         data = pd.read_csv(self.file_path) 
         if self.verbose: 
            print('Options data recovered from: ' + self.file_path)
      except FileNotFoundError:
         data = self.download_option_data(symbol, option_type, expiration_date)
         if self.verbose: 
            print('Options data downloaded from the web.')
         if self.save_data: 
            data.to_csv(self.file_path)
            if self.verbose: 
               print('Data saved under {}'.format(self.file_path))

      return data

   def get_risk_free_rates(self):
      """ Return and save a dataframe of the historical yields of the 13 weeks treasury bill """

      risk_free_data = self.get_underlying_data('^IRX', '10y')

      return risk_free_data

   def get_risk_free_rate(self): 

      risk_free_data = self.get_risk_free_rates()
      
      risk_free_rate = risk_free_data.iloc[-1]["Close"] / 100

      return risk_free_rate

   def historical_volatility(self):
      return None 

   def SigmaFromIv(self, symbol, option_type, expiration_date):
        ''' K(strike price) needs to be precised here ''' 
        K =  float(input("choose a strike price"))
        donnees= self.get_option_data(symbol, option_type, expiration_date)
        row = donnees[donnees['strike']==K]
        if row.empty:
            print("No data found for the strike price of {}, try to find another mesure for sigma".format(K))
            v = float(input("value for volatility"))
            return v
        else:
            iv = row['impliedVolatility'].values[0]
            return iv

   def timeT(self):
    today =datetime.today()
    expiration= datetime.strptime(self.expiration_date,"%Y-%m-%d")
    difference = expiration-today
    return difference.days

if __name__ == '__main__': 

   gatherer = OptionDataGathering('AAPL', 'call', '2023-04-21')
   print(gatherer.risk_free_rate)
   # T = gatherer.timeT()/365 
   # print(T)
   # sigma = gatherer.SigmaFromIv('GOOG', 'call', '2023-04-21')
   # print(sigma)
   # K is given by us
   
    


