import yfinance as yf
from datetime import datetime
import os.path
import re 
import pandas as pd

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
         if not os.path.exists('./results'):
               os.mkdir('./results')
               if self.verbose: 
                  print('===================================================')
                  print('Created a /results/ directory in {}/'.format(os.getcwd()))
                  print('===================================================')
         # TODO: check that underlying symbol is valid!
         # if self.underlying not in underlying_symbols:
         #       error_message = 'ERROR: '{}' is not in the list underlying_symbols. '.format(asset)
         #       error_message += 'Please enter an asset from the list.'
         #       sys.exit(error_message)
         if not os.path.exists('./results/{}'.format(self.underlying)):
            os.mkdir('./results/{}'.format(self.underlying))
            if self.verbose: 
               print('\tCreated a /{}/ directory in {}/results/'.format(self.underlying, os.getcwd()))

      self.data = self.get_option_data(self.underlying, self.option_type, self.expiration_date)

   def download_option_data(self, symbol, option_type, expiration_date):
      """Download the option data using the yfinance library."""
           
      # Convert expiration date to datetime object
      expiration_datetime = datetime.strptime(expiration_date, '%Y-%m-%d')

      # Create a Ticker object for the underlying symbol
      ticker = yf.Ticker(symbol)

      # Get the option chain for the specified expiration date
      option_chain = ticker.option_chain(expiration_date)

      # Get the data for the specified option type (call or put)
      if option_type == 'call':
         option_data = option_chain.calls
      elif option_type == 'put':
         option_data = option_chain.puts

      return option_data

   def get_option_data(self, symbol, option_type, expiration_date): 
      """Get the option data, either from the web or the locale memory if they have already been queried."""
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

if __name__ == '__main__': 

   gatherer = OptionDataGathering('GOOG', 'call', '2023-04-21')
   # print(gatherer.data) 