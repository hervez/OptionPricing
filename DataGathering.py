import numpy
import yfinance as yf
from datetime import datetime
import os.path
import re
import pandas as pd
from Utils import *
from datetime import datetime
import numpy as np

import ImpliedVolatility

class OptionDataGathering():
    """ Class to gather the information relative to an underlying.
    If save_data is true, the data is automatically saved as a CSV file in the appropriate file structure. """

    def __init__(self,
                 save_data: bool = True,
                 verbose: bool = True,
                 reload: bool = False):

        self.save_data = save_data
        self.verbose = verbose
        self.reload = reload

        if self.save_data:
            folder_creation('results', self.verbose)

    def download_underlying_data(self, symbol: str, period: str):
        """ Download a dataframe containing the historical data of a given stock """

        ticker = yf.Ticker(symbol)

        underlying_data = ticker.history(period)

        return underlying_data

    def get_underlying_data(self, symbol: str, period: str = '10y'):
        """Get the underlying data, either from the web or the locale memory if they have already been queried."""

        if self.save_data:
            folder_creation('results/{}'.format(symbol), self.verbose)

        path = './results/{}/{}_data.csv'.format(symbol, symbol)
        try:
            if self.reload:
                raise FileNotFoundError("Reloading the data from the web")
            data = pd.read_csv(path, parse_dates=[0], index_col=False)
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce', utc=True)
            data = data.dropna(subset=['Date'])
            data['Date'] = data['Date'].dt.date
            if self.verbose:
                print('{} data recovered from: '.format(symbol) + path)
        except FileNotFoundError:
            data = self.download_underlying_data(symbol, period)
            data['Return'] = data['Close'].pct_change()
            data['Log_return'] = np.log(data['Close']) - np.log(data['Close'].shift(1))
            data['Date'] = pd.to_datetime(data.index, errors='coerce', utc=True)
            # data['Date'] = pd.to_datetime(data['Date'], errors='coerce', utc=True)
            data = data.dropna(subset=['Date'])
            data['Date'] = data['Date'].dt.date
            data.reset_index(inplace=True, drop=True)
            if self.verbose:
                print('{} data downloaded from the web.'.format(symbol))
            if self.save_data:
                data.to_csv(path, index=False)
                if self.verbose:
                    print('Data saved under {}'.format(path))

        return data

    def get_underlying_value_at_evaluation_date(self, symbol: str, evaluation_date: str):

        underlying_data = self.get_underlying_data(symbol)
        underlying_value_at_evaluation_date = \
            underlying_data.loc[
                underlying_data['Date'] == datetime.strptime(evaluation_date, '%Y-%m-%d').date(), 'Close'].iloc[0]

        return underlying_value_at_evaluation_date

    def download_option_data(self, symbol: str, option_type: str, expiration_date: str):
        """Download the option data using the yfinance library."""

        expiration_datetime = datetime.strptime(expiration_date, '%Y-%m-%d')
        ticker = yf.Ticker(symbol)
        option_chain = ticker.option_chain(expiration_date)

        if option_type == 'call':
            option_data = option_chain.calls
        elif option_type == 'put':
            option_data = option_chain.puts
        else:
            raise ValueError('Invalid option type. Must be "call" or "put".')

        return option_data

    def get_option_data(self, symbol, option_type, expiration_date):
        """Get the option data, either from the web or the locale memory if they have already been queried."""

        if self.save_data:
            folder_creation('results/{}'.format(symbol), self.verbose)

        if option_type == 'call' or option_type == 'put':
            self.option_type = option_type
        else:
            raise ValueError('Invalid option type. Must be "call" or "put".')
        file_path = './results/{}/{}_{}'.format(symbol, option_type, expiration_date)

        try:
            if self.reload:
                raise FileNotFoundError("Reloading the data from the web")
            data = pd.read_csv(file_path)
            if self.verbose:
                print('Options data recovered from: ' + file_path)
        except FileNotFoundError:
            data = self.download_option_data(symbol, option_type, expiration_date)
            if self.verbose:
                print('Options data downloaded from the web.')
            if self.save_data:
                data.to_csv(file_path)
                if self.verbose:
                    print('Data saved under {}'.format(file_path))

        return data

    def get_risk_free_rates(self):
        """ Return and save a dataframe of the historical yields of the 13 weeks treasury bill """

        path = './results/^IRX/INT_^IRX_data.csv'
        try:
            if self.reload:
                raise FileNotFoundError("Reloading the data from the web")
            data = pd.read_csv(path, parse_dates=[0])
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce', utc=True)
            data = data.dropna(subset=['Date'])
            data['Date'] = data['Date'].dt.date
            if self.verbose:
                print('Interpolated ^IRX data recovered from: ' + path)
        except FileNotFoundError:
            raw_data = self.get_underlying_data('^IRX', '15y')
            start_date = raw_data['Date'].iloc[0]
            end_date = raw_data['Date'].iloc[-1]
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            date_range = [d.date() for d in date_range]
            new_df = pd.DataFrame({'Date': date_range})
            merged_df = pd.merge(new_df, raw_data, on='Date', how='left')
            merged_df['Close'] = merged_df['Close'].interpolate()
            if self.save_data:
                merged_df.to_csv(path, index=False)
                if self.verbose:
                    print('Data saved under {}'.format(path))
            data = merged_df

        interpolated_risk_free_data = data

        return interpolated_risk_free_data

    def get_risk_free_rate(self, evaluation_date: str):

        risk_free_data = self.get_risk_free_rates()

        risk_free_rate_at_evaluation_date = \
            risk_free_data.loc[
                risk_free_data['Date'] == datetime.strptime(evaluation_date, '%Y-%m-%d').date(), 'Close'].iloc[0]

        return risk_free_rate_at_evaluation_date / 365

    def get_historical_volatilities(self, symbol: str):
        """ Return the historical volatility of the price of the undelrying at the close """

        underlying_data = self.get_underlying_data(symbol)
        historical_volatilities = pd.DataFrame()
        historical_volatilities['Date'] = underlying_data['Date']
        historical_volatilities['Vol'] = underlying_data['Close'].rolling(100).std()

        return historical_volatilities

    def get_historical_volatility(self, symbol: str, evaluation_date: str):
        """ Return the historical volatility at the valuation date """

        historical_volatilities = self.get_historical_volatilities(symbol)
        historical_volatility_at_evaluation_date = \
            historical_volatilities.loc[
                historical_volatilities['Date'] == datetime.strptime(evaluation_date, '%Y-%m-%d').date(), 'Vol'].iloc[0]

        return historical_volatility_at_evaluation_date

    def get_implied_volatility(self, symbol, strike, expiration_date, option_type='call'):
        list_underlying_iv = ['AAPL']
        if symbol in list_underlying_iv:
            time_to_maturity = self.time_to_maturity(expiration_date)
            iv =  ImpliedVolatility.OptionImpliedVolatility(symbol).get_implied_volatility(strike, time_to_maturity)
        else:
            print("Error: there is no data to compute the implied volatility for this asset.")
            iv = self.SigmaFromIv(symbol, strike, expiration_date, option_type)

        return iv

    def SigmaFromIv(self, symbol, strike, expiration_date, option_type):
        ''' K(strike price) needs to be precised here '''
        # K = float(input("choose a strike price"))
        donnees = self.get_option_data(symbol, option_type, expiration_date)
        row = donnees[donnees['strike'] == strike]
        if row.empty:
            print("No data found for the strike price of {}, try to find another mesure for sigma".format(strike))
            iv = float(input("value for volatility"))
            return iv
        else:
            iv = row['impliedVolatility'].values[0]
            return iv

    def time_to_maturity(self, expiration_date, evaluation_date=None):
        if evaluation_date == None:
            evaluation = datetime.today()
        else:
            evaluation = datetime.strptime(evaluation_date, "%Y-%m-%d")
        expiration = datetime.strptime(expiration_date, "%Y-%m-%d")
        difference = expiration - evaluation
        return difference.days


if __name__ == '__main__':
    gatherer = OptionDataGathering(reload=True)
    # gatherer.get_risk_free_rate('2020-01-01')
    # print(gatherer.get_implied_volatility('AAPL', 100, '2023-07-21' , 'call'))
    # print(gatherer.get_option_data('AAPL', 'call', '2022-04-21'))
    gatherer.get_underlying_value_at_evaluation_date('AAPL', '2023-04-10')
    # gatherer.get_historical_volatilities('AAPL')
    # print(gatherer.get_historical_volatility('AAPL', '2023-04-10'))
    # print(gatherer.get_risk_free_rate('2023-04-10'))
    # print(gatherer.historical_volatility)
    # print(gatherer.underlying_value_at_evaluation_date)
    # print(gatherer.risk_free_rate)
    # T = gatherer.timeT()/365
    # print(T)
    # sigma = gatherer.SigmaFromIv('GOOG', 'call', '2023-04-21')
    # print(sigma)
    # K is given by us
