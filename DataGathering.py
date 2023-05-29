import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from Utils import *
import numpy as np
from arch import arch_model
import sys
import os

import ImpliedVolatility
import Calibration

class OptionDataGathering:
    """ Class to gather the information relative to an underlying.
    If save_data is true, the data is automatically saved as a CSV file in the appropriate file structure. """

    def __init__(self,
                 save_data: bool = True,
                 verbose: bool = True,
                 reload: bool = False):
        """
        Args:
            save_data: if true, save the data in a CSV in the appropriate file structure a load it from there rather
            than from the web
            verbose: if true, provide information from where the data is being gathered
            reload: re-gather the data from the web. Necessary if the evaluation date is not in the saved data.
        """

        self.save_data = save_data
        self.verbose = verbose
        self.reload = reload
        self.closest_evaluation_date = None  # Used if the input evaluation date is not valid.

        if self.save_data:
            folder_creation('results', self.verbose)

    @staticmethod
    def download_underlying_data(symbol: str, period: str):
        """ Download a dataframe containing the historical data of a given stock """

        ticker = yf.Ticker(symbol)
        underlying_data = ticker.history(period)

        return underlying_data

    def get_underlying_data(self, symbol: str, evaluation_date: str, period: str = '10y'):
        """
        Get the underlying data, either from the web or the locale memory if they have already been queried.

        Args:
            symbol: four or five letters unique identifier to find the underlying asset. See the Yahoo Finance
            website for valid symbols
            evaluation_date: date at which the pricing is done, is necessary to get up-to-date data
            period: period over which the data is collected
        """

        # Create a folder to save the data
        if self.save_data:
            folder_creation('results/{}'.format(symbol), self.verbose)

        # Gather the data from the file system or from the web if it is not there
        path = './results/{}/{}_{}_data.csv'.format(symbol, symbol, evaluation_date)
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

            # Create additional information columns such as the return, log-return and the date in datetime
            data['Return'] = data['Close'].pct_change()
            # Avoid the numpy error warning about the 0
            old_settings = np.seterr()
            np.seterr(divide='ignore', invalid='ignore')
            data['Log_return'] = np.log(data['Close']) - np.log(data['Close'].shift(1))
            np.seterr(**old_settings)

            data['Vol'] = data['Log_return'].rolling(100).std()
            data['Date'] = pd.to_datetime(data.index, errors='coerce', utc=True)
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
        """
        returns: an asset price given by its symbol at a given evaluation date

         Args:
             symbol: four or five letters unique identifier to find the underlying asset. See the Yahoo Finance
             website for valid symbols
             evaluation_date: date at which the pricing is done, is necessary to get up-to-date data
         """

        # Get the asset complete data
        underlying_data = self.get_underlying_data(symbol, evaluation_date)
        evaluation_datetime = datetime.strptime(evaluation_date, '%Y-%m-%d').date()
        # Get the asset price at the given evaluation date or at the closest valid date.
        try:
            underlying_value_at_evaluation_date = \
                underlying_data.loc[
                    underlying_data['Date'] == evaluation_datetime, 'Close'].iloc[0]
        except IndexError:
            print(
                "Error: the evaluation date you gave is not in the database. "
                "Make sure that it is a valid date for faster runtime. "
                "You can reload the cache to get an up-to-date dataset. "
                "Here are the last available dates:")
            print(underlying_data['Date'].tail(5))
            if underlying_data['Date'].iloc[-1] < evaluation_datetime:
                underlying_value_at_evaluation_date = underlying_data['Close'].iloc[-1]
                self.closest_evaluation_date = underlying_data['Date'].iloc[-1]
                print(
                    f"The date you used is not yet in the date base. Last date available used: "
                    f"{self.closest_evaluation_date}")
            elif (underlying_data['Date'].iloc[-1] > evaluation_datetime) and (
                    underlying_data['Date'].iloc[0] < evaluation_datetime):
                previous_day = datetime.strptime(evaluation_date, '%Y-%m-%d').date() - timedelta(days=1)
                while True:
                    try:
                        underlying_value_at_evaluation_date = \
                            underlying_data.loc[
                                underlying_data['Date'] == previous_day, 'Close'].iloc[0]
                        print(f"The date you used is not in the database. Closest date used instead: {previous_day}")
                        self.closest_evaluation_date = previous_day
                        break
                    except IndexError:
                        previous_day = previous_day - timedelta(days=1)
                        continue
            else:
                underlying_value_at_evaluation_date = underlying_data['Close'].iloc[0]
                self.closest_evaluation_date = underlying_data['Date'].iloc[0]
                print(
                    f"The date you used is too early to be in the database. First available date used:"
                    f" {self.closest_evaluation_date}")

        return underlying_value_at_evaluation_date

    @staticmethod
    def download_option_data(symbol: str, option_type: str, expiration_date: str):
        """
        Download the option data using the yfinance library.

        Args:
            symbol: four or five letters unique identifier to find the underlying asset. See the Yahoo Finance
            website for valid symbols
            option_type: call or put
            expiration_date: date at which the options can be exercised
        """

        ticker = yf.Ticker(symbol)
        option_chain = ticker.option_chain(expiration_date)

        if option_type == 'call':
            option_data = option_chain.calls
        elif option_type == 'put':
            option_data = option_chain.puts
        else:
            raise ValueError('Invalid option type. Must be "call" or "put".')

        return option_data

    def get_option_data(self, symbol: str, option_type: str, expiration_date: str, evaluation_date: str):
        """
        Get the option data, either from the web or the locale memory if they have already been queried.

        Args:
            symbol: four or five letters unique identifier to find the underlying asset. See the Yahoo Finance
            website for valid symbols
            option_type: call or put
            expiration_date: date at which the options can be exercised
            evaluation_date: date at which the options are priced, necessary for up to date data
        """

        # Create the folder to save the data
        if self.save_data:
            folder_creation('results/{}'.format(symbol), self.verbose)

        # Check the option type
        if not option_type == 'call' and not option_type == 'put':
            raise ValueError('Invalid option type. Must be "call" or "put".')
        file_path = './results/{}/{}_{}_{}'.format(symbol, option_type, evaluation_date, expiration_date)

        # Load the option data
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

    def get_dates_available_option(self, symbol):
        """
        Return all the dates for which options are available

        Args:
            symbol: four or five letters unique identifier to find the underlying asset. See the Yahoo Finance
            website for valid symbols
        """

        ticker = yf.Ticker(symbol)
        options_dates = ticker.options

        return options_dates

    def get_risk_free_rates(self, evaluation_date):
        """
        Return and save a dataframe of the historical yields of the 13 weeks treasury bill

        Args:
            evaluation_date: date at which the risk-free rate is queried, necessary for up-to-date data
        """

        path = f'./results/^IRX/INT_^IRX_data_{evaluation_date}.csv'
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
            raw_data = self.get_underlying_data('^IRX', evaluation_date, '15y')
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
        """
        Get the risk-free rate of the 13 weeks treasury bill at a given evaluation date

        Args:
            evaluation_date: date at which the risk-free rate is queried, necessary for up-to-date data
        """

        risk_free_data = self.get_risk_free_rates(evaluation_date)
        evaluation_datetime = datetime.strptime(evaluation_date, '%Y-%m-%d').date()

        try:
            if self.closest_evaluation_date is None:
                risk_free_rate_at_evaluation_date = \
                    risk_free_data.loc[
                        risk_free_data['Date'] == evaluation_datetime, 'Close'].iloc[0]
            else:
                risk_free_rate_at_evaluation_date = \
                    risk_free_data.loc[
                        risk_free_data['Date'] == self.closest_evaluation_date, 'Close'].iloc[0]
        except IndexError:
            if risk_free_data['Date'].iloc[-1] < evaluation_datetime:
                risk_free_rate_at_evaluation_date = risk_free_data['Date'].iloc[-1]
            elif (risk_free_data['Date'].iloc[-1] > evaluation_datetime) and (
                    risk_free_data['Date'].iloc[0] < evaluation_datetime):
                previous_day = datetime.strptime(evaluation_date, '%Y-%m-%d').date() - timedelta(days=1)
                while True:
                    try:
                        risk_free_rate_at_evaluation_date = \
                            risk_free_data.loc[
                                risk_free_data['Date'] == previous_day, 'Close'].iloc[0]
                        break
                    except IndexError:
                        previous_day = previous_day - timedelta(days=1)
                        continue
            else:
                risk_free_rate_at_evaluation_date = risk_free_data['Date'].iloc[0]

        risk_free_rate_at_evaluation_date = np.log((1 + (risk_free_rate_at_evaluation_date / 100)) ** (1/365))

        return risk_free_rate_at_evaluation_date

    def get_historical_volatilities(self, symbol: str, evaluation_date: str):
        """
        Return the historical volatility of the price of the underlying at the close

        Args:
            symbol: four or five letters unique identifier to find the underlying asset. See the Yahoo Finance
            website for valid symbols
            evaluation_date: date at which the volatilities are queried, necessary for up-to-date data
        """

        underlying_data = self.get_underlying_data(symbol, evaluation_date)
        historical_volatilities = pd.DataFrame()
        historical_volatilities['Date'] = underlying_data['Date']
        historical_volatilities['Vol'] = underlying_data['Log_return'].rolling(100).std()

        return historical_volatilities

    def get_historical_volatility(self, symbol: str, evaluation_date: str):
        """
         Return the historical volatility at the valuation date

         Args:
            symbol: four or five letters unique identifier to find the underlying asset. See the Yahoo Finance
            website for valid symbols
            evaluation_date: date at which the volatility is queried, necessary for up-to-date data
         """

        historical_volatilities = self.get_underlying_data(symbol, evaluation_date)
        if self.closest_evaluation_date is None:
            historical_volatility_at_evaluation_date = \
                historical_volatilities.loc[
                    historical_volatilities['Date'] == datetime.strptime(evaluation_date,
                                                                         '%Y-%m-%d').date(), 'Vol'].iloc[0]
        else:
            historical_volatility_at_evaluation_date = \
                historical_volatilities.loc[
                    historical_volatilities['Date'] == self.closest_evaluation_date, 'Vol'].iloc[0]

        return historical_volatility_at_evaluation_date
    
    def get_GARCH_volatility(self, symbol, evaluation_date: str, log_return: bool = True):
        """
        Function to compute the GARCH volatility of an asset

        Args:
            symbol: four or five letters unique identifier to find the underlying asset. See the Yahoo Finance
            website for valid symbols
            evaluation_date: date at which the GARCH volatility is queried, necessary for up-to-date data
            log_return: True to compute the GARCH volatility on the log returns, false to compute it on the returns
        """

        if log_return:
            returns = np.array(self.get_underlying_data(symbol, evaluation_date )['Log_return'])
        else:
            returns = np.array(self.get_underlying_data(symbol, evaluation_date)['Return'])
        # Dirty hack to avoid printing the arch_model.fit() iteration warning since the show_warning is broken
        old_stdout = sys.stdout  # backup current stdout
        sys.stdout = open(os.devnull, "w")
        returns = returns[~np.isnan(returns)]
        model = arch_model(returns, vol="GARCH", p=1, q=1, rescale=False)
        fit = model.fit(show_warning=False).conditional_volatility
        sys.stdout = old_stdout  # reset old stdout

        return fit

    def get_calibration_Merton(self, symbol: str, evaluation_date: str):
        """
        Get the calibration for an asset for the Merton model

        Args:
            symbol: four or five letters unique identifier to find the underlying asset. See the Yahoo Finance
            website for valid symbols
            evaluation_date: date at which the calibration is queried, necessary for up-to-date data
        """

        # Create a file path to load the cache
        file_path = file_path = './results/{}/MertonParameters.csv'.format(symbol)

        try:
            variables = np.genfromtxt(file_path, delimiter=',')
            if self.verbose:
                print("Merton model parameters obtained from the cache.")
        except OSError:
            # Get the log returns to calibrate
            log_returns = np.array(self.get_underlying_data(symbol, evaluation_date)['Log_return'])
            log_returns = log_returns[~np.isnan(log_returns)]

            # Get the calibration for the Merton model
            calibrator = Calibration.CalibrateVanilla(log_returns)
            variables = np.array(calibrator.merton_calibrate())
            if self.verbose:
                print('Computed the Merton model parameters.')
            if self.save_data:
                np.savetxt(file_path, variables, delimiter=",")
                if self.verbose:
                    print(f"Merton model parameters saved under {file_path}")

        return variables

    def get_calibration_Heston(self, symbol, evaluation_date):
        """
         Get the calibration for an asset for the Heston model

         Args:
             symbol: four or five letters unique identifier to find the underlying asset. See the Yahoo Finance
             website for valid symbols
             evaluation_date: date at which the calibration is queried, necessary for up-to-date data
         """

        # Create a file path to load the cache
        file_path = file_path = './results/{}/HestonParameters.csv'.format(symbol)

        try:
            variables = np.genfromtxt(file_path, delimiter=',')
            if self.verbose:
                print("Heston model parameters obtained from the cache.")
        except OSError:
            # Get the log returns to calibrate and their GARCH volatility
            log_returns = np.array(self.get_underlying_data(symbol, evaluation_date)['Log_return'])
            log_returns = log_returns[~np.isnan(log_returns)]
            volatility = self.get_GARCH_volatility(symbol, evaluation_date)
            spotvol = volatility[-1]

            # Get the calibration for the Heston model
            calibrator = Calibration.CalibrateVanilla(log_returns, volatility)
            variables = np.array(np.append(calibrator.heston_calibrate(),spotvol))

            if self.verbose:
                print('Computed the Heston model parameters.')
            if self.save_data:
                np.savetxt(file_path, variables, delimiter=",")
                if self.verbose:
                    print(f"Heston model parameters saved under {file_path}")

        return variables

    def get_implied_volatility(self, symbol, strike, expiration_date, evaluation_date, option_type='call'):
        """
        Get the implied volatility internally computed. WARNING: very time-consuming, only works for AAPL

        Args:
            symbol: four or five letters unique identifier to find the underlying asset. See the Yahoo Finance
            website for valid symbols
            strike: strike price of the option for whom the implied volatility is computed
            expiration_date: exercise date of the option
            evaluation_date: date at which the implied volatility is queried, necessary for up-to-date data
            option type: etiher call or put
        """

        list_underlying_iv = ['AAPL']
        if symbol in list_underlying_iv:
            time_to_maturity = self.time_to_maturity(expiration_date)
            iv = ImpliedVolatility.OptionImpliedVolatility(symbol).get_implied_volatility(strike, time_to_maturity)
        else:
            print("Error: there is no data to compute the implied volatility for this asset.")
            iv = self.SigmaFromIv(symbol, strike, expiration_date, evaluation_date, option_type)

        return iv

    def SigmaFromIv(self, symbol, strike, expiration_date, evaluation_date, option_type):
        """
        Get the implied volatility of an option from the Yahoo Finance data

        Args:
            symbol: four or five letters unique identifier to find the underlying asset. See the Yahoo Finance
            website for valid symbols
            strike: strike price of the option for whom the implied volatility is computed
            expiration_date: exercise date of the option
            evaluation_date: date at which the implied volatility is queried, necessary for up-to-date data
            option type: etiher call or put
        """

        donnees = self.get_option_data(symbol, option_type, expiration_date, evaluation_date)
        row = donnees[donnees['strike'] == strike]
        if row.empty:
            print("No data found for the strike price of {}, try to find another mesure for sigma".format(strike))
            iv = float(input("value for volatility"))
            return iv
        else:
            iv = row['impliedVolatility'].values[0]
            return iv

    def time_to_maturity(self, expiration_date, evaluation_date=None):
        """
        Return the time difference between two dates in days

        Args:
            expiration_date: end date
            evaluation_date: begin date
        """
        if evaluation_date is None:
            evaluation = datetime.today()
        else:
            evaluation = datetime.strptime(evaluation_date, "%Y-%m-%d")
        expiration = datetime.strptime(expiration_date, "%Y-%m-%d")
        difference = expiration - evaluation
        return difference.days



