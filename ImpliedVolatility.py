import numpy as np
import pandas as pd
from typing import List
from scipy.optimize import minimize
from scipy.interpolate import griddata, RegularGridInterpolator, interp2d
from math import log, sqrt, exp, floor, ceil
from scipy.stats import norm
from datetime import datetime

import DataGathering
from OptionDataType import OptionData


class OptionImpliedVolatility:
    """
    Class to estimate the implied volatility of option given an underlying.
    """

    def __init__(self, underlying: str):
        self.underlying = underlying
        self.gatherer = DataGathering.OptionDataGathering(verbose=False)
        self.option_data = self.get_options_data()
        self.volatility_computer = ImpliedVolatilitySurfaceComputer(self.option_data)
        self.surface = self.volatility_computer.calculate_surface()
        # print(self.surface)
        # if save_data:
        #     path = "./ImpliedVolatilityData/{}/iv_surface.csv".format(self.underlying)
        #     np.savetxt(path, self.surface, delimiter=",")

    def get_options_data(self):
        options_type = ['call', 'put']
        options_list = list()
        for type in options_type:
            file_name = '{}.csv'.format(type)
            path = './ImpliedVolatilityData/{}/{}'.format(self.underlying, file_name)
            option_data = pd.read_csv(path, sep=',')
            option_data.dropna(inplace=True)

            for index, row in option_data.iterrows():
                # Conversion date to correct format
                option_date = str(row['DATE'])
                option_date_dt = datetime.strptime(option_date, '%d.%m.%Y')
                option_date = datetime.strftime(option_date_dt, '%Y-%m-%d')

                risk_free_rate = self.gatherer.get_risk_free_rate(option_date)
                option = OptionData( \
                    option_price=row['PRICE'], \
                    underlying_price=row['U/LYING PRICE'], \
                    strike_price=row['STRIKE'], \
                    time_to_maturity=row['LIFE DAYS'], \
                    option_type=type, \
                    risk_free_rate=risk_free_rate)
                options_list.append(option)

        return options_list

    def get_implied_volatility(self, strike: float, time_to_maturity: int):
        iv = self.surface(strike, time_to_maturity)[0]
        if iv != np.nan:
            return iv
        else:
            print("Error occured during the computation of the implied volatility.")
            return 0

class ImpliedVolatilitySurfaceComputer:

    def __init__(self, option_data: List[OptionData]):

        self.option_data = option_data

    def black_scholes(self, S, K, T, r, sigma, option_type):
        d1 = (log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        if option_type == 'call':
            option_price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        elif option_type == 'put':
            option_price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("Option type must be 'call' or 'put'")

        return option_price

    def calculate_implied_volatility(self, option_price, underlying_price, strike_price, time_to_maturity,
                                     risk_free_rate, option_type):
        """
        Calculates the implied volatility for a given option price using the Black-Scholes model
        """

        def objective_function(volatility):
            return (self.black_scholes(underlying_price, strike_price, time_to_maturity, risk_free_rate, volatility,
                                       option_type) - option_price) ** 2

        result = minimize(objective_function, x0=0.5, method='Nelder-Mead')
        return result.x[0]

    def calculate_surface(self):
        """
        Calculates the implied volatility surface using the option data
        """
        strikes = []
        maturities = []
        implied_vols = []
        # Calculate the implied volatility for each option
        for option in self.option_data:
            implied_volatility = self.calculate_implied_volatility( \
                option.option_price, \
                option.underlying_price, \
                option.strike_price, \
                option.time_to_maturity, \
                option.risk_free_rate, \
                option.option_type)
            strikes.append(option.strike_price)
            maturities.append(option.time_to_maturity)
            implied_vols.append(implied_volatility)


        f_iv = interp2d(strikes, maturities, implied_vols, kind='linear')


        return f_iv

