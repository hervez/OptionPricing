from typing import List
import matplotlib.pyplot as plt
from OptionDataType import OptionData

class OptionGraphicalAnalysis():

    def __init__(self):

       self.test = "test"

    def plot_price_strikes(self, option_list):
        prices = []
        strikes = []
        estimated_BS_price = []
        for option in option_list:
            prices.append(option.option_price)
            strikes.append(option.strike_price)
            estimated_BS_price.append(option.BS_pricing)

        plt.plot(prices, prices, label="Estimated Price")
        plt.plot(prices, strikes, label="Actual Price")
        plt.xlabel("Strike")
        plt.ylabel("Price")
        plt.legend()
        plt.show()
