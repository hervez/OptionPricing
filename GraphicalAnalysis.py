from typing import List
import matplotlib
import matplotlib.pyplot as plt
from OptionDataType import OptionData
import os
import seaborn as sns


# matplotlib.use('pgf')


class OptionGraphicalAnalysis():

    def __init__(self, underlying: str):

       self.underlying = underlying
       self.directory =   f"./results/{self.underlying}/analysis/figures/"
       sns.set_theme()

    def plot_price_strikes(self, option_list):

        directory = f"./results/{self.underlying}/analysis/figures"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = self.directory + "figure_price_strike.pgf"

        prices = []
        strikes = []
        estimated_BS_price = []
        for option in option_list:
            prices.append(option.option_price)
            strikes.append(option.strike_price)
            estimated_BS_price.append(option.BS_pricing)

        # Create the figure and axes objects
        fig, ax = plt.subplots()

        # Plot the true prices as scatter plot
        sns.lineplot(x=strikes, y=prices, ax=ax, label='True Price')
        sns.scatterplot(x=strikes, y=prices, ax=ax)

        # Plot the estimated prices as line plot
        sns.scatterplot(x=strikes, y=estimated_BS_price, ax=ax, label='Estimated Price')

        # Set the title and axis labels
        ax.set_title('True vs. Estimated Option Prices')
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Option Price')

        # Show the plot
        plt.show()

        #plt.savefig(file_path)
        # plt.show()

    def plot_implied_volatility_3D(self, option_list: List[OptionData]):

        time_to_maturity = []
        strike_price = []
        implied_volatility = []
        for option in option_list:
            time_to_maturity.append(option.time_to_maturity)
            strike_price.append(option.strike_price)
            implied_volatility.append(option.implied_volatility)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Create the scatter plot
        ax.scatter(time_to_maturity, strike_price, implied_volatility, c=implied_volatility)

        # Set the labels for the axes
        ax.set_xlabel('Time to Maturity')
        ax.set_ylabel('Strike Price')
        ax.set_zlabel('Implied Volatility')

        plt.show()

    def plot_implied_volatility_2D(self, option_list: List[OptionData]):
        estimated_implied_volatility = []
        strike_price = []
        implied_volatility = []
        for option in option_list:
            estimated_implied_volatility.append(option.estimated_implied_vol)
            strike_price.append(option.strike_price)
            implied_volatility.append(option.implied_volatility)

        # Create the figure and axes objects
        fig, ax = plt.subplots()

        # Plot the true prices as scatter plot
        sns.lineplot(x=strike_price, y=implied_volatility, ax=ax, label='Implied volatility')
        sns.scatterplot(x=strike_price, y=implied_volatility, ax=ax)

        # Plot the estimated prices as line plot
        sns.scatterplot(x=strike_price, y=estimated_implied_volatility, ax=ax, label='Estimated Implied Volatility')

        # Set the title and axis labels
        ax.set_title('True vs. Estimated Option Implied Volatility')
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Implied Volatility')

        # Show the plot
        plt.show()