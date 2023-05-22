from typing import List
import matplotlib
import matplotlib.pyplot as plt
from OptionDataType import OptionData
import os
import seaborn as sns


# matplotlib.use('pgf')


class OptionGraphicalAnalysis():

    def __init__(self, underlying: str, save_fig: bool = False):

       self.underlying = underlying
       self.directory =   f"./results/{self.underlying}/analysis/figures/"
       sns.set_theme()

    def plot_price_strikes(self, option_list, save_fig):

        directory = f"./results/{self.underlying}/analysis/figures"
        if not os.path.exists(directory):
            os.makedirs(directory)
        option_type = option_list[0].option_type
        file_path = self.directory + f"figure_price_strike_{option_type}.png"

        prices = []
        strikes = []
        estimated_BS_price = []
        estimated_CRR_price = []
        #estimated_BSM_price =[]
        estimated_Fourier_price = []
        for option in option_list:
            if option.option_type == option_type:
                prices.append(option.option_price)
                strikes.append(option.strike_price)
                estimated_BS_price.append(option.BS_pricing)
                estimated_CRR_price.append(option.CRR_pricing)
                #estimated_BSM_price.append(option.BSM_pricing)
                estimated_Fourier_price.append(option.Fourier_pricing)

        # Create the figure and axes objects
        fig, ax = plt.subplots()

        # Plot the true prices as scatter plot
        sns.lineplot(x=strikes, y=prices, ax=ax, label='True Price')
        sns.scatterplot(x=strikes, y=prices, ax=ax)

        # Plot the estimated prices as scatter plots
        sns.scatterplot(x=strikes, y=estimated_BS_price, ax=ax, label='BS estimated Price')
        sns.scatterplot(x=strikes, y=estimated_CRR_price, ax=ax, label='CRR estimated Price')
        #sns.scatterplot(x=strikes, y=estimated_BSM_price, ax=ax, label='BSM estimated Price')
        sns.scatterplot(x=strikes, y=estimated_Fourier_price, ax=ax, label='Fourier estimated Price')

        # Set the title and axis labels
        ax.set_title(f'True vs. Estimated {option_type.capitalize()} Prices')
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Option Price')

        if save_fig:
            plt.savefig(file_path)
        else:
            plt.show()

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