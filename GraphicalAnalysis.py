from typing import List
import matplotlib.pyplot as plt
from OptionDataType import OptionData
import os
import seaborn as sns


# matplotlib.use('pgf')


class OptionGraphicalAnalysis:
    """
    Class to create plots of the different results obtained in the code.
    """

    def __init__(self, underlying: str):
        """
        Args:
            underlying: underlying of the option priced. Necessary to save the plot in the correct folder
        """

        self.underlying = underlying
        self.directory = f"./results/{self.underlying}/analysis/figures/"  # Directory for the LaTex document figures
        sns.set_theme()  # Set the theme for the graphics

    def plot_price_strikes(self, option_list, save_fig, pricer: str = None):
        """ Creates a plot of the estimated prices and the real prices for each strike prices """

        option_type = option_list[0].option_type
        if pricer is None:
            file_path = self.directory + f"figure_price_strike_{option_type}.png"
        else:
            file_path = self.directory + f"figure_price_strike_{option_type}_{pricer}.png"

        # Gather the data from the option list
        prices = []
        strikes = []
        if pricer is None:
            estimated_BS_price = []
            estimated_CRR_price = []
            estimated_FFT_price = []
            estimated_Fourier_price = []
            estimated_Merton_price = []
            estimated_Heston_price = []
        else:
            estimated_price = []
            pricing_name = f"{pricer}_pricing"
        for option in option_list:
            if option.option_type == option_type:
                prices.append(option.option_price)
                strikes.append(option.strike_price)
                if pricer is None:
                    estimated_BS_price.append(option.BS_pricing)
                    estimated_CRR_price.append(option.CRR_pricing)
                    estimated_FFT_price.append(option.FFT_pricing)
                    estimated_Fourier_price.append(option.Fourier_pricing)
                    estimated_Merton_price.append(option.Merton_pricing)
                    estimated_Heston_price.append(option.Heston_pricing)
                else:
                    estimated_price.append(getattr(option, pricing_name))

        # Create the figure and axes objects
        fig, ax = plt.subplots()

        # Plot the true prices as scatter plot
        sns.lineplot(x=strikes, y=prices, ax=ax, label='True Price')
        sns.scatterplot(x=strikes, y=prices, ax=ax)

        # Plot the estimated prices as scatter plots
        if pricer is None:
            sns.scatterplot(x=strikes, y=estimated_BS_price, ax=ax, label='BS estimated prices')
            sns.scatterplot(x=strikes, y=estimated_CRR_price, ax=ax, label='CRR estimated prices')
            sns.scatterplot(x=strikes, y=estimated_Fourier_price, ax=ax, label='Fourier estimated prices')
            sns.scatterplot(x=strikes, y=estimated_FFT_price, ax=ax, label='FFT estimated prices')
            sns.scatterplot(x=strikes, y=estimated_Merton_price, ax=ax, label='Merton estimated prices')
            sns.scatterplot(x=strikes, y=estimated_Heston_price, ax=ax, label='Heston estimated prices')
        else:
            match pricer:
                case "BS":
                    color = sns.color_palette()[1]
                case "CRR":
                    color = sns.color_palette()[2]
                case "Fourier":
                    color = sns.color_palette()[3]
                case "FFT":
                    color = sns.color_palette()[4]
                case "Merton":
                    color = sns.color_palette()[5]
                case "Heston":
                    color = sns.color_palette()[6]
            sns.scatterplot(x=strikes, y=estimated_price, ax=ax, label= f"{pricer} estimated prices", color=color)

        # Set the title and axis labels
        if pricer is None:
            ax.set_title(f'True vs. Estimated {option_type.capitalize()} Prices')
        else:
            ax.set_title(f'True vs. Estimated {option_type.capitalize()} {pricer} Prices')
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Option Price')

        if save_fig:
            # Create a directory to save the figures
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
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
        """ Create a plot of the estimated and Yahoo implied volatility by strike prices """

        # Gather the data from the option list
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
