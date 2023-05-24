import datetime
from typing import List

from OptionDataType import OptionData
from DataGathering import OptionDataGathering
from ImpliedVolatility import OptionImpliedVolatility
from Pricing import *
from GraphicalAnalysis import OptionGraphicalAnalysis
from TexDocumentCreator import TexDocument


class OptionAnalysis:
    """
    Class orchestrating the different utilities to provide a comprehensive analysis.
    """

    def __init__(self, underlying: str = 'AAPL', expiration_date: str = '2023-04-21',
                 evaluation_date: str = datetime.datetime.today().strftime('%Y-%m-%d')):
        """
        Args:
            underlying: underlying of the option priced
            expiration_date: date at which the option expire
            evaluation_date: date at which the option is priced. Make sure that it is a valid date. By default, it is
            the current date.
        """

        self.underlying = underlying
        self.expiration_date = expiration_date
        self.gatherer = OptionDataGathering(verbose=False, reload=False)
        self.evaluation_date = evaluation_date
        self.ploter = OptionGraphicalAnalysis(self.underlying)

    def get_options(self, option_type: str = 'call'):
        """ Collect the option in a list of OptionData for a better access """

        # Gather the parameter common to each option
        underlying_price = self.gatherer.get_underlying_value_at_evaluation_date(self.underlying, self.evaluation_date)
        risk_free_rate = self.gatherer.get_risk_free_rate(self.evaluation_date)
        time_to_maturity = self.gatherer.time_to_maturity(self.expiration_date, self.evaluation_date)
        historical_volatility = self.gatherer.get_historical_volatility(self.underlying, self.evaluation_date)

        # Get the option from DataGatherer in a dataframe
        option_df = self.gatherer.get_option_data(self.underlying, option_type, self.expiration_date)

        # Generate the list of OptionData
        option_list = []
        for index, row in option_df.iterrows():
            option = OptionData(
                option_type=option_type,
                option_price=row['lastPrice'],
                strike_price=row['strike'],
                implied_volatility=row['impliedVolatility'],
                historical_volatility=historical_volatility,
                historical_std=math.sqrt(historical_volatility),
                underlying_price=underlying_price,
                underlying=self.underlying,
                evaluation_date=self.evaluation_date,
                expiration_date=self.expiration_date,
                risk_free_rate=risk_free_rate,
                time_to_maturity=time_to_maturity,
            )

            option_list.append(option)

        return option_list

    def estimate_implied_volatility(self, options_list: List[OptionData]):
        """ Add the estimated volatility data to the OptionData in the option list. WARNING: very time-consuming!"""

        implied_vol_surface = OptionImpliedVolatility(self.underlying)
        for option in options_list:
            estimated_implied_volatility = implied_vol_surface.get_implied_volatility(option.strike_price,
                                                                                      option.time_to_maturity)
            option.estimated_implied_vol = estimated_implied_volatility

        return options_list

    def price_option(self, options_list: List[OptionData], verbose: bool = False):
        """ Price the OptionData using the different Pricer """

        for option in options_list:

            # BS pricing
            pricer = OptionPricerBlackScholes(S_0=option.underlying_price, K=option.strike_price,
                                              T=option.time_to_maturity, r=option.risk_free_rate,
                                              sigma=option.historical_std)
            if option.option_type == 'call':
                option.BS_pricing = pricer.get_call()
            if option.option_type == 'put':
                option.BS_pricing = pricer.get_put()
            if verbose:
                print("Black Scholes Priced")

            # CRR pricing
            pricerCRR = OptionPricerCRR(S_0=option.underlying_price, K=option.strike_price,
                                        T=option.time_to_maturity, r=option.risk_free_rate,
                                        sigma=option.historical_std, M=10*option.time_to_maturity)
            if option.option_type == 'call':
                option.CRR_pricing = pricerCRR.get_call()
            if option.option_type == 'put':
                option.CRR_pricing = pricerCRR.get_put()
            if verbose:
                print("CRR priced")

            # Fourier pricing
            pricerFourier = OptionPricerFourier(S_0=option.underlying_price, K=option.strike_price,
                                                T=option.time_to_maturity, r=option.risk_free_rate,
                                                sigma=option.historical_std)
            if option.option_type == 'call':
                option.Fourier_pricing = pricerFourier.get_call()
            if option.option_type == 'put':
                option.Fourier_pricing = pricerFourier.get_put()
            if verbose:
                print("Fourier priced")

            # Fast Fourier Transform pricing
            pricerFFT = OptionPricerFFT(S_0=option.underlying_price, K=option.strike_price, T=option.time_to_maturity,
                                        r=option.risk_free_rate, sigma=option.historical_std)
            if option.option_type == 'call':
                option.FFT_pricing = pricerFFT.get_call()
            if option.option_type == 'put':
                option.FFT_pricing = pricerFFT.get_put()
            if verbose:
                print("FFT priced")

            # Merton pricing
            Merton_variables = self.gatherer.get_calibration_Merton(self.underlying)
            alpha = Merton_variables[0]
            lamda = Merton_variables[1]
            delta = Merton_variables[2]
            mu = Merton_variables[3]
            Msigma = Merton_variables[4]

            for option in options_list:
                pricer = OptionPricerMerton(S_0=option.underlying_price, K=option.strike_price,
                                            T=option.time_to_maturity, r=option.risk_free_rate,
                                            sigma=option.historical_std, alpha=alpha, lamda=lamda, delta=delta, mu=mu,
                                            Msigma=Msigma)
                if option.option_type == 'call':
                    option.Merton_pricing = pricer.get_call()
                if option.option_type == 'put':
                    option.Merton_pricing = pricer.get_put()
            if verbose:
                print("Merton priced")

            # Heston pricing
            Heston_variables = self.gatherer.get_calibration_Heston(self.underlying)
            mu = Heston_variables[0]
            rho = Heston_variables[1]
            kappa = Heston_variables[2]
            eta = Heston_variables[3]
            theta = Heston_variables[4]

            for option in options_list:
                pricer = OptionPricerHeston(S_0=option.underlying_price, K=option.strike_price,
                                            T=option.time_to_maturity, r=option.risk_free_rate,
                                            sigma=option.historical_std, rho=rho, kappa=kappa, eta=eta, theta=theta)
                if option.option_type == 'call':
                    option.Heston_pricing = pricer.get_call()
                if option.option_type == 'put':
                    option.Heston_pricing = pricer.get_put()
            if verbose:
                print("Heston priced")

        return options_list


    def pricing_test(self, options_list):
        """ Function to test if a Pricer works correctly. Used for development only """

        for option in options_list:
            pricerCRR = OptionPricerCRR(S_0=option.underlying_price, K=option.strike_price,
                                        T=option.time_to_maturity, r=option.risk_free_rate,
                                        sigma=option.historical_std, M=10 * option.time_to_maturity)
            if option.option_type == 'call':
                option.CRR_pricing = pricerCRR.get_call()
            if option.option_type == 'put':
                option.CRR_pricing = pricerCRR.get_put()

        return options_list

    def plot(self, option_list: List[OptionData], save_fig=False, pricer: str = None):
        """ Create a plot of the predicted price and the actual price of the OptionData """

        self.ploter.plot_price_strikes(option_list=option_list, save_fig=save_fig, pricer=pricer)

    def plot_iv(self, option_list: List[OptionData]):
        """ Plot the implied volatility of the OptionData """

        self.ploter.plot_implied_volatility_2D(option_list)

    def tex_document(self, underlying_price: float):
        """ Generate the LaTex document containing the result of the code """

        texer = TexDocument(self.underlying, self.evaluation_date, self.expiration_date,
                            underlying_price=underlying_price)
        texer.generate_document()

    def complete_analysis(self):
        """ Does a complete analysis by getting the option list, pricing it, generating the graphs and the LaTex
        document """

        # Get the options
        calls = self.get_options('call')
        puts = self.get_options('put')
        print("Options gathered")

        # Price the option
        priced_calls = self.price_option(calls)
        priced_puts = self.price_option(puts)
        print("Options priced")

        # Create the figure
        self.plot(priced_calls, save_fig=True)
        self.plot(priced_puts, save_fig=True)

        # Create the individual figures
        self.plot(priced_calls, save_fig=True, pricer="BS")
        self.plot(priced_calls, save_fig=True, pricer="CRR")
        self.plot(priced_calls, save_fig=True, pricer="Fourier")
        self.plot(priced_calls, save_fig=True, pricer="FFT")
        self.plot(priced_calls, save_fig=True, pricer="Merton")
        self.plot(priced_calls, save_fig=True, pricer="Heston")

        self.plot(priced_puts, save_fig=True, pricer="BS")
        self.plot(priced_puts, save_fig=True, pricer="CRR")
        self.plot(priced_puts, save_fig=True, pricer="Fourier")
        self.plot(priced_puts, save_fig=True, pricer="FFT")
        self.plot(priced_puts, save_fig=True, pricer="Merton")
        self.plot(priced_puts, save_fig=True, pricer="Heston")

        print("Figures created")

        # Generate the tex document
        underlying_price = priced_calls[0].underlying_price
        self.tex_document(underlying_price)


if __name__ == "__main__":
    analyser = OptionAnalysis(expiration_date='2023-05-26', evaluation_date='2023-05-22')
    analyser.complete_analysis()
    # print(analyser.underlying)time_to_maturity
    #options = analyser.get_options()

    # options_iv = analyser.estimate_implied_volatility(options)

    #priced_options = analyser.pricing_test(options)
    # priced_options = analyser.price_option(options)
    #print(priced_options[5])
    #for option in priced_options:
    #    print(option)
    #analyser.plot(priced_options, False, 'CRR')
    #analyser.plot_iv(options_iv)
    # analyser.TexDocument()
