from typing import List

from OptionDataType import OptionData
from DataGathering import OptionDataGathering
from ImpliedVolatility import OptionImpliedVolatility
from Pricing import *
from GraphicalAnalysis import OptionGraphicalAnalysis
from TexDocumentCreator import TexDocument


class OptionAnalysis:

    def __init__(self, underlying: str = 'AAPL', expiration_date: str = '2023-04-21',
                 evaluation_date: str = datetime.today().strftime('%Y-%m-%d')):
        self.underlying = underlying
        self.expiration_date = expiration_date
        self.gatherer = OptionDataGathering(verbose=False, reload=True)
        self.evaluation_date = evaluation_date
        self.ploter = OptionGraphicalAnalysis(self.underlying)

    def get_options(self, option_type: str = 'call'):
        underlying_price = self.gatherer.get_underlying_value_at_evaluation_date(self.underlying, self.evaluation_date)
        risk_free_rate = self.gatherer.get_risk_free_rate(self.evaluation_date)
        time_to_maturity = self.gatherer.time_to_maturity(self.expiration_date, self.evaluation_date)
        historical_volatility = self.gatherer.get_historical_volatility(self.underlying, self.evaluation_date)

        option_df = self.gatherer.get_option_data(self.underlying, option_type, self.expiration_date)
        option_list = []
        for index, row in option_df.iterrows():
            option = OptionData(
                option_type=option_type,
                option_price=row['lastPrice'],
                strike_price=row['strike'],
                implied_volatility=row['impliedVolatility'],
                historical_volatility=historical_volatility,
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
        implied_vol_surface = OptionImpliedVolatility(self.underlying)
        for option in options_list:
            estimated_implied_volatility = implied_vol_surface.get_implied_volatility(option.strike_price,
                                                                                      option.time_to_maturity)
            option.estimated_implied_vol = estimated_implied_volatility

        return options_list

    @staticmethod
    def price_option(options_list: List[OptionData]):
        for option in options_list:
            # BS pricing
            pricer = OptionPricerBlackScholes(S_0=option.underlying_price, K=option.strike_price,
                                               T=option.time_to_maturity, r=option.risk_free_rate,
                                               sigma=option.implied_volatility)
            if option.option_type == 'call':
                option.BS_pricing = pricer.get_call()
            if option.option_type == 'put':
                option.BS_pricing = pricer.get_put()

            # CRR pricing
            pricerCRR = OptionPricerCRR(S_0=option.underlying_price, K=option.strike_price,
                                        T=option.time_to_maturity, r=option.risk_free_rate,
                                        sigma=option.implied_volatility, M=100)
            if option.option_type == 'call':
                option.CRR_pricing = pricerCRR.get_call()
            if option.option_type == 'put':
                option.CRR_pricing = pricerCRR.get_put()

            # Merton pricing
            #pricerMerton = OptionPricerMerton(S_0=option.underlying_price, K=option.strike_price,
            #                            T=option.time_to_maturity, r=option.risk_free_rate,
            #                            sigma=option.implied_volatility)
            #if option.option_type == 'call':
            #    option.BSM_pricing = pricerMerton.get_call()
            #if option.option_type == 'put':
            #    option.BSM_pricing = pricerCRR.get_put()
            #print("BSM priced")

            # Houston pricing
            #pricerFourier = OptionPricerFourierPricing(S_0=option.underlying_price, K=option.strike_price,
            #                                  T=option.time_to_maturity, r=option.risk_free_rate,
            #                                  sigma=option.implied_volatility)
            #if option.option_type == 'call':
            #    option.Fourier_pricing = pricerFourier.get_call()
            #if option.option_type == 'put':
            #    option.Fourier_pricing = pricerFourier.get_put()
            #print("Fourier priced")

        return options_list

    @staticmethod
    def pricing_test(options_list):
        for option in options_list:
            pricerCRR = OptionPricerCRR(S_0=option.underlying_price, K=option.strike_price,
                                     T=option.time_to_maturity, r=option.risk_free_rate,
                                     sigma=option.implied_volatility, M=100)
            if option.option_type == 'call':
                option.CRR_pricing = pricerCRR.get_call()
            if option.option_type == 'put':
                option.CRR_pricing = pricerCRR.get_put()

        return options_list

    def plot(self, option_list: List[OptionData], save_fig=False):

        self.ploter.plot_price_strikes(option_list=option_list, save_fig=save_fig)

    def plot_iv(self, option_list: List[OptionData]):

        self.ploter.plot_implied_volatility_2D(option_list)

    def tex_document(self, underlying_price: float):

        texer = TexDocument(self.underlying, self.evaluation_date, self.expiration_date,
                            underlying_price=underlying_price)
        texer.generate_document()

    def complete_analysis(self):
        # Get the options
        calls = self.get_options('call')
        puts = self.get_options('put')

        # Price the option
        priced_calls = self.price_option(calls)
        priced_puts = self.price_option(puts)

        # Create the figure
        self.plot(priced_calls, save_fig=True)
        self.plot(priced_puts, save_fig=True)

        # Generate the tex document
        underlying_price = priced_calls[0].underlying_price
        self.tex_document(underlying_price)


if __name__ == "__main__":
    analyser = OptionAnalysis(expiration_date='2023-06-23', evaluation_date='2023-05-19')
    #analyser.complete_analysis()
    # print(analyser.underlying)time_to_maturity
    options = analyser.get_options()
    # options_iv = analyser.estimate_implied_volatility(options)
    priced_options = analyser.price_option(options)
    print(priced_options[5])
    for option in priced_options:
       print(option)
    analyser.plot(priced_options)
    # analyser.plot_iv(options_iv)
    # analyser.TexDocument()
