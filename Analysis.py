import datetime
from typing import List

from OptionDataType import OptionData
from DataGathering import OptionDataGathering
from ImpliedVolatility import OptionImpliedVolatility
from PricingIncludingMerton import OptionPricingBlackScholesMerton
from GraphicalAnalysis import OptionGraphicalAnalysis
from TexDocumentCreator import TexDocument


class OptionAnalysis():

    def __init__(self, underlying: str = 'AAPL', expiration_date: str = '2023-04-21',
                 evaluation_date: str = datetime.datetime.today().strftime('%Y-%m-%d')):
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
            estimated_implied_volatility = implied_vol_surface.get_implied_volatility(option.strike_price, option.time_to_maturity)
            option.estimated_implied_vol = estimated_implied_volatility

        return options_list

    def price_option(self, options_list: List[OptionData]):
        for option in options_list:
            pricer = OptionPricingBlackScholesMerton(S_0=option.underlying_price, K=option.strike_price,
                                                     T=option.time_to_maturity, r=option.risk_free_rate,
                                                     sigma=option.implied_volatility)
            if option.option_type == 'call':
                option.BS_pricing = pricer.get_call()
            if option.option_type == 'put':
                option.BS_pricing = pricer.get_put()

        return options_list

    def plot(self, option_list: List[OptionData]):

        path = "./results/" + self.underlying
        self.ploter.plot_price_strikes(option_list)

    def plot_iv(self, option_list: List[OptionData]):

        path = "./results/" + self.underlying
        self.ploter.plot_implied_volatility_2D(option_list)

    def TexDocument(self):

        option_type = 'call'
        option_df = self.gatherer.get_option_data(self.underlying, option_type, self.expiration_date)
        texer = TexDocument(self.underlying, self.evaluation_date, option_df)
        texer.generate_document()

if __name__ == "__main__":
    analyser = OptionAnalysis(expiration_date='2023-05-19', evaluation_date='2023-05-11')
    # print(analyser.underlying)time_to_maturity
    options = analyser.get_options()
    options_iv = analyser.estimate_implied_volatility(options)
    priced_options = analyser.price_option(options)
    #for option in options_iv:
    #    print(option)
    #analyser.plot(priced_options)
    analyser.plot_iv(options_iv)
    # analyser.TexDocument()