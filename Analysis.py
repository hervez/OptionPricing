import datetime
from typing import List

from OptionDataType import OptionData
from DataGathering import OptionDataGathering
from ImpliedVolatility import OptionImpliedVolatility
from PricingIncludingMerton import OptionPricingBlackScholesMerton
from GraphicalAnalysis import OptionGraphicalAnalysis


class OptionAnalysis():

    def __init__(self, underlying: str = 'AAPL', expiration_date: str = '2023-04-21'):
        self.underlying = underlying
        self.expiration_date = expiration_date
        self.gatherer = OptionDataGathering( verbose=False)
        self.evaluation_date = datetime.datetime.today().strftime('%Y-%m-%d')
        self.ploter = OptionGraphicalAnalysis()

    def get_options(self, option_type: str = 'call'):
        underlying_price = self.gatherer.get_underlying_value_at_evaluation_date(self.underlying, self.evaluation_date)
        risk_free_rate = self.gatherer.get_risk_free_rate(self.evaluation_date)
        time_to_maturity = self.gatherer.time_to_maturity(self.expiration_date, self.evaluation_date)
        historical_volatility = self.gatherer.get_historical_volatility(self.underlying, self.evaluation_date)
        # implied_vol_surface = OptionImpliedVolatility(self.underlying)

        option_df = self.gatherer.get_option_data(self.underlying, option_type, self.expiration_date)
        option_list = []
        for index, row in option_df.iterrows():
            # estimated_implied_volatility =  implied_vol_surface.get_implied_volatility(row['strike'], time_to_maturity)
            option = OptionData(
                option_type = option_type,
                option_price= row['lastPrice'],
                strike_price= row['strike'],
                implied_volatility= row['impliedVolatility'],
                historical_volatility=historical_volatility,
                underlying_price=underlying_price,
                underlying=self.underlying,
                evaluation_date=self.evaluation_date,
                expiration_date=self.expiration_date,
                risk_free_rate= risk_free_rate,
                time_to_maturity=time_to_maturity,
                # estimated_implied_vol= estimated_implied_volatility
            )

            option_list.append(option)

        return option_list

    def price_option(self, options_list: List[OptionData]):
        for option in options_list:
            pricer = OptionPricingBlackScholesMerton(self.underlying, option.expiration_date, option.evaluation_date, option.strike_price)
            if option.option_type == 'call':
                option.BS_pricing = pricer.get_call()
            if option.option_type == 'put':
                option.BS_pricing = pricer.get_put()

        return options_list

    def plot(self, option_list : List[OptionData]):

        self.ploter.plot_price_strikes(option_list)

if __name__ == "__main__":
    analyser = OptionAnalysis(expiration_date='2023-05-12')
    #print(analyser.underlying)
    options = analyser.get_options()
    priced_options = analyser.price_option(options)
    analyser.plot(priced_options)
    # for option in priced_options:
    #     print(option)