from datetime import datetime
import math
import numpy as np 
import matplotlib as mat 
from scipy.stats import norm

from DataGathering import OptionDataGathering


class OptionPricingBlackScholes(): 

    def __init__(self, underlying: str, option_type: str, expiration_date: str, evaluation_date: str): 

        self.underlying = underlying 
        self.option_type = option_type 
        self.expiration_date = expiration_date 
        self.evaluation_date = evaluation_date

        gatherer = OptionDataGathering(self.underlying, self.option_type, self.expiration_date, self.evaluation_date, verbose=False)

        self.S_0 = gatherer.underlying_value_at_evaluation_date
        self.r = gatherer.risk_free_rate
        self.K = 100 
        self.T = int((datetime.strptime(self.expiration_date, '%Y-%m-%d') - datetime.strptime(self.evaluation_date, '%Y-%m-%d')).days)
        self.sigma = gatherer.historical_volatility

        self.d1 = (math.log(self.S_0/self.K) + (self.r + (self.sigma**2)/2)*self.T)/(self.sigma*math.sqrt(self.T))
        self.d2 = self.d1 - self.sigma*math.sqrt(self.T)
        self.Nd1 = norm.cdf(self.d1, 0, 1)     # N(d1)
        self.Nd2 = norm.cdf(self.d2, 0, 1)     # N(d2)
        self.Nmind1 = norm.cdf(-self.d1, 0, 1) # N(-d1)
        self.Nmind2 = norm.cdf(-self.d2, 0, 1) # N(-d2)
    
    def get_call(self):
   
        C = self.S_0*self.Nd1 - self.K*self.Nd2*math.e**(-self.r*self.T)

        return C

    def get_put(self):

        P = self.K*self.Nmind2*math.e**(-self.r*self.T) - self.S_0*self.Nmind1

        return P


if __name__ == '__main__': 

    
    black_scholes_pricer = OptionPricingBlackScholes('AAPL', 'call', '2023-04-21', '2023-04-10')
    call = black_scholes_pricer.get_call()
    put = black_scholes_pricer.get_put()
    print(call)
    print(put)