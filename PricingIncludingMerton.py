from datetime import datetime
import math
import numpy as np 
import matplotlib as mat 
from scipy.stats import norm
from scipy import integrate
import cmath

from DataGathering import OptionDataGathering


class OptionPricingBlackScholesMerton(): 

    def __init__(self, underlying: str, expiration_date: str, evaluation_date: str, strike: int):

        self.underlying = underlying 
        # self.option_type = option_type
        self.expiration_date = expiration_date 
        self.evaluation_date = evaluation_date

        gatherer = OptionDataGathering(verbose=False)

        self.S_0 = gatherer.get_underlying_value_at_evaluation_date(self.underlying, self.evaluation_date)
        self.r = gatherer.get_risk_free_rate(self.evaluation_date)
        self.K = strike
        self.T = int((datetime.strptime(self.expiration_date, '%Y-%m-%d') - datetime.strptime(self.evaluation_date, '%Y-%m-%d')).days)
        self.sigma = gatherer.get_historical_volatility(self.underlying, self.evaluation_date) # Historical volx
        # self.sigma = gatherer.get_implied_volatility(self.underlying, self.strike, self.expiration_date) # Implied vol
        #For BS:
        self.d1 = (math.log(self.S_0/self.K) + (self.r + (self.sigma**2)/2)*self.T)/(self.sigma*math.sqrt(self.T))
        self.d2 = self.d1 - self.sigma*math.sqrt(self.T)
        self.Nd1 = norm.cdf(self.d1, 0, 1)     # N(d1)
        self.Nd2 = norm.cdf(self.d2, 0, 1)     # N(d2)
        self.Nmind1 = norm.cdf(-self.d1, 0, 1) # N(-d1)
        self.Nmind2 = norm.cdf(-self.d2, 0, 1) # N(-d2)
        #For Merton :
        self.alpha = 0.1
        self.lamda = 1 # If lambda = 0, we have the traditinal BS case
        self.delta = 0.04
        self.y = self.alpha + (self.delta**2)/2
        self.k = math.exp(self.y)-1
        self.lambdaprime = self.lamda*(1+self.k)
        #For Fourier Pricing methods:
        self.mu = self.r - 0.5 * (self.sigma**2)*self.T
        self.vol = self.sigma*math.sqrt(self.T)

    
    def get_call(self):
   
        C = self.S_0*self.Nd1 - self.K*self.Nd2*math.e**(-self.r*self.T)

        return C

    def get_put(self):

        P = self.K*self.Nmind2*math.e**(-self.r*self.T) - self.S_0*self.Nmind1

        return P
    def get_MertonLognormalJumpCall(self):
        Fpoisson= 0
        i = 0
        s=0
        while Fpoisson< 0.999: #0.999 is arbitrary, the closer to 1, the better, but it becomes computationally expensive
            ''' I calculate the probability that the number of jumps = i, then sigma_n and r_n, and put
            them in the Black and Scholes formula and I get the sum of P(N=i)*(BSCall) from 0 to
            a large enough number'''
            fpoisson = (self.lambdaprime*self.T)**i*(math.e**(-self.lambdaprime*self.T))/math.factorial(i)
            Fpoisson += fpoisson
            r_n = self.r -self.lamda*self.k + i*self.y/self.T
            sigma_n = math.sqrt(self.sigma**2 + i*(self.delta**2)/self.T)
            d1_n = (math.log(self.S_0/self.K) + (r_n + (sigma_n**2)/2)*self.T)/(sigma_n*math.sqrt(self.T))
            d2_n = d1_n - sigma_n*math.sqrt(self.T)
            Nd1_n = norm.cdf(d1_n, 0, 1)     #N(d1_n)
            Nd2_n = norm.cdf(d2_n, 0, 1)     # N(d2_n)
            cn = self.S_0*Nd1_n - self.K*Nd2_n*math.e**(-r_n*self.T)
            s += cn*fpoisson
        return s
    def get_MertonLognormalJumpPut(self):
        Fpoisson= 0
        i = 0
        s=0
        while Fpoisson< 0.999: #0.999 is arbitrary, the closer to 1, the better, but it becomes computationally expensive
            ''' I calculate the probability that the number of jumps = i, then sigma_n and r_n, and put
            them in the Black and Scholes formula and I get the sum of P(N=i)*(BSCall) from 0 to
            a large enough number'''
            fpoisson = (self.lambdaprime*self.T)**i*(math.e**(-self.lambdaprime*self.T))/math.factorial(i)
            Fpoisson += fpoisson
            r_n = self.r -self.lamda*self.k + i*self.y/self.T
            sigma_n = math.sqrt(self.sigma**2 + i*(self.delta**2)/self.T)
            d1_n = (math.log(self.S_0/self.K) + (r_n + (sigma_n**2)/2)*self.T)/(sigma_n*math.sqrt(self.T))
            d2_n = d1_n - sigma_n*math.sqrt(self.T)
            Nmind1_n = norm.cdf(-d1_n, 0, 1) # N(-d1_n)
            Nmind2_n = norm.cdf(-d2_n, 0, 1) # N(-d2_n)
            p_n = self.K*Nmind2_n*math.e**(-r_n*self.T) - self.S_0*Nmind1_n
            s += p_n*fpoisson
        return s
    # Here lie the characteristic functions:
    def Brownian_Motion_cf(self,u): # cf(Normal law) 
        return np.exp(1j*u*(self.mu + math.log(self.S_0)) - 0.5*self.vol**2*u**2)
    # Here lies the semi-analytical method after founding the characteristic function of ln(ST)
    def delta_option(self, cf): #Need to specify characteristic function
        f = lambda u: np.real( (np.exp(-u*math.log(self.K)*1j)  * cf(u-1j) / (u*1j)) / cf(-1j) ) 
        F = integrate.quad(f, 0, np.inf, limit = 10000)
        print(F)
        print(self.Nd1)
        return 1/2 + F[0]/math.pi
    def prob2(self,cf): #Probability of finishing in-the-money #Need to specify the characteristic function
        f = lambda u:  np.real( np.exp(-u*math.log(self.K)*1j) /(u*1j) * cf(u) )
        F = integrate.quad(f, 0, np.inf, limit = 10000)
        print(F)
        print(self.Nd2)
        return 1/2 + F[0]/math.pi
    # Here lie the functions that produce an output depending on what type of Fourier-priced option is called
    def get_call_usingFourier(self, cf):
              return self.S_0*self.delta_option(cf) - self.prob2(cf)*self.K*math.exp(-self.r*self.T)
    def get_Call_BS_Fourier(self):
              c = self.get_call_usingFourier(self.Brownian_Motion_cf)
              return c
    

if __name__ == '__main__': 

    
    black_scholes_pricer = OptionPricingBlackScholesMerton('AAPL', 'call', '2023-04-21', '2023-04-10', 100)
    call = black_scholes_pricer.get_call()
    merton_log_call = black_scholes_pricer.get_MertonLognormalJumpCall()
    put = black_scholes_pricer.get_put()
    merton_log_put = black_scholes_pricer.get_MertonLognormalJumpPut()
    print(call)
    print(merton_log_call) # should be greater or equal to the BS call
    print(put)
    print(merton_log_put) # Should be greater or equal to the BS Put
    #Double check values with https://demonstrations.wolfram.com/OptionPricesInMertonsJumpDiffusionModel/
