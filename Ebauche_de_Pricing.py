import math
import numpy as np 
import matplotlib as mat 
from scipy.stats import norm
from scipy.fft import fft, ifft # allow us to perform Fast Fourier transform and its inverse

class OptionPricing:
    """S0 = initial price, K = Striking price, T = option maturity (in years),
    r = constant short rate, sigma = constant volatility, pf is an array approximating the fx(x)
    all of which must be >0 ? """
    def __init__(self, S0, K, T, r, sigma, pf=0):
        self.S0 = S0 
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.pf= pf
        self.d1 = (math.log(self.S0/self.K) + (self.r + (self.sigma**2)/2)*self.T)/(self.sigma*math.sqrt(self.T))
        self.d2 = self.d1 - self.sigma*math.sqrt(self.T)
        self.Nd1 = norm.cdf(self.d1, 0, 1)     # N(d1)
        self.Nd2 = norm.cdf(self.d2, 0, 1)     # N(d2)
        self.Nmind1 = norm.cdf(-self.d1, 0, 1) # N(-d1)
        self.Nmind2 = norm.cdf(-self.d2, 0, 1) # N(-d2)
        
    def BSCall(self):
        
        C = self.S0*self.Nd1 - self.K*self.Nd2*math.e**(-self.r*self.T)
        return C
    def BSPut(self):
        P = self.K*self.Nmind2*math.e**(-self.r*self.T) - self.S0*self.Nmind1
        return P
''' Check up to make sure there is no mistake (could be useful if we want to do some arbitrage)
(computing the price on a website on the interent showed the same results)
    def PutCallParity(self):
        return self.BSCall() + self.K*math.e**(-self.r*self.T) == self.BSPut() + self.S0 '''
    
