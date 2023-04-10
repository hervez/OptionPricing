import math
import numpy as np 
import matplotlib as mat 
from scipy.stats import norm
from scipy.fft import fft, ifft # allow us to perform Fast Fourier transform and its inverse

class OptionPricing:
  """   S0 = initial price, K = Striking price, T = option maturity (in years), r = constant short rate, sigma = constant volatility, pf is an array approximating the fx(x)  """
    def __init__(self, S0, K, T, r, sigma, pf = 0):
        self.S0 = S0 
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.pf= pf
        
    def BandSPrice(self):
        d1 = (math.log(self.S0/self.K) + (self.r + self.sigma**2)*self.T)/(self.sigma*math.sqrt(self.T))
        d2 = d1 - self.sigma*math.sqrt(self.T)
        Nd1 = norm.cdf(d1, 0, 1)
        Nd2 = norm.cdf(d2, 0, 1)
        C = self.S0*Nd1 - self.K*Nd2*math.e**(-self.r*self.T)
        return C
    