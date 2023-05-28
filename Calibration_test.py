import Calibration as cab
import numpy as np
'''
These methods were solely used in order to simulate sample following different paths.
The first one creates random sample following a normal law, the second one includes jumps
and the last one includes the volatility following a stochastic process
'''
samplesize = 1000
sample = np.random.normal(loc=0.7, scale=0.3, size=samplesize)
volatility = cab.stochastic_vol(sample)
ex = cab.CalibrateVanilla(sample, volatility)
ex.plotreturns()


samplesize = 1000
sample = np.random.normal(loc=0.7, scale=0.3, size=samplesize)
for i in range(0,1000):
    jumpoccurence = np.random.poisson(lam = 1)
    sum=0
    for i in range(0,jumpoccurence):
        jump = np.random.normal(loc=0,scale=0.3)
        sum += jump
    sample[i] += sum
volatility = cab.stochastic_vol(sample)
ex = cab.CalibrateVanilla(sample, volatility)
ex.plotreturns()

samplesize = 1000
sample = np.zeros(samplesize, dtype=float)
sample[0] = np.random.normal(0.7, 0.3)
v= 0.3
for i in range(1, samplesize):
    v = max(v + 2 * (0.3 - v) + 0.09 * np.sqrt(v) * np.random.normal(), 0.001)
    sample[i] = np.random.normal(0.7,np.sqrt(v))
results = cab.stochastic_vol(sample)
volatility = results

ex = cab.CalibrateVanilla(sample,volatility)
ex.plotreturns()


"""
If we want to price exotic options, this is a rough sketch of how to calibrate the Heston Model on the price of vanilla Call options
class CalibrateExotic:
    def __init__(self, callprices, K, T):
        self.callprices = callprices
        self.K = K
        self.T = T

   def objective_function(self, variables):
        r = 0.05  # please put the real risk-free rate
        S = 100  # please put the real stock price
        rHeston = 0.05
        rho, kappa, eta, theta = variables
        sum_diff = 0

        for i in self.callprices:
            true_call = self.callprices[i]
            estimation = Heston(S, r, self.K[i], self.T[i], self.rho, self.kappa, self.eta, self.theta)
            squared_diff = (true_call - estimation) ** 2
            sum_diff += np.log(squared_diff)
        return -sum_diff

    def heston_calibrate(self):
        initial_guess = [0, 2, 0.01, 0.01]
        bounds = [(-1, 1), (1e-10, None), (1e-10, None), (1e-10, None)]
        result = scipy.optimize.minimize(self.objective_function, initial_guess, method='L-BFGS-B', bounds=bounds)
        optimized_variables = result.x
        return optimized_variables


"""
