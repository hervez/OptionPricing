import numpy as np
import matplotlib.pyplot as plot
import statsmodels.api as sm
import math
import scipy
samplesize = 2520
sample = np.random.normal(loc=0.7, scale=0.3, size=samplesize)
""""
TO USE IT, YOU NEED AN ARRAY OF LOG RETURNS OF TIME T = 1
"""
class Calibrate:
    def __init__(self, sample):
        self.sample = sample
    def BScalibrate(self):
        mean = np.mean(self.sample)
        vol = np.std(self.sample)
        return mean,vol #Please use only vol in BSPricing, the mean is useless after we calibrated
    def normalpdf(self, x, mean, vol):
        return (1 / (vol * np.sqrt(2 * np.pi))) * np.exp(-(x - mean) ** 2 / (2 * vol ** 2))
    def Mertonpdf(self, variables, x): #"works only for t =1!!!"
        alpha, lamda,delta, mu, sigma = variables
        sum = 0
        fi = 0
        y = alpha + (delta ** 2) / 2
        k = np.exp(y) - 1
        for i in range(0,10):
            mean = mu- 0.5*sigma**2 - lamda*k + i*alpha
            vol = np.sqrt(sigma**2 + i*delta**2)
            fi = (lamda**i) * self.normalpdf(x,mean, vol ) / math.factorial(i)
            sum += fi
        sum *= np.exp(-lamda)
        return sum
    def ObjectiveMerton(self, variables):
        sum_diff = 0
        for obs in self.sample:
            merton_val = self.Mertonpdf(variables, obs)
            sample_val = 1 / len(self.sample)  # Adjust the scale of the sample value to match the density
            diff = merton_val - sample_val
            sum_diff += np.log(diff ** 2)
        return -sum_diff
    def Mertoncalibrate(self):
        initial_guess = [0, 1, 0.3, 0.7, 0.3] 
        bounds = [(-10, 10), (0, 3), (1e-10, None), (-10, 10), (1e-10, None)] 
        result = scipy.optimize.minimize(self.ObjectiveMerton, initial_guess, method='L-BFGS-B', bounds=bounds)
        optimized_variables = result.x
        print(optimized_variables)
        return optimized_variables # WILL RETURN alpha, lamda, delta, mu, sigma. we don't need mu, the rest is useful
    def plotBS(self):
        mean, vol = self.BScalibrate()
        x = np.linspace(np.min(self.sample), np.max(self.sample), 100)
        gauss = (1 / (vol * np.sqrt(2 * np.pi))) * np.exp(-(x - mean) ** 2 / (2 * vol ** 2))
        variables = self.Mertoncalibrate()
        Merton = self.Mertonpdf(variables,x)
        variables = [1,3,2,-1,2]
        truemerton = self.Mertonpdf(variables,x)
        plot.figure()
        plot.hist(self.sample,density=True, label='Random Sample',bins=500)
        plot.plot(x,gauss, label='Normal Curve')
        plot.plot(x, Merton, label='Merton Density')
        plot.plot(x,truemerton,label='true merton')
        plot.xlabel('Value')
        plot.ylabel('Density')
        plot.title('Random Sample, Merton Jump Diffusion Curve and Normal Curve')
        plot.legend()
        sm.qqplot(self.sample)
        plot.show()

ex = Calibrate(sample)
ex.plotBS()
samplesize = 2520
sample = np.random.normal(loc=0.7, scale=0.3, size=samplesize)
for i in range(0,252):
    jumpoccurence = np.random.poisson(lam = 1)
    sum=0
    for i in range(0,jumpoccurence):
        jump = np.random.normal(loc=0,scale=0.3)
        sum += jump
    sample[i] += sum
ex2 = Calibrate(sample)
ex2.plotBS()
