import numpy as np
import matplotlib.pyplot as plot
import statsmodels.api as sm
import math
import scipy
from arch import arch_model
""""
TO USE IT, YOU NEED AN ARRAY OF LOG RETURNS OF TIME T = 1
"""
def stochastic_vol(returns):
    model = arch_model(returns, vol='Garch', p=1, q=1)
    results = model.fit()
    return results


class CalibrateVanilla:
    def __init__(self, sample, volatility=None):
        self.sample = sample
        self.volatility = volatility
        self.retours = []
    def BScalibrate(self):
        mean = np.mean(self.sample)
        vol = np.std(self.sample)
        return mean,vol
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
        print("Merton", optimized_variables)
        return optimized_variables
    def simplereturns(self): 
        Qt = []
        for i in range(len(self.sample)):
            Qt.append(np.exp(self.sample[i]))
        self.retours = Qt

    def Hestonpdf(self,variables,i):
        mu, rho, kappa, eta, theta = variables
        if i == 0:
            vol = self.volatility[i]**2
        else:
            vol = self.volatility[i-1]**2
        Qt1 = self.retours[i+1]
        Vt1 = self.volatility[i+1]
        multiplier = 1 / (2 * math.pi * theta * vol * np.sqrt(1 - rho**2))
        exp1 = np.exp(-((Qt1 - 1 - mu)**2) / (2 * Vt1 * (1 - rho**2)))
        try:
            exp2 = np.exp(rho * (Qt1 - 1 - mu) * (Vt1 - vol - eta * kappa + kappa * vol) / (vol * theta * (1 - rho**2)))
        except:
            exp2=1
        finally:
            exp3 = np.exp( - ((Vt1-vol - eta * kappa + kappa * vol)**2) / (2 * theta**2 * vol * (1 - rho**2)))
            return multiplier*exp1*exp2*exp3
    def HestonObjective(self,variables):
        sum_diff = 0
        for i,obs in enumerate(self.sample[:-1]):
            heston_val = self.Hestonpdf(variables, i)
            sample_val = 1 / (len(self.sample)-1) 
            diff = heston_val - sample_val
            sum_diff += np.log(diff ** 2)
        return -sum_diff
    def HestonCalibrate(self):
        self.simplereturns()
        initial_guess = [0.7, 0, 2, 0.5, 0.5] 
        bounds = [(None,None), (-1+1e-10, 1-1e-10), (1e-10, None), (1e-10, None), (1e-10, None)] 
        result = scipy.optimize.minimize(self.HestonObjective , initial_guess, method='L-BFGS-B', bounds=bounds)
        optimized_variables = result.x
        print("Heston", optimized_variables)
        return optimized_variables #Drop the mu after this
    def plotBS(self):
        mean, vol = self.BScalibrate()
        x = np.linspace(np.min(self.sample), np.max(self.sample), 100)
        gauss = (1 / (vol * np.sqrt(2 * np.pi))) * np.exp(-(x - mean) ** 2 / (2 * vol ** 2))
        variables = self.Mertoncalibrate()
        Merton = self.Mertonpdf(variables,x)
        Heston = self.HestonCalibrate()
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


samplesize = 200
sample = np.random.normal(loc=0.7, scale=0.3, size=samplesize)
results = stochastic_vol(sample)
volatility = results.conditional_volatility
print(volatility)
ex = CalibrateVanilla(sample,volatility)
ex.plotBS()


samplesize = 200
sample = np.random.normal(loc=0.7, scale=0.3, size=samplesize)
for i in range(0,200):
    jumpoccurence = np.random.poisson(lam = 1)
    sum=0
    for i in range(0,jumpoccurence):
        jump = np.random.normal(loc=0,scale=0.3)
        sum += jump
    sample[i] += sum
results = stochastic_vol(sample)
volatility = results.conditional_volatility
ex = CalibrateVanilla(sample,volatility)
ex.plotBS()

samplesize = 900
sample = np.linspace(-1,samplesize)
sample[0] = np.random.normal(0.7, 0.3)
v= 0.3
for i in range(1, len(sample)):
    v = 0.5*v + 0.5*np.random.chisquare(1)
    sample[i] = sample[0] = np.random.normal(0.7,np.sqrt(v))
sample = sample[np.isfinite(sample)]
results = stochastic_vol(sample)
volatility = results.conditional_volatility
ex = CalibrateVanilla(sample,volatility)
ex.plotBS()


""""
If we want to price exotic options, this could be a nice way to calibrate the Heston Model
class CalibrateExotic:
    def __init__(self, callprices, K, T):
        self.callprices= callprices
        self.K = K
        self.T = T
    def objectivefunction(self, variables):
        r = 0.05 #please put the real riskfree rate
        S = 100 #please put the real stock price
        rHeston = 0.05
        sigmaHeston, self.rho, self.kappa, self.eta, self.theta = variables
        sum_diff = 0
        for i in self.callprices:
            truecall= self.callprices[i]
            estimation = Heston(S,r, self.K[i], self.T[i],sigmaHeston,rho, kappa, eta, theta) #We would have to put the function to calculate heston's price here
            squared_diff = (truecall-estimation)**2
            sum_diff += np.log(squared_diff)
        return -sum_diff
    def Hestoncalibrate(self):
        initial_guess = [0.01, 0, 2, 0.01, 0.01] 
        bounds = [(1e-10, None), (-1, 1), (1e-10, None), (1e-10, None), (1e-10, None)] 
        result = scipy.optimize.minimize(self.objectivefunction, initial_guess, method='L-BFGS-B', bounds=bounds)
        optimized_variables = result.x
        print(optimized_variables)
        return optimized_variables


"""
