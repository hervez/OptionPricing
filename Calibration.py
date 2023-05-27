import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
import scipy
from arch import arch_model

""""
TO USE IT, YOU NEED AN ARRAY OF LOG RETURNS OF TIME T = 1
"""
def stochastic_vol(returns):
    """"
    Uses GARCH to estimate volatility for each day in our sample.
    You can use it if you want to calibrate the Heston Model 
    BUT you need to have time series, you cannot just pick any random sample
    """
    model = arch_model(returns, vol='Garch', p=1, q=1)
    results = model.fit(show_warning=False)
    volatility = results.conditional_volatility
    return volatility


class CalibrateVanilla:
    """
    General interface for calibrating vanilla options
    """
    def __init__(self, sample, volatility=None):
        """
        Args:
            Sample: array containing the daily log returns of an asset, ln(St+1/St)
            volatility: array containing the volatility of the log returns of an asset.
                        Use it only if Sample is a time series
            retours: will be the returns of an asset St, St+1/St
        """
        self.sample = sample
        self.volatility = volatility
        self.retours = []

    def bscalibrate(self):
        """
        Returns the mean and volatility of an asset's log return
        """
        mean = np.mean(self.sample)
        vol = np.std(self.sample)
        return mean, vol #The volatility is used to price the classical BS model, the mean was just useful for the graph in plot returns
    
    def normalpdf(self, x, mean, vol):
        """
        Return the Normal Law's pdf, given a mean, a volatility and a value x.
        """
        return (1 / (vol * np.sqrt(2 * np.pi))) * np.exp(-(x - mean) ** 2 / (2 * vol ** 2))
    
    def mertonpdf(self, variables, x): #"works only for t = 1!!!"
        """
        Gives the pdf of a Merton Jump Diffusion model, given a set of variables and x
        !!! Here, I have chosen t = 1 in order to simplify the calculation.
        The t was dropped in order to ease the calculation, the lambda's bound constraint in merton_calibrate(self) should change:
        This formula should be rewritten as it is in the following paper for another t != 1 (page 725)
        https://www.ijpam.eu/contents/2016-109-3/19/19.pdf  
        The varaibles are the following:
            Alpha : average of the log jump
            lamda : average number of jumps
            delta : volatility of the log jumps
            mu : stock return on average. You never use it when you price the option but it is needed for calibration
            sigma : volatility of the Brownian motion of the stock
        """
        alpha, lamda, delta, mu, sigma = variables
        sum = 0
        fi = 0
        y = alpha + (delta ** 2) / 2
        k = np.exp(y) - 1
        for i in range(0, 10):
            mean = mu - 0.5 * sigma ** 2 - lamda * k + i * alpha
            vol = np.sqrt(sigma ** 2 + i * delta ** 2)
            fi = (lamda ** i) * self.normalpdf(x, mean, vol) / math.factorial(i)
            sum += fi
        sum *= np.exp(-lamda)
        return sum
    
    def objective_merton(self, variables):
        """
        Log-likelihood function
        """
        sum_diff = 0
        for obs in self.sample:
            merton_val = self.mertonpdf(variables, obs)
            sample_val = 1 / len(self.sample)  # Adjust the scale of the sample value to match the density
            diff = merton_val - sample_val
            sum_diff += np.log(diff ** 2)
        return - sum_diff
    
    def merton_calibrate(self):
        """
        Goal: we want to maximize the log-likelihood function (minimize -1*log-likelihood function)
        Returns: alpha, lamda, delta, mu, sigma in that order
        """
        initial_guess = [0, 1, 0.3, np.mean(self.sample), np.std(self.sample)]
        bounds = [(-10, 10), (0, 3), (1e-10, None), (-10, 10), (1e-10, None)]
        result = scipy.optimize.minimize(self.objective_merton, initial_guess, method='L-BFGS-B', bounds=bounds)
        optimized_variables = result.x
        return optimized_variables
    
    def simplereturns(self):
        """
        Goal: fill self.retours with the returns of the assets"""
        Qt = []
        for i in range(len(self.sample)):
            Qt.append(np.exp(self.sample[i]))
        self.retours = Qt

    def hestonpdf(self, variables, i):
        """
        Gives the pdf of the Heston model for the next movement of the stock, given a set of variables and x 
        The variables are the following:
            Mu : Stock's log returns
            Rho : Correlation coefficient between the Brownian motion of the stock and the BM of the variance
            Kappa : rate of reversion to the long-term price variance
            Eta : long-term price variance
            Theta : volatility of the volatility
        This method isn't reliable and can sometimes compute weird estimations.
        The formula comes from this paper at page 7: https://www.valpo.edu/mathematics-statistics/files/2015/07/Estimating-Option-Prices-with-Heston%E2%80%99s-Stochastic-Volatility-Model.pdf
        The Heston method is often calibrated with the price of vanilla options and used for more exotic options
        """
        mu, rho, kappa, eta, theta = variables
        if i == 0:
            vol = self.volatility[i] ** 2
        else:
            vol = self.volatility[i - 1] ** 2
        Qt1 = self.retours[i + 1]
        Vt1 = self.volatility[i + 1]
        multiplier = 1 / (2 * math.pi * theta * vol * np.sqrt(1 - rho ** 2))
        exp1 = np.exp(-((Qt1 - 1 - mu) ** 2) / (2 * Vt1 * (1 - rho ** 2)))
        try:
            exp2 = np.exp(rho * (Qt1 - 1 - mu) * (Vt1 - vol - eta * kappa + kappa * vol) / (vol * theta * (1 - rho ** 2)))
        except:
            exp2 = 1
        finally:
            exp3 = np.exp(-((Vt1 - vol - eta * kappa + kappa * vol) ** 2) / (2 * theta ** 2 * vol * (1 - rho ** 2)))
            return multiplier * exp1 * exp2 * exp3
        
    def heston_objective(self, variables):
        """
        Log-likelihood function
        """
        sum_diff = 0
        for i, obs in enumerate(self.sample[:-1]):
            heston_val = self.hestonpdf(variables, i)
            sample_val = 1 / (len(self.sample) - 1)
            diff = heston_val - sample_val
            sum_diff += np.log(diff ** 2)
        return -sum_diff

    def heston_calibrate(self):
        """
        Goal: we want to maximize the log-likelihood function (minimize -1*log-likelihood function)
        Returns: mu, rho, kappa, eta, theta in that order
        """
        self.simplereturns()
        initial_guess = [np.mean(self.sample), 0, 2, 0.5, np.var(self.sample)]
        bounds = [(None, None), (-1 + 1e-10, 1 - 1e-10), (1e-10, None), (1e-10, None), (1e-10, None)]
        result = scipy.optimize.minimize(self.heston_objective, initial_guess, method='L-BFGS-B', bounds=bounds)
        optimized_variables = result.x
        return optimized_variables  # Drop the mu after this
    
    def plotreturns(self):
        """
        Function used in order to get nice graphics for the paper
        """
        mean, vol = self.bscalibrate()
        x = np.linspace(-2, 5, 100)
        gauss = self.normalpdf(x, mean, vol)
        variables = self.merton_calibrate()
        merton = self.mertonpdf(variables, x)
        if self.volatility.any() :
            heston = self.heston_calibrate()
            print(heston)
        plt.figure()
        plt.hist(self.sample, density=True, range=(-2, 5), label='Random Sample', bins=100)
        plt.plot(x, gauss, label='Normal Curve')
        plt.plot(x, merton, label='Merton Density')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Random Sample, Merton Jump Diffusion Curve and Normal Curve')
        plt.legend()
        sm.qqplot(self.sample)
        plt.show()


