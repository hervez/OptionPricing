import math
import numpy as np
from numpy.fft import fft
from scipy.stats import norm
from scipy import integrate
from abc import ABC, abstractmethod


class OptionPricer(ABC):
    """
    General interface of the class for pricing vanilla options.
    """

    def __init__(self, S_0: float, K: float, T: int, r: float, sigma: float):
        """
        Args:
            S_0: Price of the underlying of the option at the date of the pricing
            K: strike price of the option
            T: time to maturity in days
            r: risk-free rate
            sigma: standard deviation of the underlying of the option
        """
        self.S_0 = S_0
        self.r = r
        self.K = K
        self.T = T
        self.sigma = sigma

    @abstractmethod
    def get_call(self):
        """ Returns: the price of a call with the given pricer """
        pass

    @abstractmethod
    def get_put(self):
        """ Returns: the price of a put with the given pricer """
        pass


class OptionPricerBlackScholes(OptionPricer):
    """
    Option pricing with basic Black and Scholes model
    """

    def __init__(self, S_0: float, K: float, T: int, r: float, sigma: float):
        super().__init__(S_0=S_0, K=K, T=T, r=r, sigma=sigma)

        # For BS:
        self.d1 = (math.log(self.S_0 / self.K) + (self.r + ((self.sigma ** 2) / 2)) * self.T) / (
                self.sigma * math.sqrt(self.T))
        self.d2 = self.d1 - (self.sigma * math.sqrt(self.T))
        self.Nd1 = norm.cdf(self.d1, 0, 1)  # N(d1)
        self.Nd2 = norm.cdf(self.d2, 0, 1)  # N(d2)
        self.Nmind1 = norm.cdf(-self.d1, 0, 1)  # N(-d1)
        self.Nmind2 = norm.cdf(-self.d2, 0, 1)  # N(-d2)

    def get_call(self):
        C = self.S_0 * self.Nd1 - self.K * self.Nd2 * math.e ** (- self.r * self.T)

        return C

    def get_put(self):
        P = self.K * self.Nmind2 * math.e ** (-self.r * self.T) - self.S_0 * self.Nmind1

        return P


class OptionPricerCRR(OptionPricer):
    """
    Option pricing with the Cox-Rox-Rubinstein binomial model
    """

    def __init__(self, S_0: float, K: float, T: int, r: float, sigma: float, M: int):
        """
        Args:
            M: number of steps for the binomial process
        """

        super().__init__(S_0=S_0, K=K, T=T, r=r, sigma=sigma)

        self.M = M  # Number of time intervals
        self.dt = T / self.M  # length of time interval
        self.df = math.exp(-r * self.dt)  # discount factor per interval

        # Binomial Parameters
        self.u = math.exp(sigma * math.sqrt(self.dt))  # up movement
        self.d = 1 / self.u  # down movement
        self.q = (math.exp(r * self.dt) - self.d) / (self.u - self.d)  # martingale branch probability
        # Array Initialization for Index Levels
        self.mu = np.arange(self.M + 1)
        self.mu = np.resize(self.mu, (self.M + 1, self.M + 1))
        self.md = np.transpose(self.mu)
        self.mu = self.u ** (self.mu - self.md)
        self.md = self.d ** self.md
        self.S = self.S_0 * self.mu * self.md

    def get_call(self):

        V = np.maximum(self.S - self.K, 0)  # values of the European call option

        z = 0
        for t in range(self.M - 1, -1, -1):
            V[0:self.M - z, t] = (self.q * V[0:self.M - z, t + 1] + (1 - self.q) * V[1:self.M - z + 1, t + 1]) * self.df
            z += 1
        # print(V[0,0])

        return V[0, 0]

    def get_put(self):

        V = np.maximum(self.K - self.S, 0)  # values of the European call option

        z = 0
        for t in range(self.M - 1, -1, -1):
            V[0:self.M - z, t] = (self.q * V[0:self.M - z, t + 1] + (1 - self.q) * V[1:self.M - z + 1, t + 1]) * self.df
            z += 1

        return V[0, 0]


class OptionPricerFourier(OptionPricer):
    """
    Option pricing with the Lewis-Fourier model
    """

    def __init__(self, S_0: float, K: float, T: int, r: float, sigma: float):
        super().__init__(S_0=S_0, K=K, T=T, r=r, sigma=sigma)

    def get_call(self):
        integral_value = \
            integrate.quad(lambda u: self.BSM_integral_function(u, self.S_0, self.K, self.T, self.r, self.sigma), 0,
                           100)[0]
        call = max(0, self.S_0 - np.exp(-self.r * self.T) * np.sqrt(self.S_0 * self.K) / np.pi * integral_value)

        return call

    def get_put(self):
        call = self.get_call()
        put = call - self.S_0 + self.K * np.exp(-self.r * self.T)  # Computed using put-call parity

        return put

    def BSM_integral_function(self, u, S_0, K, T, r, sigma):
        cf_value = self.BSM_characteristic_function(u - 1j * 0.5, 0.0, T, r, sigma)
        integral_value = 1 / (u ** 2 + 0.25) * (np.exp(1j * u * np.log(S_0 / K)) * cf_value).real

        return integral_value

    @staticmethod
    def BSM_characteristic_function(v, x0, T, r, sigma):
        cf_value = np.exp(((x0 / T + r - 0.5 * sigma ** 2) * 1j * v - 0.5 * sigma ** 2 * v ** 2) * T)

        return cf_value


class OptionPricerFFT(OptionPricer):
    """
    Option pricing with Fast Fourier Transform
    """

    def __init__(self, S_0: float, K: float, T: int, r: float, sigma: float):

        super().__init__(S_0=S_0, K=K, T=T, r=r, sigma=sigma)

    def get_call(self):
        k = np.log(self.K / self.S_0)
        x0 = np.log(self.S_0 / self.S_0)
        g = 1  # factor to increase accuracy
        N = g * 4096
        eps = (g * 150.) ** -1
        eta = 2 * np.pi / (N * eps)
        b = 0.5 * N * eps - k
        u = np.arange(1, N + 1, 1)
        vo = eta * (u - 1)

        delt = np.zeros(N, dtype=float)
        delt[0] = 1
        j = np.arange(1, N + 1, 1)
        SimpsonW = (3 + (-1) ** j - delt) / 3

        # Modifications to Ensure int_value integrability
        if self.S_0 >= 0.95 * self.K:  # ITM case
            alpha = 1.5
            v = vo - (alpha + 1) * 1j
            modcharFunc = np.exp(-self.r * self.T) * (OptionPricerFourier.BSM_characteristic_function(
                v, x0, self.T, self.r, self.sigma) / (alpha ** 2 + alpha - vo ** 2 + 1j * (2 * alpha + 1) * vo))
            FFTFunc = np.exp(1j * b * vo) * modcharFunc * eta * SimpsonW
            payoff = (fft(FFTFunc)).real
            CallValueM = np.exp(-alpha * k) / np.pi * payoff
        else:
            alpha = 1.1
            v = (vo - 1j * alpha) - 1j
            modcharFunc1 = np.exp(-self.r * self.T) * (1 / (1 + 1j * (vo - 1j * alpha)) -
                                                       np.exp(self.r * self.T) / (1j * (vo - 1j * alpha))
                                                       - OptionPricerFourier.BSM_characteristic_function(v, x0, self.T,
                                                                                                         self.r,
                                                                                                         self.sigma) /
                                                       ((vo - 1j * alpha) ** 2 - 1j * (vo - 1j * alpha)))
            v = (vo + 1j * alpha) - 1j
            modcharFunc2 = np.exp(-self.r * self.T) * (1 / (1 + 1j * (vo + 1j * alpha))
                                                       - np.exp(self.r * self.T) / (1j * (vo + 1j * alpha))
                                                       - OptionPricerFourier.BSM_characteristic_function(v, x0, self.T,
                                                                                                         self.r,
                                                                                                         self.sigma) / (
                                                               (vo + 1j * alpha) ** 2 - 1j * (vo + 1j * alpha)))
            FFTFunc = (np.exp(1j * b * vo)
                       * (modcharFunc1 - modcharFunc2)
                       * 0.5 * eta * SimpsonW)
            payoff = (fft(FFTFunc)).real
            CallValueM = payoff / (np.sinh(alpha * k) * np.pi)

        pos = int((k + b) / eps)
        CallValue = CallValueM[pos] * self.S_0

        return CallValue

    def get_put(self):
        call = self.get_call()
        put = call - self.S_0 + self.K * np.exp(-self.r * self.T)  # Computed using put-call parity

        return put


class OptionPricerMerton(OptionPricer):
    """
    Option pricing with the Merton jump model
    """

    def __init__(self, S_0: float, K: float, T: int, r: float, sigma: float, alpha: float, lamda: float, delta: float,
                 mu: float, Msigma: float):

        super().__init__(S_0=S_0, K=K, T=T, r=r, sigma=sigma)

        self.sigmaMerton = Msigma
        self.alpha = alpha
        self.lamda = lamda  # If lambda = 0, we have the traditinal BS case
        self.delta = delta
        self.y = self.alpha + (self.delta ** 2) / 2
        self.k = math.exp(self.y) - 1
        self.lambdaprime = self.lamda * (1 + self.k)

    def get_call(self):
        Fpoisson = 0
        i = 0
        s = 0
        while Fpoisson < 0.95:  # 0.95 is arbitrary, the closer to 1, the better, but it becomes computationally expensive
            # I calculate the probability that the number of jumps = i, then sigma_n and r_n, and put
            # them in the Black and Scholes formula and I get the sum of P(N=i)*(BSCall) from i=0 to
            # a large enough number
            fpoisson = (self.lambdaprime * self.T) ** i * (math.e ** (-self.lambdaprime * self.T)) / math.factorial(i)
            Fpoisson += fpoisson
            r_n = self.r - self.lamda * self.k + i * self.y / self.T
            sigma_n = math.sqrt(self.sigmaMerton ** 2 + i * (self.delta ** 2) / self.T)
            d1_n = (math.log(self.S_0 / self.K) + (r_n + (sigma_n ** 2) / 2) * self.T) / (sigma_n * math.sqrt(self.T))
            d2_n = d1_n - sigma_n * math.sqrt(self.T)
            Nd1_n = norm.cdf(d1_n, 0, 1)  # N(d1_n)
            Nd2_n = norm.cdf(d2_n, 0, 1)  # N(d2_n)
            cn = self.S_0 * Nd1_n - self.K * Nd2_n * math.e ** (-r_n * self.T)
            i += 1
            s += cn * fpoisson
        return s

    def get_put(self):
        Fpoisson = 0
        i = 0
        s = 0
        while Fpoisson < 0.95:  # 0.999 is arbitrary, the closer to 1, the better, but it becomes computationally expensive
            # I calculate the probability that the number of jumps = i, then sigma_n and r_n, and put
            # them in the Black and Scholes formula and I get the sum of P(N=i)*(BSCall) from 0 to
            # a large enough number
            fpoisson = (self.lambdaprime * self.T) ** i * (math.e ** (-self.lambdaprime * self.T)) / math.factorial(i)
            Fpoisson += fpoisson
            r_n = self.r - self.lamda * self.k + i * self.y / self.T
            sigma_n = math.sqrt(self.sigmaMerton ** 2 + i * (self.delta ** 2) / self.T)
            d1_n = (math.log(self.S_0 / self.K) + (r_n + (sigma_n ** 2) / 2) * self.T) / (sigma_n * math.sqrt(self.T))
            d2_n = d1_n - sigma_n * math.sqrt(self.T)
            Nmind1_n = norm.cdf(-d1_n, 0, 1)  # N(-d1_n)
            Nmind2_n = norm.cdf(-d2_n, 0, 1)  # N(-d2_n)
            p_n = self.K * Nmind2_n * math.e ** (-r_n * self.T) - self.S_0 * Nmind1_n
            i += 1
            s += p_n * fpoisson
        return s


class OptionPricerFourier2(OptionPricer):

    def __init__(self, S_0: float, K: float, T: int, r: float, sigma: float):
        super().__init__(S_0=S_0, K=K, T=T, r=r, sigma=sigma)

        self.mu = self.r - 0.5 * (self.sigma ** 2) * self.T
        self.vol = self.sigma * math.sqrt(self.T)

    def Brownian_Motion_cf(self, u):  # cf(Normal law)
        return np.exp(1j * u * (self.mu + math.log(self.S_0)) - 0.5 * self.vol ** 2 * u ** 2)

    # Here lies the semi-analytical method after founding the characteristic function of ln(ST)
    def delta_option(self, cf):  # Need to specify characteristic function
        f = lambda u: np.real((np.exp(-u * math.log(self.K) * 1j) * cf(u - 1j) / (u * 1j)) / cf(-1j))
        F = integrate.quad(f, 0, np.inf, limit=10000)
        return 1 / 2 + F[0] / math.pi

    def prob2(self, cf):  # Probability of finishing in-the-money #Need to specify the characteristic function
        f = lambda u: np.real(np.exp(-u * math.log(self.K) * 1j) / (u * 1j) * cf(u))
        F = integrate.quad(f, 0, np.inf, limit=10000)
        return 1 / 2 + F[0] / math.pi

    # Here lie the functions that produce an output depending on what type of Fourier-priced option is called
    def get_call(self, cf):
        return self.S_0 * self.delta_option(cf) - self.prob2(cf) * self.K * math.exp(-self.r * self.T)

    def get_Call(self):
        c = self.get_call_usingFourier(self.Brownian_Motion_cf)
        return c

    def get_call_usingFourier(self, cf):
        return self.S_0 * self.delta_option(cf) - self.prob2(cf) * self.K * math.exp(-self.r * self.T)

    def Put_parity_fourier(self,
                           cf):  # searching the price of the put given S_0, K and the call price computed using Fourier
        c = self.get_call_usingFourier(cf)
        p = c + self.K * np.exp(-self.r * self.T) - self.S_0
        return p

    def get_Call_BS_Fourier(self):
        c = self.get_call_usingFourier(self.Brownian_Motion_cf)
        return c

    def get_Put_BS_Fourier(self):
        put = self.Put_parity_fourier(self.Brownian_Motion_cf)
        return put


class OptionPricerHeston(OptionPricerFourier2):

    def __init__(self, S_0: float, K: float, T: int, r: float, sigma: float, rho: float, kappa: float,
                 eta: float, theta: float, spotvol:float):
        super().__init__(S_0=S_0, K=K, T=T, r=r, sigma=sigma)

        self.sigmainitial = sigma #is actually useless
        self.rho = rho
        self.kappa = kappa
        self.eta = eta
        self.theta = theta
        self.spotvol = spotvol

    def Modified_Heston_cf(self, u):  
        # as proposed by Shoutens in "A perfect Calibration! Now what?" (page 4) https://perswww.kuleuven.be/~u0009713/ScSiTi03.pdf
        d = np.sqrt((self.rho * self.theta * u * 1j - self.kappa) ** 2 - self.theta ** 2 * (-1j * u - u ** 2))
        b = self.kappa - self.rho * self.theta * u * 1j
        g = (b - d) / (b + d)
        e1 = np.exp(1j * u * (np.log(self.S_0) + self.r * self.T))
        e2 = np.exp(self.eta * self.kappa * self.theta ** (-2) * (
                (b - d) * self.T - 2 * np.log((1 - g * np.exp(-d * self.T)) / (1 - g))))
        e3 = np.exp(self.spotvol ** 2 * self.theta ** (-2) * (
                (b - d) * (1 - np.exp(-d * self.T)) / (1 - g * np.exp(-d * self.T))))
        return e1 * e2 * e3

    def get_call(self):
        c = self.get_call_usingFourier(self.Modified_Heston_cf)
        return c

    def get_put(self):
        put = self.Put_parity_fourier(self.Modified_Heston_cf)
        return put


if __name__ == '__main__':
    # black_scholes_pricer = OptionPricerlackScholes('AAPL', 'call', '2023-04-21', '2023-04-10', 100)
    # pricer_CRR = OptionPricerBlackScholes(S_0=175, K=175, T=1, r=0.014, sigma=0.01)
    # print(pricer_CRR.get_call())
    pricer_Fourier = OptionPricerFourier(S_0=175, K=175, T=1, r=0.014, sigma=0.01)
    pricer_FFT = OptionPricerFFT(S_0=175, K=175, T=1, r=0.014, sigma=0.01)
    print(pricer_Fourier.get_call())
    print(pricer_FFT.get_call())
    # call = black_scholes_pricer.get_call()
    # merton_log_call = black_scholes_pricer.get_MertonLognormalJumpCall()
    # put = black_scholes_pricer.get_put()
    # merton_log_put = black_scholes_pricer.get_MertonLognormalJumpPut()
    # print(call)
    # print(merton_log_call)  # should be greater or equal to the BS call
    # print(put)
    # print(merton_log_put)  # Should be greater or equal to the BS Put
    # Double check values with https://demonstrations.wolfram.com/OptionPricesInMertonsJumpDiffusionModel/
