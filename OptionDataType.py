import math


class OptionData():
    #set_strike_price = set()
    #set_time_to_maturity = set()

    def __init__(self,  option_type=None,
                 option_price=None,
                 underlying_price=None,
                 strike_price=None,
                 time_to_maturity=None,
                 risk_free_rate=None,
                 implied_volatility=None,
                 evaluation_date=None,
                 expiration_date = None,
                 underlying = None,
                 estimated_implied_vol = None,
                 historical_volatility = None,
                 historical_std = None,
                 BS_pricing = None,
                 CRR_pricing = None,
                 BSM_pricing = None,
                 Fourier_pricing = None,
                 FFT_pricing = None
                 ):
        self.option_price = option_price
        self.underlying_price = underlying_price
        self.strike_price = strike_price
        self.time_to_maturity = time_to_maturity
        self.option_type = option_type
        self.risk_free_rate = risk_free_rate
        self.implied_volatility = implied_volatility
        self.evaluation_date = evaluation_date
        self.expiration_date = expiration_date
        self.underlying = underlying
        self.estimated_implied_vol = estimated_implied_vol
        self.historical_volatility = historical_volatility
        self.historical_std = historical_std
        self.BS_pricing = BS_pricing
        self.CRR_pricing = CRR_pricing
        self.BSM_pricing = BSM_pricing
        self.Fourier_pricing = Fourier_pricing
        self.FFT_pricing = FFT_pricing
        # OptionData.set_strike_price.add(strike_price)
        # OptionData.set_time_to_maturity.add(time_to_maturity)

    def __str__(self):
        string_representation = f"\t Option type: {self.option_type}\
            \n \t Underlying : {self.underlying}\
            \n \t Evaluation date: {self.evaluation_date}\
            \n \t Expiration date: {self.expiration_date}\
            \n \t Option price: {self.option_price} \
            \n \t BS pricing: {self.BS_pricing}\
            \n \t CRR pricing: {self.CRR_pricing}\
            \n \t BSM pricing: {self.BSM_pricing}\
            \n \t Fourier pricing: {self.Fourier_pricing}\
            \n \t FFT pricing: {self.FFT_pricing}\
            \n \t Underlying price: {self.underlying_price} \
            \n \t Strike price: {self.strike_price} \
            \n \t Time to maturity: {self.time_to_maturity}\
            \n \t Risk free rate: {self.risk_free_rate}\
            \n \t Implied volatility: {self.implied_volatility} \
            \n \t Estimated implied vol: {self.estimated_implied_vol} \
            \n \t Historical volatility: {self.historical_volatility} \
            \n \t Historical standard deviation: {self.historical_std}\n"
        return string_representation