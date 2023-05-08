from abc import ABC, abstractmethod

class OptionPricer(ABC): 

    @abstractmethod
    def price_call(self): 
        pass

    @abstractmethod 
    def price_put(self): 
        pass 