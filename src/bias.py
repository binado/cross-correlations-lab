from abc import ABC, abstractmethod

import numpy as np

from .utils import linear_growth_rate

class Bias(ABC):
    @abstractmethod
    def at_z(self, z):
        pass

class ConstantBias(Bias):
    def __init__(self, val=1):
        super().__init__()
        self._val = val

    def at_z(self, z):
        return self._val

class SqrtZBias(Bias):
    def at_z(self, z):
        return np.sqrt(1 + z)
    
class GrowthRateBias(Bias):
    def __init__(self, cosmology, b1=1, b2=1):
        self.b1, self.b2 = b1, b2
        self.cosmology = cosmology

    def at_z(self, z):
        return self.b1 + self.b2 / linear_growth_rate(z, self.cosmology)