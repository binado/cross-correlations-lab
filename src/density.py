from abc import ABC, abstractmethod

import numpy as np
from scipy.special import erfc

from .cosmology import Cosmology
from .utils import lognormal_arg, c

class NumberDensity(ABC):
    def __init__(self, number_density, h_unit=False) -> None:
        super().__init__()
        self._val = number_density
        self.h_unit = h_unit

    @abstractmethod
    def at_z(self, z):
        pass

class UniformInRedshiftNumberDensity(NumberDensity):
    def at_z(self, z):
        return self._val

    
class UniformInVolumeNumberDensity(NumberDensity):
    def __init__(self, number_density, cosmology: Cosmology, **kwargs) -> None:
        super().__init__(number_density, **kwargs)
        self.cosmo = cosmology

    def at_z(self, z):
        return 4. * np.pi * c * self._val * self.cosmo.chi2_over_hz(z)

    
class ArrayNumberDensity(NumberDensity):
    def at_z(self, z):
        return self._val


class RadialBin:
    def __init__(self, low, high) -> None:
        self.low = low
        self.high = high
    
    @property
    def center(self):
        return 0.5 * (self.low + self.high)
    
class WindowFunction(ABC):
    @abstractmethod
    def at_z(self, z, low, high):
        pass

class BoxWindowFunction(WindowFunction):
    def at_z(self, z, low, high):
        cut = (z >= low) & (z <= high)
        return np.ones_like(z) * cut

class GWClusteringWindowFunction(WindowFunction):
    def __init__(self, cosmology, sigma) -> None:
        self.cosmology = cosmology
        self.sigma = sigma

    def at_z(self, z, low, high):
        dlow = self.cosmology.dl(low)
        dhigh = self.cosmology.dl(high)
        d = self.cosmology.dl(z)
        xlow = lognormal_arg(dlow, d, self.sigma)
        xhigh = lognormal_arg(dhigh, d, self.sigma)
        return 0.5 * (erfc(xlow) - erfc(xhigh))
    
class GWLensingWindowFunction(WindowFunction):
    def __init__(self, cosmology, sigma) -> None:
        self.cosmology = cosmology
        self.sigma = sigma

    def at_z(self, z, low, high):
        dlow = self.cosmology.dl(low)
        dhigh = self.cosmology.dl(high)
        d = self.cosmology.dl(z)
        xlow_squared = lognormal_arg(dlow, d, self.sigma) ** 2
        xhigh_squared = lognormal_arg(dhigh, d, self.sigma) ** 2
        norm = self.sigma * np.sqrt(2 * np.pi)
        return (np.exp(-xhigh_squared) - np.exp(-xlow_squared)) / norm
