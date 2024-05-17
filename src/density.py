from abc import ABC, abstractmethod

import numpy as np

from .utils import c

class NumberDensity(ABC):
    """Base class for representing a number density in redshift.
    """
    def __init__(self, number_density, h_unit=False):
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
    def __init__(self, number_density, cosmology, **kwargs):
        super().__init__(number_density, **kwargs)
        self.cosmo = cosmology

    def at_z(self, z):
        # Multiply by dVc / dz
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
