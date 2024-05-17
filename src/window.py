from abc import ABC, abstractmethod

import numpy as np
from scipy.special import erfc

from .utils import lognormal_arg

class WindowFunction(ABC):
    """The window function of a bin to be convolved with the number density."""
    @abstractmethod
    def at_z(self, z, low, high):
        """Window function at a redshift array z.

        Parameters
        ----------
        z : array_like
            Redshift array
        low : float
            The minimum redshift value of the bin
        high : float
            The maximum redshift value of the bin
        """

class BoxWindowFunction(WindowFunction):
    def at_z(self, z, low, high):
        cut = (z >= low) & (z <= high)
        return np.ones_like(z) * cut

class GWClusteringWindowFunction(WindowFunction):
    def __init__(self, cosmology, sigma):
        """The window function for GW clustering as defined in
        Eq. (5) of https://arxiv.org/abs/1603.02356.

        Parameters
        ----------
        cosmology : cosmology.Cosmology
            The assumed cosmology
        sigma : float
            The \sigma_{ln D} parameter from Eq. (1) of the paper
        """        
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
    def __init__(self, cosmology, sigma):
        """The window function for GW lensing as defined in
        Eq. (9) of https://arxiv.org/abs/1603.02356.

        Parameters
        ----------
        cosmology : cosmology.Cosmology
            The assumed cosmology
        sigma : float
            The \sigma_{ln D} parameter from Eq. (1) of the paper
        """    
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
