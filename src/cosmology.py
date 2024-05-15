import numpy as np

from .camb_interface import CAMBInterface
from .utils import c

class Cosmology:
    def __init__(self, camb_interface: CAMBInterface):
        self.cambi = camb_interface
        self.params = self.cambi.params

        res = self.cambi.res
        self.cosmo = res if res is not None else camb_interface.run_solver()

    @property
    def H0(self):
        return self.params.H0
    
    @property
    def h(self):
        return self.params.h
    
    @property
    def Om0(self):
        return self.params.omegam
    
    @property
    def hubble_distance(self):
        return c / self.H0

    def hz(self, z: np.ndarray):
        return self.cosmo.hubble_parameter(z)
    
    def chi(self, z: np.ndarray):
        return self.cosmo.comoving_radial_distance(z)
    
    def chi2_over_hz(self, z: np.ndarray):
        return self.chi(z) ** 2 / self.hz(z)
    
    def chi2hz(self, z: np.ndarray):
        return self.chi(z) ** 2 * self.hz(z)
    
    def dl(self, z):
        return self.cosmo.luminosity_distance(z)
    
    def da12(self, z1, z2):
        return self.cosmo.angular_diameter_distance2(z1, z2)

    def matter_power_spectrum_interpolator(self, **kwargs):
        """
        Get matter power spectrum interpolator from CAMB
        assuming a Planck 18 cosmology.
        """

        return self.cosmo.get_matter_power_interpolator(k_hunit=False, **kwargs)