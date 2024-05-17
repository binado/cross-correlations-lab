from .utils import c

class Cosmology:
    """Cosmology class with wrapper methods to the CAMB api.
    """
    def __init__(self, camb_interface):
        """Create a new `Cosmology` class instance.

        Parameters
        ----------
        camb_interface : camb_interface.CAMBInterface
            CAMB Interface instance
        """        
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

    def hz(self, z):
        return self.cosmo.hubble_parameter(z)

    def efunc(self, z):
        return self.hz(z) / self.H0

    def Om(self, z):
        return self.Om0 * (1. + z) ** 3 / self.efunc(z)

    def chi(self, z):
        return self.cosmo.comoving_radial_distance(z)

    def chi2_over_hz(self, z):
        return self.chi(z) ** 2 / self.hz(z)

    def chi2hz(self, z):
        return self.chi(z) ** 2 * self.hz(z)

    def dl(self, z):
        return self.cosmo.luminosity_distance(z)
    
    def da12(self, z1, z2):
        return self.cosmo.angular_diameter_distance2(z1, z2)

    def matter_power_spectrum_interpolator(self, **kwargs):
        """
        Get matter power spectrum interpolator from CAMB.
        """
        return self.cosmo.get_matter_power_interpolator(k_hunit=False, **kwargs)