import math
import camb

def run_solver(**kwargs):
    # See Table 2 of https://arxiv.org/pdf/1807.06209.pdf
    cosmo_params = {
        'H0': 67.36,
        'ombh2': 0.02237,
        'omch2': 0.12,
        'tau': 0.0544,
        'As': 1e-10 * math.exp(3.044),
        'ns': 0.9649,
        'mnu': 0.06,
        'omk': 0.0
    }
    camb_params = {
        'WantTransfer': True,
        'halofit_version': 'mead'
    }

    pars = camb.set_params(**cosmo_params, **camb_params, **kwargs)
    return camb.get_results(pars)

class CAMBInterface:
    def __init__(self) -> None:
        self.res = run_solver()

    def matter_power_spectrum_interpolator(self):
        """
        Get matter power spectrum interpolator from CAMB
        assuming a Planck 18 cosmology.
        """

        return self.res.get_matter_power_interpolator(k_hunit=False)

    def linear_growth_rate(self):
        planck18_sigma8 = 0.8111
        return self.res.get_fsigma8() / planck18_sigma8
