import numpy as np

from .utils import c
from .logger import LoggerConfig

logger = LoggerConfig(__name__, level='DEBUG').get()

class Power:
    def __init__(self, cosmology):
        self.cosmology = cosmology
        self.pm = self.cosmology.matter_power_spectrum_interpolator()

    def k(self, l, z):
        return (l + 0.5) / self.cosmology.chi(z)

    def cls(self, l, bin1, bin2, tracer1, tracer2, z):
        nz = z.size
        # Matter power spectra using Limber's approximation
        k = self.k(l, z)
        pm = self.pm(z, k, grid=False)
        
        kfs1 = tracer1.compute_kernel_functions(bin1, z, self.cosmology)
        kfs2 = tracer2.compute_kernel_functions(bin2, z, self.cosmology)
        res = np.empty((tracer1.nkernels, tracer2.nkernels, nz))
        
        chi2hz = self.cosmology.chi2hz(z)
        for i in range(tracer1.nkernels):
            for j in range(tracer2.nkernels):
                res[i, j, :] = c * kfs1[i] * kfs2[j] * pm / chi2hz

        return np.trapz(res, z)

