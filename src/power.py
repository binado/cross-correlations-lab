import numpy as np

from .utils import c
from .logger import LoggerConfig

logger = LoggerConfig(__name__, level='DEBUG').get()

class Power:
    """Class to compute the angular power spectra between tracers.
    """
    def __init__(self, cosmology):
        self.cosmology = cosmology
        self.pm = self.cosmology.matter_power_spectrum_interpolator()

    def k(self, l, z):
        """Compute wavenumber using Limber's approximation.

        Parameters
        ----------
        l : int
            The angular power spectrum multipole.
        z : array_like
            Array of redshifts

        Returns
        -------
        array_like
            Array of wavenumbers
        """   
        return (l + 0.5) / self.cosmology.chi(z)

    def cls(self, l, tracer1, tracer2, bin1, bin2, z):
        """Compute angular power spectra between two tracers.

        Parameters
        ----------
        l : int
            The angular power spectrum multipole.
        tracer1 : tracer.Tracer
            The first tracer
        tracer2 : tracer.Tracer
            The second tracer
        bin1 : density.RadialBin
            The redshift bin of the first tracer
        bin2 : density.RadialBin
            The redshift bin of the second tracer
        z : array_like
            Array of redshifts

        Returns
        -------
        array_like
            A (tracer1.nkernels, tracer2.nkernels) 2D array with the $C_\ell$ between each kernel.
        """     
        nz = z.size
        # Matter power spectra using Limber's approximation
        k = self.k(l, z)
        pm = self.pm(z, k, grid=False)
        
        kfs1 = tracer1.compute_kernel_functions(bin1, self.cosmology, z)
        kfs2 = tracer2.compute_kernel_functions(bin2, self.cosmology, z)
        res = np.empty((tracer1.nkernels, tracer2.nkernels, nz))
        
        chi2hz = self.cosmology.chi2hz(z)
        for i in range(tracer1.nkernels):
            for j in range(tracer2.nkernels):
                res[i, j, :] = c * kfs1[i] * kfs2[j] * pm / chi2hz

        return np.trapz(res, z)

