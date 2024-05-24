import numpy as np

from .utils import normalize

class Tracer:
    """Class representing a tracer of matter.
    """
    def __init__(self, number_density, kernels, window_function, bias):
        """Create a new `Tracer` class instance.

        See Eqs.(5), (9) of https://arxiv.org/abs/1603.02356.

        Parameters
        ----------
        number_density : density.NumberDensity
            The number density of the tracer
        kernels : list[kernel.Kernel]
            The different kernels to apply for the tracer (e.g clustering, lensing)
        window_function : window.WindowFunction
            The different window functions to apply for the tracer. 
            The length of the list must equal that of `kernels`.
        bias : bias.Bias
            The bias model for the tracer
        """        
        self.density = number_density
        self.kernels = kernels
        self.window_function = window_function
        self.bias = bias

    @property
    def nkernels(self):
        return len(self.kernels)

    def normalized_density(self, zbin, z):
        """Compute normalized density for given redshift bin and window function.

        Parameters
        ----------
        zbin : density.RadialBin
            The redshift bin
        z : array_like
            Array of redshifts

        Returns
        -------
        array_like
            Normalized density over z
        """        
        density = self.density.at_z(z) * self.window_function.at_z(z, zbin.low, zbin.high)
        return normalize(density, z)

    def compute_kernel_functions(self, zbin, cosmology, z):
        """Compute the kernel functions for the tracer.

        Parameters
        ----------
        zbin : density.RadialBin
            The redshift bin
        cosmology : cosmology.Cosmology
            The assumed cosmology
        z : array_like
            Array of redshifts

        Returns
        -------
        array_like
            A (nkernels, nz) 2D array 
        """        
        nz = z.size
        res = np.empty((self.nkernels, nz))
        bias = self.bias.at_z(zbin.center)
        n = self.normalized_density(zbin, z)
        for i, kernel in enumerate(self.kernels):
            res[i, :] = kernel.at_z(z, n, cosmology, bias)

        return res
