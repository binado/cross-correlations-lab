import numpy as np

from .utils import normalize

class Tracer:
    """Class representing a tracer of matter.
    """
    def __init__(self, number_density, kernels, window_functions, bias):
        """Create a new `Tracer` class instance.

        We allow for a different window function for each kernel for flexibility.
        See Eqs.(5), (9) of https://arxiv.org/abs/1603.02356.

        Parameters
        ----------
        number_density : density.NumberDensity
            The number density of the tracer
        kernels : list[kernel.Kernel]
            The different kernels to apply for the tracer (e.g clustering, lensing)
        window_functions : list[window.WindowFunction]
            The different window functions to apply for the tracer. 
            The length of the list must equal that of `kernels`.
        bias : bias.Bias
            The bias model for the tracer
        """        
        self.density = number_density

        if len(kernels) != len(window_functions):
            raise ValueError

        self.kernels = kernels
        self.window_functions = window_functions
        self.bias = bias

    @property
    def nkernels(self):
        return len(self.kernels)

    def normalized_density(self, zbin, window_function, z):
        """Compute normalized density for given redshift bin and window function.

        Parameters
        ----------
        zbin : density.RadialBin
            The redshift bin
        window_function : window.WindowFunction
            The window function to consider
        z : array_like
            Array of redshifts

        Returns
        -------
        array_like
            Normalized density over z
        """        
        density = self.density.at_z(z) * window_function.at_z(z, zbin.low, zbin.high)
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
        for i, (kernel, window_function) in enumerate(zip(self.kernels, self.window_functions)):
            n = self.normalized_density(zbin, window_function, z)
            res[i, :] = kernel.at_z(z, n, cosmology, bias)

        return res
