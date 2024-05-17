import numpy as np

from .utils import normalize

class Tracer:
    def __init__(self, number_density, kernels, window_functions, bias):
        self.density = number_density
        self.kernels = kernels
        kernel_names = [kernel.name for kernel in kernels]
        self.wf_dict = dict(zip(kernel_names, window_functions))
        self.bias = bias

    @property
    def nkernels(self):
        return len(self.kernels)

    def normalized_density(self, zbin, z, kernel_name):
        wf = self.wf_dict.get(kernel_name)
        if wf is None:
            raise ValueError

        density = self.density.at_z(z) * wf.at_z(z, zbin.low, zbin.high)
        return normalize(density, z)

    def compute_kernel_functions(self, zbin, z, cosmology):
        nz = z.size
        res = np.empty((self.nkernels, nz))
        bias = self.bias.at_z(zbin.center)
        for i, kernel in enumerate(self.kernels):
            n = self.normalized_density(zbin, z, kernel.name)
            res[i, :] = kernel.at_z(z, n, cosmology, bias)

        return res
