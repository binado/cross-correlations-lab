from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .utils import mesh, c

@dataclass
class Kernel(ABC):
    name : str = None
    symbol : str = None

    @classmethod
    @abstractmethod
    def at_z(cls, z, n, cosmology, bias=1):
        pass

class ClusteringKernel(Kernel):
    @classmethod
    def at_z(cls, z, n, cosmology, bias=1):
        return bias * n * cosmology.hz(z) / c


class WeakLensingKernel(Kernel):
    @classmethod
    def at_z(cls, z, n, cosmology, bias=1):
        zzp, zz = np.meshgrid(z, z, indexing='xy')
        dh = cosmology.hubble_distance
        chi = cosmology.chi(z)

        res = 1.5 * cosmology.Om0 * (1. + z) * chi / dh ** 2
        n_mesh = np.meshgrid(n, n, indexing='xy')[0]
        chi_mesh = np.meshgrid(chi, chi, indexing='xy')[0]

        reduced_kernel = n_mesh * cosmology.da12(zz, zzp) / chi_mesh
        reduced_kernel[zz >= zzp] = 0.
        integrated_reduced_kernel = np.trapz(reduced_kernel, z, axis=1)
        return res * integrated_reduced_kernel
