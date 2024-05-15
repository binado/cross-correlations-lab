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
        return bias * n *  cosmology.hz(z) / c


class WeakLensingKernel(Kernel):
    @classmethod
    def at_z(cls, z, n, cosmology, bias=1):
        zz, zzp = np.meshgrid(z, z, indexing='ij')
        dh = cosmology.hubble_distance
        chi = cosmology.chi(z)

        res = 1.5 * bias * cosmology.Om0 * (1. + z) * chi / dh ** 2
        n_mesh = mesh(n, indexing='ij')
        chi_mesh = mesh(chi, indexing='ij')

        reduced_kernel = n_mesh * cosmology.da12(zzp, zz) / chi_mesh
        integrated_reduced_kernel = np.trapz(reduced_kernel, z, axis=0)
        return res * integrated_reduced_kernel
