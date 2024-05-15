import numpy as np
from astropy import constants as const

G = const.G.value
c = const.c.value

def mesh(x: np.ndarray, **kwargs):
    xx, _ = np.meshgrid(x, x, **kwargs)
    return xx

def normalize(y, x, eps=1e-10):
    norm = np.trapz(y, x)
    if norm < eps:
        norm += eps
    return y / norm

def dc2hz(z, cosmology):
    """
    Return \chi^2(z){H(z) in Mpc km / s
    """
    dc = cosmology.comoving_distance(z).value
    hz = cosmology.H(z).value
    return dc ** 2 * hz


def dc2_over_hz(z, cosmology):
    """
    Return \frac{\chi^2(z)}{H(z)} in Mpc^3 s / km
    """
    dc = cosmology.comoving_distance(z).value
    hz = cosmology.H(z).value
    return dc ** 2 / hz

def angular_diameter_distance12(z1, z2, cosmology):
    """
    See https://arxiv.org/pdf/astro-ph/9905116.pdf
    """
    # dm1 = cosmology.comoving_distance(z1).value
    # dm2 = cosmology.comoving_distance(z2).value
    # return (dm2 - dm1) / (1. + z2)
    return cosmology.angular_diameter_distance_z1z2(z1, z2).value


def critical_surface_density(zlens, zsource, cosmology):
    ds = cosmology.angular_diameter_distance(zsource).value
    dd = cosmology.angular_diameter_distance(zlens).value
    dds = angular_diameter_distance12(zlens, zsource, cosmology)
    prefactor = 0.25 * c**2 / G / np.pi
    return prefactor * ds / dd / dds

def lognormal_arg(x1, x2, sigma):
    norm = sigma * np.sqrt(2)
    return (np.log(x1) - np.log(x2)) / norm

def linear_growth_rate(z, cosmology):
    """
    Compute linear growth rate $f(z)$ in the approximation that $f(z) \simeq \Omega_m^\gamma$,

    where $\gamma \simeq 0.55$.
    """
    return (cosmology.Om0 * (1. + z) ** 3) ** 0.55
