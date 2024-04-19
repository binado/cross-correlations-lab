import numpy as np
from scipy.special import erfc

from .utils import lognormal_arg, dc2_over_hz, critical_surface_density, normalize

def si(z, zmin, zmax, cosmology, sigma):
    """
    See Eq. 6.
    """
    dmin = cosmology.luminosity_distance(zmin).value
    dmax = cosmology.luminosity_distance(zmax).value
    d = cosmology.luminosity_distance(z).value
    return 0.5 * (erfc(lognormal_arg(dmin, d, sigma)) - erfc(lognormal_arg(dmax, d, sigma)))

def ti(z, zmin, zmax, cosmology, sigma):
    """
    See Eq. 9
    """
    dmin = cosmology.luminosity_distance(zmin).value
    dmax = cosmology.luminosity_distance(zmax).value
    d = cosmology.luminosity_distance(z).value
    xmin = lognormal_arg(dmin, d, sigma)
    xmax = lognormal_arg(dmax, d, sigma)
    norm = sigma * np.sqrt(2 * np.pi)
    return (np.exp(-xmax ** 2) - np.exp(-xmin ** 2)) / norm

def ws(z, cosmology, ngw, zmin, zmax, sigma, norm=False):
    """
    See Eq. 8
    Normalization is equivalent to dividing by \bar{n} in the paper
    """
    res = ngw * dc2_over_hz(z, cosmology) * si(z, zmin, zmax, cosmology, sigma)
    return normalize(res, z) if norm else res

def wt(z, cosmology, ngw, zmin, zmax, sigma, norm=False):
    """
    See Eq. 8
    Normalization is equivalent to dividing by \bar{n} in the paper
    """
    res = ngw * dc2_over_hz(z, cosmology) * ti(z, zmin, zmax, cosmology, sigma)
    return normalize(res, z) if norm else res

def wg(z, cosmology, ng, zmin, zmax, norm=False):
    """
    See Eq. 15
    """
    cut = (z >= zmin) & (z <= zmax)
    res = ng * dc2_over_hz(z, cosmology) * cut
    return normalize(res, z) if norm else res


def wkappa(z, zprime, cosmology):
    """
    See Eq. 4
    """
    if z > zprime:
        return 0.
    pcr = cosmology.critical_density(0)
    omegam = cosmology.Om0
    num  = omegam * pcr * (1 + zprime) ** 3
    denom = cosmology.H(zprime) * critical_surface_density(z, zprime, cosmology)
    return num / denom

