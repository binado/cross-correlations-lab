import numpy as np
from scipy.special import erfc

from .camb_interface import CAMBInterface
from .utils import lognormal_arg, critical_surface_density, normalize, G, c

def sfunc(z, zmin, zmax, cambi: CAMBInterface, sigma):
    """
    See Eq. 6.
    """
    dmin = cambi.dl(zmin)
    dmax = cambi.dl(zmax)
    d = cambi.dl(z)
    return 0.5 * (erfc(lognormal_arg(dmin, d, sigma)) - erfc(lognormal_arg(dmax, d, sigma)))

def tfunc(z, zmin, zmax, cambi: CAMBInterface, sigma):
    """
    See Eq. 9
    """
    dmin = cambi.dl(zmin)
    dmax = cambi.dl(zmax)
    d = cambi.dl(z)
    xmin = lognormal_arg(dmin, d, sigma)
    xmax = lognormal_arg(dmax, d, sigma)
    norm = sigma * np.sqrt(2 * np.pi)
    return (np.exp(-xmax ** 2) - np.exp(-xmin ** 2)) / norm

def ws(ngw, si, dc2_over_hz):
    """
    See Eq. 8
    Normalization is equivalent to dividing by \bar{n} in the paper
    """
    return c * ngw * dc2_over_hz * si

def wt(ngw, ti, dc2_over_hz):
    """
    See Eq. 8
    Normalization is equivalent to dividing by \bar{n} in the paper
    """
    return c * ngw * dc2_over_hz * ti

def wg(z, ng, dc2_over_hz, zmin, zmax):
    """
    See Eq. 15
    """
    cut = (z >= zmin) & (z <= zmax)
    return c * ng * dc2_over_hz * cut


def wkappa(z, zprime, cosmology):
    """
    See Eq. 4
    """
    domain = z <= zprime
    pcr = cosmology.critical_density(0).value
    omegam = cosmology.Om0
    num  = omegam * pcr * (1 + zprime) ** 3
    denom = cosmology.H(zprime).value * critical_surface_density(z, zprime, cosmology)
    return domain * num / denom

