import numpy as np
from astropy import constants as const

c = const.c.to('km/s').value

def mesh(x, **kwargs):
    xx, _ = np.meshgrid(x, x, **kwargs)
    return xx

def normalize(y, x, eps=1e-10):
    norm = np.trapz(y, x)
    if norm < eps:
        norm += eps
    return y / norm

def lognormal_arg(x1, x2, sigma):
    norm = sigma * np.sqrt(2)
    return (np.log(x1) - np.log(x2)) / norm

def linear_growth_rate(z, cosmology):
    """
    Compute linear growth rate $f(z)$ in the approximation that $f(z) \simeq \Omega_m^\gamma$,

    where $\gamma \simeq 0.55$.
    """
    return cosmology.Om(z) ** 0.55
