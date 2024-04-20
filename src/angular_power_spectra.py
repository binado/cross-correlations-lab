import numpy as np
from astropy.cosmology import Planck18

from . import equations, utils

class RedshiftBin:
    def __init__(self, zmin, zmax) -> None:
        self.zmin = zmin
        self.zmax = zmax

        # Pre-computed quantities for each bin for fast evaluation
        self.computed = False
        self.si = None
        self.ti = None
        self.ws = None
        self.wt = None
        self.wg = None

    def set_functions(self, z, cosmology, ngw, ng, dc2_over_hz, sigma_lnd, **kwargs):
        self.si = equations.sfunc(z, self.zmin, self.zmax, cosmology, sigma_lnd)
        self.ti = equations.tfunc(z, self.zmin, self.zmax, cosmology, sigma_lnd)
        self.ws = equations.ws(z, ngw, self.si, dc2_over_hz, **kwargs)
        self.wt = equations.wt(z, ngw, self.ti, dc2_over_hz, **kwargs)
        self.wg = equations.wg(z, ng, dc2_over_hz, self.zmin, self.zmax, **kwargs)
        self.computed = True

class CrossPowerSpectra:
    def __init__(self,
                 z,
                 bins: list[RedshiftBin],
                 pm, # Matter power spectrum interpolator
                 ngw=3e-6,
                 ng=1e-3,
                 sigma_lnd=0.05,
                 cosmology=Planck18,
                 **kwargs) -> None:
        self.z = z
        self.bins = bins
        self.cosmology = cosmology
        self.ngw = ngw
        self.ng = ng
        self.sigma_lnd = sigma_lnd
        self.pm = pm

        # Pre-compute quantities for faster evaluation
        self.dc2_over_hz = utils.dc2_over_hz(z, cosmology)
        for zbin in self.bins:
            zbin.set_functions(z, cosmology, ngw, ng, self.dc2_over_hz, sigma_lnd, **kwargs)


    def k(self, l, z):
        """
        Get first argument of Pm(k, z) using Limber's approximation
        """
        return (l + 0.5) / self.chi(z)

    def chi(self, z):
        return self.cosmology.comoving_distance(z).value
    
    def wkappa(self, z, zprime):
        return equations.wkappa(z, zprime, self.cosmology)

    def csisj(self, l, bini, binj, z, bias):
        arr = bini.ws * binj.ws / self.dc2_over_hz
        k = self.k(l, z)
        arr *= bias ** 2 * self.pm(k, z)
        return np.trapz(arr, z)
    
    def csitj(self, l, bini, binj, z, bgw):
        zz, zzprime = np.meshgrid(z, z)
        arr = self.ws(binj, zz) * self.wt(bini, zzprime) * self.wkappa(zzprime, zz)
        k = self.k(l, zzprime)
        arr *= bgw * self.pm(k, zzprime) / self.dc2_over_hz(zz, self.cosmology)
        return np.trapz(np.trapz(arr, z, axis=-1), z)
    
    def _ctitj_int(self, l, z, zz, zzprime):
        integrand = self.wkappa(z, zz) * self.wkappa(z, zzprime)
        k = self.k(l, z)
        integrand *= self.pm(k, z) / self.dc2_over_hz
        return np.trapz(integrand, z)
    
    def ctitj(self, l, bini, binj, z):
        zz, zzprime = np.meshgrid(z, z)
        arr = self.wt(bini, zz) * self.wt(binj, zzprime) * self._ctitj_int(l, z, zz, zzprime)
        return np.trapz(np.trapz(arr, z, axis=-1), z)
    
    def cwiwj(self, l, bini, binj, z, bgw):
        csisj = self.csisj(l, bini, binj, z, bgw)
        csitj = self.csitj(l, bini, binj, z, bgw)
        csjti = self.csitj(l, binj, bini, z, bgw)
        ctitj = self.ctitj(l, bini, binj, z)
        cwiwj = csisj + csitj + csjti + ctitj
        return csisj, csitj, csjti, ctitj, cwiwj
    
    def cgigj(self, l, bini, binj, z, bg):
        arr = bini.wg * binj.wg / self.dc2_over_hz
        k = self.k(l, z)
        arr *= self.pm(k, z) * bg ** 2
        return np.trapz(arr, z)

    def csigj(self, l, bini, binj, z, bgw, bg):
        arr = bini.ws * binj.wg / self.dc2_over_hz
        k = self.k(l, z)
        arr *= self.pm(k, z) * bgw * bg 
        return np.trapz(arr, z)
    
    def ctigj(self, l, bini, binj, z, bg):
        zz, zzprime = np.meshgrid(z, z)
        arr = self.wg(binj, zz) * self.wt(bini, zzprime) * self.wkappa(zzprime, zz)
        k = self.k(l, zzprime)
        arr *= bg * self.pm(k, zzprime) / self.dc2_over_hz
        return np.trapz(np.trapz(arr, z, axis=-1), z)
    
    def cwigj(self, l, bini, binj, z, bgw, bg):
        csigj = self.csigj(l, bini, binj, z, bgw, bg)
        ctigj = self.ctigj(l, bini, binj, z, bg)
        cwigj = csigj + ctigj
        return csigj, ctigj, cwigj


