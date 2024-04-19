import numpy as np
from astropy.cosmology import Planck18

import equations
from .utils import dc2_over_hz

class RedshiftBin:
    def __init__(self, zmin, zmax) -> None:
        self.zmin = zmin
        self.zmax = zmax

class AngularPowerSpectraModel:
    def __init__(self, 
                 bins, 
                 pm, # Matter power spectrum interpolator
                 ngw=3e-6, 
                 ng=1e-3, 
                 sigma_lnd=0.05, 
                 cosmology=Planck18) -> None:
        self.bins = bins
        self.cosmology = cosmology
        self.ngw = ngw
        self.ng = ng
        self.sigma_lnd = sigma_lnd
        self.pm = pm

    def k(self, l, z):
        return (l + 0.5) / self.chi(z)

    def chi(self, z):
        return self.cosmology.comoving_distance(z).value

    def ws(self, zbin: RedshiftBin, z, **kwargs):
        return equations.ws(z, self.cosmology, self.ngw, zbin.zmin, zbin.zmax, self.sigma_lnd, **kwargs)
    
    def wt(self, zbin: RedshiftBin, z, **kwargs):
        return equations.wt(z, self.cosmology, self.ngw, zbin.zmin, zbin.zmax, self.sigma_lnd, **kwargs)
    
    def wg(self, zbin: RedshiftBin, z, **kwargs):
        return equations.wg(z, self.cosmology, self.ng, zbin.zmin, zbin.zmax, **kwargs)
    
    def wkappa(self, z, zprime):
        return equations.wkappa(z, zprime, self.cosmology)

    def csisj(self, l, bini, binj, z, bias):
        arr = self.ws(bini, z) * self.ws(binj, z) / dc2_over_hz(z, self.cosmology)
        k = self.k(l, z)
        arr *= bias ** 2 * self.pm(k, z)
        return np.trapz(arr, z)
    
    def csitj(self, l, bini, binj, z, bgw):
        zz, zzprime = np.meshgrid(z, z)
        arr = self.ws(binj, zz) * self.wt(bini, zzprime) * self.wkappa(zzprime, zz)
        k = self.k(l, zzprime)
        arr *= bgw * self.pm(k, zzprime) / dc2_over_hz(zz, self.cosmology)
        return np.trapz(np.trapz(arr, z, axis=-1), z)
    
    def _ctitj_int(self, l, z, zz, zzprime):
        integrand = self.wkappa(z, zz) * self.wkappa(z, zzprime)
        k = self.k(l, z)
        integrand *= self.pm(k, z) / dc2_over_hz(z, self.cosmology)
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
        arr = self.wg(bini, z) * self.wg(binj, z) / dc2_over_hz(z, self.cosmology)
        k = self.k(l, z)
        arr *= self.pm(k, z) * bg ** 2
        return np.trapz(arr, z)

    def csigj(self, l, bini, binj, z, bgw, bg):
        arr = self.ws(bini, z) * self.wg(binj, z) / dc2_over_hz(z, self.cosmology)
        k = self.k(l, z)
        arr *= self.pm(k, z) * bgw * bg 
        return np.trapz(arr, z)
    
    def ctigj(self, l, bini, binj, z, bg):
        zz, zzprime = np.meshgrid(z, z)
        arr = self.wg(binj, zz) * self.wt(bini, zzprime) * self.wkappa(zzprime, zz)
        k = self.k(l, zzprime)
        arr *= bg * self.pm(k, zzprime) / dc2_over_hz(zz, self.cosmology)
        return np.trapz(np.trapz(arr, z, axis=-1), z)
    
    def cwigj(self, l, bini, binj, z, bgw, bg):
        csigj = self.csigj(l, bini, binj, z, bgw, bg)
        ctigj = self.ctigj(l, bini, binj, z, bg)
        cwigj = csigj + ctigj
        return csigj, ctigj, cwigj


