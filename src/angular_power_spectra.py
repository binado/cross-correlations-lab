import numpy as np
from astropy.cosmology import Planck18

from . import equations
from .camb_interface import CAMBInterface
from .utils import mesh
from .logger import LoggerConfig

logger = LoggerConfig(__name__, level='DEBUG').get()

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

        self.si_mesh = None
        self.ti_mesh = None
        self.ws_mesh = None
        self.wt_mesh = None
        self.wg_mesh = None

    @property
    def zmid(self):
        return 0.5 * (self.zmin + self.zmax)

    def set_functions(self, z, cambi, ngw, ng, dc2_over_hz, sigma_lnd, **kwargs):
        self.si = equations.sfunc(z, self.zmin, self.zmax, cambi, sigma_lnd)
        self.ti = equations.tfunc(z, self.zmin, self.zmax, cambi, sigma_lnd)
        self.ws = equations.ws(ngw, self.si, dc2_over_hz, **kwargs)
        self.wt = equations.wt(ngw, self.ti, dc2_over_hz, **kwargs)
        self.wg = equations.wg(ng, dc2_over_hz, self.zmin, self.zmax, **kwargs)

        self.si_mesh = mesh(self.si)
        self.ti_mesh = mesh(self.ti)
        self.ws_mesh = mesh(self.ws)
        self.wt_mesh = mesh(self.wt)
        self.wg_mesh = mesh(self.wg)

class CrossPowerSpectra:
    labels = ['css', 'cst', 'cgg', 'csg', 'ctg']

    def __init__(self,
                 z,
                 bins: list[RedshiftBin],
                 cambi: CAMBInterface,
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
        self.cambi = cambi
        self.pm = self.cambi.matter_power_spectrum_interpolator()

        # Pre-compute quantities for faster evaluation
        logger.debug('Pre-computing quantities')
        self.zz, self.zzprime = np.meshgrid(z, z)
        self.domain = self.zzprime < self.zz
        self.dc2_over_hz = self.cambi.chi2_over_hz(self.z)
        self.dc2_over_hz_mesh = mesh(self.dc2_over_hz)
        self.wkappa = equations.wkappa(self.zzprime, self.zz, self.cosmology)

        logger.debug('Setting kernels for each redshift bin')
        for zbin in self.bins:
            zbin.set_functions(z, cambi, ngw, ng, self.dc2_over_hz, sigma_lnd, **kwargs)

    def k(self, l, z):
        """
        Get first argument of Pm(k, z) using Limber's approximation
        """
        return (l + 0.5) / self.cambi.chi(z)
    
    def compute(self, l, bini, binj, bg, bgw):
        # Matter power spectra using Limber's approximation
        k1dim = self.k(l, self.z)
        pm1dim = self.pm(self.z, k1dim, grid=False)
        pm2dim = mesh(pm1dim)

        # ss
        logger.debug('Computing css')
        arr = bini.ws * binj.ws / self.dc2_over_hz
        arr *= pm1dim * bgw ** 2
        css = np.trapz(arr, self.z)

        # st
        logger.debug('Computing cst')
        arr = binj.ws_mesh * bini.wt_mesh * self.wkappa / self.dc2_over_hz_mesh
        arr *= pm2dim * bgw
        cst = np.trapz(np.trapz(arr * self.domain, self.z, axis=0), self.z)

        # gg
        logger.debug('Computing cgg')
        arr = bini.wg * binj.wg / self.dc2_over_hz
        arr *= pm1dim * bg ** 2
        cgg = np.trapz(arr, self.z)

        # sg
        logger.debug('Computing csg')
        arr = bini.ws * binj.wg / self.dc2_over_hz
        arr *= pm1dim * bg * bgw
        csg = np.trapz(arr, self.z)
        
        # tg
        logger.debug('Computing ctg')
        arr = binj.wg_mesh * bini.wt_mesh * self.wkappa / self.dc2_over_hz_mesh
        arr *= pm2dim * bg
        ctg = np.trapz(np.trapz(arr * self.domain, self.z, axis=0), self.z)

        cls = np.array([css, cst, cgg, csg, ctg])
        return self.labels, cls
