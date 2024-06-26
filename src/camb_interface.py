import math

import numpy as np
import camb

def camb_params(zmin, zmax, nz=256, kmax=1.2):
    # See Table 2 of https://arxiv.org/pdf/1807.06209.pdf
    cosmo_params = {
        'H0': 67.36,
        'ombh2': 0.02237,
        'omch2': 0.12,
        'tau': 0.0544,
        'mnu': 0.06,
        'omk': 0.0
    }
    init_params = {
        'As': 1e-10 * math.exp(3.044),
        'ns': 0.9649
    }
    cp = camb.model.CAMBparams()
    cp.set_cosmology(**cosmo_params)
    cp.WantTransfer = True
    cp.Transfer.PK_num_redshifts = nz
    cp.set_matter_power(redshifts=np.linspace(zmin, zmax, nz), kmax=kmax, nonlinear=True)
    cp.InitPower.set_params(**init_params)
    cp.NonLinear = camb.model.NonLinear_both
    return cp

def run_solver(params):
    return camb.get_results(params)

class CAMBInterface:
    def __init__(self, zmin, zmax, **kwargs) -> None:
        self.params = camb_params(zmin, zmax, **kwargs)
        self.res = None

    def run_solver(self):
        self.res = run_solver(self.params)
        return self.res
