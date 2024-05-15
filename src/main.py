import os
import json

import numpy as np

from .camb_interface import CAMBInterface
from .cosmology import Cosmology
from .density import *
from .kernel import *
from .tracer import *
from .parser import parser
from .power import Power
from .logger import LoggerConfig

if __name__ == '__main__':
    args = parser.parse_args()
    z = np.linspace(args.zmin, args.zmax, args.npoints)
    k = np.logspace(np.log(args.kmin), np.log(args.kmax), args.npoints)

    # Create output directory
    outdir = args.output_dir
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    logger = LoggerConfig(__name__, level='INFO', verbose=args.verbose).get()
    configfile = f'{outdir}config.json'
    with open(configfile, mode='w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    logger.info('Script config saved to %s', configfile)

    # Run camb to get matter power spectrum
    logger.info('Running Boltzmann solver')
    camb_interface = CAMBInterface(args.zmin, args.zmax)
    camb_interface.run_solver()
    cosmology = Cosmology(camb_interface)

    # Defining redshift bins
    logger.info('Defining redshift bins')
    nbins = args.nbins
    zbins = np.linspace(args.zmin, args.zmax, nbins + 1)
    bins = [RadialBin(zbins[i], zbins[i + 1]) for i in range(nbins)]
    zmids = [bin.center for bin in bins]

    # Defining tracers
    logger.info('Defining tracers')
    g_density = UniformInVolumeNumberDensity(1e-3, cosmology)
    gw_density = UniformInVolumeNumberDensity(3e-6, cosmology)

    g_kernels = [ClusteringKernel('galaxy clustering', 'g')]
    gw_kernels = [
        ClusteringKernel('gw clustering', 's'),
        WeakLensingKernel('gw lensing', 't')
        ]

    g_window_functions = [BoxWindowFunction()]
    gw_window_functions = [
        GWClusteringWindowFunction(cosmology, 0.05),
        GWLensingWindowFunction(cosmology, 0.05)
        ]

    g_bias = ConstantBias()
    gw_bias = ConstantBias()

    g_tracer = Tracer(g_density, g_kernels, g_window_functions, g_bias)
    gw_tracer = Tracer(gw_density, gw_kernels, gw_window_functions, gw_bias)

    # Power spectra pre-processing
    logger.info('Power spectra pre-processing')
    cp = Power(zbins, cosmology)

    l = args.l
    galbin = args.galbin
    cls = np.empty((nbins, g_tracer.nkernels, gw_tracer.nkernels))
    for i in range(nbins):
        logger.info('Computing cls for bins %s, %s', i, galbin)
        cls[i, :, :]= cp.cls(l, bins[i], bins[galbin], g_tracer, gw_tracer, z)

    if args.save_figure:
        import matplotlib.pyplot as plt

        logger.info('Plotting figures')
        fig, axs = plt.subplots(g_tracer.nkernels, gw_tracer.nkernels, squeeze=False)
        binvec = np.arange(nbins)
        labels = [
            [f'c{k2.symbol}{k1.symbol}' for k2 in gw_tracer.kernels]
            for k1 in g_tracer.kernels
        ]
        for i in range(g_tracer.nkernels):
            for j in range(gw_tracer.nkernels):
                axs[i][j].plot(zmids, cls[:, i, j])
                axs[i][j].set_xlabel('z')
                axs[i][j].set_title(labels[i][j])
                axs[i][j].grid()
        
        fig.tight_layout()
        fig.savefig(f"{outdir}/cls.{args.figure_format}", dpi=400)