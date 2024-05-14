import os
import json

import numpy as np

from .angular_power_spectra import CrossPowerSpectra, RedshiftBin
from .camb_interface import CAMBInterface
from .parser import parser
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

    # Defining redshift bins
    logger.info('Defining redshift bins')
    nbins = args.nbins
    zbins = np.linspace(args.zmin, args.zmax, nbins + 1)
    bins = [RedshiftBin(zbins[i], zbins[i + 1]) for i in range(nbins)]
    zmids = [bin.zmid for bin in bins]

    # Power spectra pre-processing
    logger.info('Power spectra pre-processing')
    cp = CrossPowerSpectra(z, bins, camb_interface)

    l = args.l
    k = cp.k(l, cp.zz)
    logger.info('%s', k)
    galbin = args.galbin
    #print(cp.pm(z, 1, grid=False))
    cls = np.empty((nbins, len(cp.labels)))
    for i in range(nbins):
        logger.info('Computing cls for bins %s, %s', i, galbin)
        _,  cls[i, :]= cp.compute(l, bins[i], bins[galbin], 1, 1)

    if args.save_figure:
        import matplotlib.pyplot as plt

        logger.info('Plotting figures')
        nfig = len(cp.labels)
        fig, axs = plt.subplots(nfig, 1, sharex=True, figsize=(4, 3 * nfig))
        binvec = np.arange(nbins)
        for i, (label, ax) in enumerate(zip(cp.labels, axs)):
            ax.plot(zmids, cls[:, i])
            ax.set_ylabel(label)
            ax.set_xlabel('z')
            ax.grid()
        
        fig.tight_layout()
        fig.savefig(f"{outdir}/cls.{args.figure_format}", dpi=400)