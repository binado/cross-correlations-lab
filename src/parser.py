import argparse

# Adding CLI arguments
parser = argparse.ArgumentParser(
    prog="cross_power_spectrum", 
    description="Compute cross-power spectra between GWs and galaxies"
)

parser.add_argument("-n", "--nbins", type=int, default=6, help="Number of bins")
parser.add_argument("--zmin", type=float, default=1e-3, help="Minimum redshift")
parser.add_argument("--zmax", type=float, default=1.2, help="Maximum redshift")
parser.add_argument("--kmin", type=float, default=1e-4, help="Minimum k")
parser.add_argument("--kmax", type=float, default=1, help="Maximum k")
parser.add_argument("--galbin", type=int, default=3, help="Galaxy bin")
parser.add_argument("--npoints", type=int, default=1000, help="Number of points in z array")
parser.add_argument("-o", "--output-dir", default="out/", help="Output file directory")
parser.add_argument("-l", type=int, default=100, help="l multipole")
parser.add_argument("-s", "--save-figure", action="store_true", help="Save output to figure")
parser.add_argument("--figure-format", type=str, default="png", help="Figure format")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-d", "--debug", action="store_true", help="Display logs for debugging")
        