#!/usr/bin env python3

from pathlib import Path
from numpy import loadtxt, savetxt, nan

from stalk.io.XyzGeometry import XyzGeometry
from stalk.params.ParameterSet import ParameterSet
from stalk.params.PesResult import PesResult
from stalk.util.util import directorize


def write_xyz_sigma(
    structure: ParameterSet,
    suffix='structure.xyz',
    sigma=None,
    sigma_suffix='sigma.dat',
    **kwargs
):
    g = XyzGeometry()
    if sigma is not None:
        sigmafilename = directorize(structure.file_path) + sigma_suffix
        savetxt(sigmafilename, [sigma])
    # end if
    g.write(structure=structure, path=structure.file_path, suffix=suffix, **kwargs)
# end def


def load_energy(structure: ParameterSet, suffix='energy.dat', **kwargs):
    filename = directorize(structure.file_path) + suffix
    if Path(filename).exists():
        data = loadtxt(filename)
        print(f"Loaded {filename}")
        result = PesResult(data[0], data[1])
    else:
        print(f"Waiting for {filename}")
        # Return null result
        result = PesResult(nan)
    # end if
    return result
# end def
