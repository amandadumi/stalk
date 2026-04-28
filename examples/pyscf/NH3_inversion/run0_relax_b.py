#!/usr/bin/env python3

from os import makedirs
from numpy import array, pi

from stalk import ParameterStructure
from stalk import XyzGeometry

from params import forward, backward, relax_pyscf

# Generate base directory for point B
basedir = 'pointB'
makedirs(basedir, exist_ok=True)

# Let us initiate a ParameterStructure object that implements the parametric mappings
params_init = array([1.04, pi - 1.2])
elem = ['N'] + 3 * ['H']
structure_init = ParameterStructure(
    forward=forward,
    backward=backward,
    params=params_init,
    elem=elem,
    units='A',
)

outfile = f'{basedir}/relax.xyz'
xyz = XyzGeometry(suffix=outfile)
structure_relax = xyz.load_or_relax(
    path='./',
    relax_func=relax_pyscf,
    structure=structure_init,
    xc='pbe',
    outfile=outfile,
)
