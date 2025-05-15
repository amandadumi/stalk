#!/usr/bin/env python3

from os import makedirs
from numpy import array

from stalk import ParameterStructure
from stalk.io import XyzGeometry

from params import forward, backward, relax_pyscf

# Generate base directory for point A
basedir = 'pointA'
makedirs(basedir, exist_ok=True)

# Let us initiate a ParameterStructure object that implements the parametric mappings
params_init = array([1.04, 1.2])
elem = ['N'] + 3 * ['H']
structure_init = ParameterStructure(
    forward=forward,
    backward=backward,
    params=params_init,
    elem=elem,
    units='A',
)

outfile = f'{basedir}/relax.xyz'
try:
    geom = XyzGeometry({'suffix': outfile}).load('./')
except FileNotFoundError:
    new_params = relax_pyscf(structure_init, outfile)
    geom = XyzGeometry({'suffix': outfile}).load('./')
# end try
new_params = structure_init.map_forward(geom.get_pos())
structure_relax = structure_init.copy(params=new_params)
