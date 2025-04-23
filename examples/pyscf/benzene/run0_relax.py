#!/usr/bin/env python3

from numpy import array

from stalk import ParameterStructure
from stalk.io import XyzGeometry
from stalk.util import Bohr

from params import forward, backward, relax_pyscf


# Let us initiate a ParameterStructure object that implements the parametric mappings
params_init = array([2.651, 2.055])
elem = 6 * ['C'] + 6 * ['H']
structure_init = ParameterStructure(
    forward=forward,
    backward=backward,
    params=params_init,
    elem=elem,
    units='B'
)

outfile = 'relax.xyz'
try:
    geom = XyzGeometry({'suffix': outfile, 'c_pos': Bohr**-1}).load('./')
except FileNotFoundError:
    new_params = relax_pyscf(structure_init, outfile)
    geom = XyzGeometry({'suffix': outfile, 'c_pos': Bohr**-1}).load('./')
# end try
new_params = structure_init.map_forward(geom.get_pos())
print(structure_init.params)
print(new_params)
structure_relax = structure_init.copy(params=new_params)
