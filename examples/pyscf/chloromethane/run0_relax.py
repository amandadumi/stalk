#!/usr/bin/env python3

from numpy import array

from stalk import ParameterStructure
from stalk import XyzGeometry

from params import forward, backward, relax_pyscf


# Let us initiate a ParameterStructure object that implements the parametric mappings
params_init = array([1.8, 0.2, 1.0])
elem = ['C'] + ['Cl'] + 3 * ['H']
structure_init = ParameterStructure(
    forward=forward,
    backward=backward,
    params=params_init,
    elem=elem,
    units='A'
)

xyz = XyzGeometry(suffix='relax.xyz')
structure_relax = xyz.load_or_relax(
    path='./',
    relax_func=relax_pyscf,
    structure=structure_init
)

if __name__ == '__main__':
    print('Initial parameters:')
    print(structure_init.params)
    print('Relaxed parameters:')
    print(structure_relax.params)
# end if
