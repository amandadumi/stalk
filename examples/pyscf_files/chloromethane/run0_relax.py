#!/usr/bin/env python3

from numpy import array
from pyscf.geomopt.geometric_solver import optimize
from pyscf.gto.mole import tofile

from stalk import ParameterStructure
from stalk.io import XyzGeometry

from params import forward, backward, kernel_pyscf


# Let us initiate a ParameterStructure object that implements the parametric mappings
params_init = array([1.8, 0.2, 1.0])
elem = ['C'] + ['Cl'] + 3 * ['H']
structure = ParameterStructure(
    forward=forward,
    backward=backward,
    params=params_init,
    elem=elem,
    units='A',
    label='init'
)

outfile = 'relax.xyz'
try:
    geom = XyzGeometry({'suffix': outfile}).load('./')
except FileNotFoundError:
    mf = kernel_pyscf(structure=structure)
    mf.xc = 'pbe'
    mf.kernel()
    mol_eq = optimize(mf, maxsteps=100)
    # Write to external file
    tofile(mol_eq, outfile, format='xyz')
    geom = XyzGeometry({'suffix': outfile}).load('./')
# end try
new_params = structure.map_forward(geom.get_pos())
print('Initial params:')
print(structure.params)
print('Relaxed params:')
print(new_params)
structure_relax = structure.copy(params=new_params)
