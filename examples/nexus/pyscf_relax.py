#!/usr/bin/env $python_exe

$pyscfimport

$system

$calculation

from pyscf.geomopt.geometric_solver import optimize
mol_eq = optimize(mf, maxsteps=100)
from pyscf.gto.mole import tofile
# Write to external file
tofile(mol_eq, 'relax.xyz', format='xyz')
# Write to output file 
print(mol_eq.atom_coords())
