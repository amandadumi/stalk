#!/usr/bin/env python3

from numpy import sin, cos, ndarray

from pyscf import dft
from pyscf import gto
from pyscf import grad
from pyscf.geomopt.geometric_solver import optimize
from pyscf.gto.mole import tofile

from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms
from ase.constraints import FixAtoms

from stalk.params.util import mean_distances, mean_param, angle, rotate_2d
from stalk import ParameterStructure
from stalk.params import PesFunction


# Forward mapping: produce parameter values from an array of atomic positions
def forward(pos: ndarray):
    pos = pos.reshape(-1, 3)  # make sure of the shape
    # for easier comprehension, list particular atoms
    N0 = pos[0]
    H0 = pos[1]
    H1 = pos[2]
    H2 = pos[3]

    # for redundancy, calculate mean bond lengths
    r = mean_distances([
        (N0, H0),
        (N0, H1),
        (N0, H2),
    ])
    # Calculate angle between x axis and H -> should be around 90 degrees
    x = [1.0, 0.0, 0.0]
    d0 = H0 - N0
    d1 = H1 - N0
    d2 = H2 - N0
    a = mean_param([
        angle(d0, x, units='rad'),
        angle(d1, x, units='rad'),
        angle(d2, x, units='rad'),
    ], tol=1e-4)
    params = [r, a]
    return params
# end def


# Backward mapping: produce array of atomic positions from parameters
def backward(params: ndarray):
    r = params[0]
    a = params[1]
    N0 = [0.0, 0.0, 0.0]
    x = r * cos(a)
    xy = [r * sin(a), 0.0]
    H0 = [x, *rotate_2d(xy, 0, units='ang')]
    H1 = [x, *rotate_2d(xy, 120, units='ang')]
    H2 = [x, *rotate_2d(xy, -120, units='ang')]
    pos = [N0, H0, H1, H2]
    return pos
# end def


def kernel_pyscf(positions, elem):
    atom = []
    for el, pos in zip(elem, positions):
        atom.append([el, tuple(pos)])
    # end for
    mol = gto.Mole()
    mol.atom = atom
    mol.verbose = 3
    mol.basis = 'ccpvdz'
    mol.unit = 'A'
    mol.ecp = 'ccecp'
    mol.charge = 0
    mol.spin = 0
    mol.symmetry = False
    mol.cart = True
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'pbe'
    return mf
# end def


def relax_pyscf(structure: ParameterStructure, outfile='relax.xyz'):
    mf = kernel_pyscf(structure.pos, structure.elem)
    mf.kernel()
    mol_eq = optimize(mf, maxsteps=100)
    # Write to external file
    tofile(mol_eq, outfile, format='xyz')
# end def


def pes_pyscf(structure: ParameterStructure, **kwargs):
    print(f'Computing: {structure.label}')
    mf = kernel_pyscf(structure.pos, structure.elem)
    e_scf = mf.kernel()
    return e_scf, 0.0
# end def


def pes_pyscf_lda(structure: ParameterStructure, **kwargs):
    print(f'Computing with LDA: {structure.label}')
    mf = kernel_pyscf(structure.pos, structure.elem)
    mf.xc = 'lda'
    e_scf = mf.kernel()
    return e_scf, 0.0
# end def


pes = PesFunction(pes_pyscf)
pes_lda = PesFunction(pes_pyscf_lda)


class PySCF(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)
    # end def

    def calculate(
        self,
        atoms=None,
        properties=['energy'],
        system_changes=all_changes
    ):
        Calculator.calculate(self, atoms, properties, system_changes)

        # Get positions and symbols from the Atoms object
        positions = self.atoms.get_positions()
        elem = self.atoms.get_chemical_symbols()

        # Calculate energy and forces here using your custom method
        energy = self.compute_energy(positions, elem)
        forces = self.compute_forces(positions, elem)

        # Set results to be used by ASE
        self.results = {'energy': energy, 'forces': forces}
    # end def

    def compute_energy(self, positions, elem):
        mf = kernel_pyscf(positions, elem)
        return mf.kernel()
    # end def

    def compute_forces(self, positions, elem):
        mf = kernel_pyscf(positions, elem)
        mf.kernel()
        mf_grad = grad.RKS(mf)
        forces = -mf_grad.kernel()
        return forces
    # end def

# end class


def neb_image(structure: ParameterStructure):
    atoms = Atoms(structure.elem, positions=structure.pos)
    atoms.calc = PySCF()
    # Fix N atom
    atoms.set_constraint(FixAtoms(indices=[0]))
    return atoms
# end def
