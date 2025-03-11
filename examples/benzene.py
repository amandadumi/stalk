#!/usr/bin/env python3

# Benzene: line-search example
#   2-parameter problem: CC/CH bond lengths
#   Surrogate theory: DFT (PBE)
#   Stochastic tehory: DMC
#
# This example executes a standard line-search minimization workflow for the
# ground state of the benzene molecule, using DFT (PBE) as the surrogate
# method and Diffusion Monte Carlo (DMC) as the stochastic method.
#
# Computing task: Suitable for institutional clusters

# First, the user must set up Nexus according to their computing environment.

from numpy import diag, mean, array, sin, pi, cos

from nexus import generate_pyscf, generate_qmcpack, job, obj
from nexus import generate_physical_system, generate_pw2qmcpack, generate_pwscf
from structure import Structure

from stalk.util.util import Bohr
from stalk.io.FilesLoader import FilesLoader
from stalk.io.XyzGeometry import XyzGeometry
from stalk.nexus import NexusStructure
from stalk.nexus.NexusGeometry import NexusGeometry
from stalk.nexus.NexusPes import NexusPes
from stalk.nexus.QmcPes import QmcPes
from stalk.lsi import LineSearchIteration
from stalk.params.ParameterHessian import ParameterHessian
from stalk.params.PesFunction import PesFunction
from stalk.pls.TargetParallelLineSearch import TargetParallelLineSearch
from stalk.util import EffectiveVariance

from nxs import pyscfjob, optjob, dmcjob, pwscfjob, p2qjob

# Pseudos (execute download_pseudos.sh in the working directory)
base_dir = 'benzene/'
qmcpseudos = ['C.ccECP.xml', 'H.ccECP.xml']
scfpseudos = ['C.ccECP.upf', 'H.ccECP.upf']

# Implement the following parametric mappings for benzene
#   p0: C-C distance
#   p1: C-H distance


# Forward mapping: produce parameter values from an array of atomic positions
def forward(pos, **kwargs):
    pos = pos.reshape(-1, 3)  # make sure of the shape
    # for easier comprehension, list particular atoms
    C0 = pos[0]
    C1 = pos[1]
    C2 = pos[2]
    C3 = pos[3]
    C4 = pos[4]
    C5 = pos[5]
    H0 = pos[6]
    H1 = pos[7]
    H2 = pos[8]
    H3 = pos[9]
    H4 = pos[10]
    H5 = pos[11]

    def distance(r1, r2):
        return sum((r1 - r2)**2)**0.5
    # end def
    # for redundancy, calculate mean bond lengths
    # 0) from neighboring C-atoms
    r_CC = mean([distance(C0, C1),
                 distance(C1, C2),
                 distance(C2, C3),
                 distance(C3, C4),
                 distance(C4, C5),
                 distance(C5, C0)])
    # 1) from corresponding H-atoms
    r_CH = mean([distance(C0, H0),
                 distance(C1, H1),
                 distance(C2, H2),
                 distance(C3, H3),
                 distance(C4, H4),
                 distance(C5, H5)])
    params = array([r_CC, r_CH])
    return params
# end def


# Backward mapping: produce array of atomic positions from parameters
def backward(params, **kwargs):
    r_CC = params[0]
    r_CH = params[1]
    # place atoms on a hexagon in the xy-directions
    hex_xy = array([[cos(3 * pi / 6), sin(3 * pi / 6), 0.],
                    [cos(5 * pi / 6), sin(5 * pi / 6), 0.],
                    [cos(7 * pi / 6), sin(7 * pi / 6), 0.],
                    [cos(9 * pi / 6), sin(9 * pi / 6), 0.],
                    [cos(11 * pi / 6), sin(11 * pi / 6), 0.],
                    [cos(13 * pi / 6), sin(13 * pi / 6), 0.]])
    # C-atoms are one C-C length apart from origin
    pos_C = hex_xy * r_CC
    # H-atoms one C-H length apart from C-atoms
    pos_H = hex_xy * (r_CC + r_CH)
    pos = array([pos_C, pos_H]).flatten()
    return pos
# end def


# Let us initiate a ParameterStructure object that conforms to the above mappings
params_init = array([2.651, 2.055])
elem = 6 * ['C'] + 6 * ['H']
structure_init = NexusStructure(
    forward=forward,
    backward=backward,
    params=params_init,
    elem=elem,
    units='B'
)


# return a 1-item list of Nexus jobs: SCF relaxation
def scf_relax_job(structure: Structure, path, **kwargs):
    system = generate_physical_system(
        structure=structure,
        C=4,
        H=1
    )
    relax = generate_pyscf(
        template='pyscf_relax.py',
        system=system,
        identifier='relax',
        job=job(**pyscfjob),
        path=path,
        mole=obj(
            verbose=4,
            ecp='ccecp',
            basis='ccpvdz',
            symmetry=False,
            cart=True
        ),
    )
    return [relax]
# end def


# LINE-SEARCH

# 1) Surrogate: relaxation
structure_relax = structure_init.copy()
relax_job = NexusGeometry(
    scf_relax_job,
    loader=XyzGeometry({'suffix': 'relax.xyz', 'c_pos': 1.0 / Bohr})
)
relax_job.relax(
    structure_relax,
    path=base_dir + 'relax/',
)
print('Initial params:')
print(structure_init.params)
print('Relaxed params:')
print(structure_relax.params)


# 2) Surrogate: Hessian
# Let us define an SCF PES job that is consistent with the earlier relaxation
def scf_pes_job(structure: Structure, path, **kwargs):
    system = generate_physical_system(
        structure=structure,
        C=4,
        H=1,
    )
    scf = generate_pyscf(
        template='pyscf_pes.py',
        system=system,
        identifier='scf',
        job=job(**pyscfjob),
        path=path,
        mole=obj(
            verbose=4,
            ecp='ccecp',
            basis='ccpvdz',
            symmetry=False,
            cart=True
        ),
    )
    return [scf]
# end def


# Hessian based on the structural mappings
pes = NexusPes(
    func=PesFunction(scf_pes_job),
    loader=FilesLoader({'suffix': 'energy.dat'})
)
hessian = ParameterHessian(structure=structure_relax)
hessian.compute_fdiff(
    path=base_dir + 'fdiff',
    pes=pes
)
print('Hessian:')
print(hessian)


# 3) Surrogate: Optimize line-search
# Use a macro to generate a parallel line-search object that samples the
# surrogate PES around the minimum along the search directions
surrogate_file = 'surrogate.p'
surrogate = TargetParallelLineSearch(
    path=base_dir + 'surrogate/',
    fit_kind='pf3',
    load=surrogate_file,
    structure=structure_relax,
    hessian=hessian,
    pes=pes,
    window_frac=0.2,  # maximum displacement relative to Lambda of each direction
    M=15  # number of points per direction to sample
)
surrogate.bracket_target_biases()

# Set target parameter error tolerances (epsilon): 0.01 Bohr accuracy for both C-C and C-H bonds.
# Then, optimize the surrogate line-search to meet the tolerances given the line-search
#   main input: M, epsilon
#   main output: windows, noises (per direction to meet all epsilon)
epsilon_p = [0.01, 0.01]
surrogate.optimize(
    epsilon_p=epsilon_p,
    fit_kind='pf3',
    M=7,
    N=400,
    reoptimize=False,
    write=surrogate_file,
)
print(surrogate)

# The check (optional) the performance, let us simulate a line-search on the surrogate PES.
# It is cheaper to debug the optimizer here than later on.
# First, shift parameters for the show
shifted_structure = surrogate.structure.copy()
shifted_structure.shift_params([0.1, -0.1])
# Then generate line-search iteration object based on the shifted surrogate
srg_ls = LineSearchIteration(
    surrogate=surrogate,
    structure=shifted_structure,
    path=base_dir + 'srg_ls',
    pes=pes,
)
# Propagate the parallel line-search (compute values, analyze, then move on) 4 times
#   add_sigma = True means that target errorbars are used to simulate random noise
for i in range(4):
    srg_ls.propagate(i, add_sigma=True)
# end for
# Evaluate the latest eqm structure
srg_ls.pls().evaluate_eqm(add_sigma=True)
# Print the line-search performance
print(srg_ls)


# 4-5) Stochastic: Line-search
# return a simple 4-item DMC workflow for Nexus:
#   1: run SCF with norm-conserving ECPs to get orbitals
#   2: convert for QMCPACK
#   3: Optimize 2-body Jastrow coefficients
#   4: Run DMC with enough steps/block to meet the target errorbar sigma
#     it is important to first characterize the DMC run into var_eff
def dmc_pes_job(
    structure: Structure,
    path,
    sigma=None,
    samples=10,
    var_eff=None,
    **kwargs
):
    # Estimate the relative number of samples needed
    if isinstance(var_eff, EffectiveVariance):
        dmcsteps = var_eff.get_samples(sigma)
    else:
        dmcsteps = samples
    # end if

    # For QMCPACK, use plane-waves for better performance
    axes = array([20., 20., 10.])
    structure.set_axes(diag(axes))
    structure.pos += axes / 2
    structure.kpoints = [[0, 0, 0]]
    system = generate_physical_system(
        structure=structure,
        C=4,
        H=1,
    )
    scf = generate_pwscf(
        system=system,
        job=job(**pwscfjob),
        path=path + '/scf',
        pseudos=scfpseudos,
        identifier='scf',
        calculation='scf',
        input_type='generic',
        input_dft='pbe',
        nosym=False,
        nogamma=True,
        conv_thr=1e-9,
        mixing_beta=.7,
        ecutwfc=300,
        ecutrho=600,
        occupations='smearing',
        smearing='gaussian',
        degauss=0.0001,
        electron_maxstep=1000,
        kgrid=(1, 1, 1),  # needed to run plane-waves with Nexus
        kshift=(0, 0, 0,),
    )
    p2q = generate_pw2qmcpack(
        identifier='p2q',
        path=path + '/scf',
        job=job(**p2qjob),
        dependencies=[(scf, 'orbitals')],
    )
    system.bconds = 'nnn'
    opt = generate_qmcpack(
        system=system,
        path=path + 'opt',
        job=job(**optjob),
        dependencies=[(p2q, 'orbitals')],
        cycles=8,
        identifier='opt',
        qmc='opt',
        input_type='basic',
        pseudos=qmcpseudos,
        J2=True,
        J1_size=6,
        J1_rcut=6.0,
        J2_size=8,
        J2_rcut=8.0,
        minmethod='oneshift',
        blocks=200,
        substeps=2,
        samples=100000,
        minwalkers=0.1,
    )
    dmc = generate_qmcpack(
        system=system,
        path=path + 'dmc',
        job=job(**dmcjob),
        dependencies=[(p2q, 'orbitals'), (opt, 'jastrow')],
        steps=dmcsteps,
        identifier='dmc',
        qmc='dmc',
        input_type='basic',
        pseudos=qmcpseudos,
        jastrows=[],
        walkers_per_rank=128,
        blocks=200,
        timestep=0.01,
        ntimesteps=1,
    )
    return [scf, p2q, opt, dmc]
# end def


# Configure a job generator and loader for the DMC PES
# -the suffix points to the correct nexus analyzer
# -the qmc_idx points to the correct QMC series (0: VMC; 1: DMC)
qmcpes = NexusPes(
    dmc_pes_job,
    loader=QmcPes({'suffix': '/dmc/dmc.in.xml', 'qmc_idx': 1})
)
structure_qmc = structure_relax.copy()


# Run a snapshot job to sample effective variance w.r.t relative DMC samples
var_eff = qmcpes.get_var_eff(
    structure_qmc,
    path=base_dir + 'dmc_var_eff',
    samples=10
)
qmcpes.args['var_eff'] = var_eff

# Finally, perform line-search iteration based on surrogate settings and DMC PES
dmc_ls = LineSearchIteration(
    surrogate=surrogate,
    path=base_dir + 'dmc_ls',
    pes=qmcpes
)
for i in range(3):
    dmc_ls.propagate(i)
# end for
pes.evaluate(srg_ls.pls().structure)

# Diagnose the line-search performance
print(dmc_ls)
