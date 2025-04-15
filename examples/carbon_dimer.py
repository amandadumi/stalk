#!/usr/bin/env python3

# C2: 1D line-search example
#   Surrogate theory: DFT (PBE)
#   Stochastic tehory: DMC
#
# Computing task: Suitable for personal computer

from numpy import array

from nexus import generate_pyscf, generate_qmcpack, job, obj
from nexus import generate_physical_system, generate_convert4qmc
from stalk.ls.LineSearch import LineSearch
from stalk.ls.TargetLineSearch import TargetLineSearch
from stalk.params.util import distance
from structure import Structure

from stalk.io.FilesLoader import FilesLoader
from stalk.io.XyzGeometry import XyzGeometry
from stalk.nexus import NexusStructure
from stalk.nexus.NexusGeometry import NexusGeometry
from stalk.nexus.NexusPes import NexusPes
from stalk.nexus.QmcPes import QmcPes
from stalk.params.ParameterHessian import ParameterHessian
from stalk.params.PesFunction import PesFunction
from stalk.util import EffectiveVariance

from nxs import pyscfjob, optjob, dmcjob

# Pseudos (execute download_pseudos.sh in the working directory)
base_dir = 'C2/'
qmcpseudos = ['C.ccECP.xml']
interactive = False


# Forward mapping: produce parameter values from an array of atomic positions
def forward(pos, **kwargs):
    pos = pos.reshape(-1, 3)  # make sure of the shape
    C0 = pos[0]
    C1 = pos[1]
    d = distance(C0, C1)
    params = array([d])
    return params
# end def


# Backward mapping: produce array of atomic positions from parameters
def backward(params, **kwargs):
    d = params[0]
    C0 = [0.0, 0.0, -d / 2]
    C1 = [0.0, 0.0, +d / 2]
    pos = array([C0, C1]).flatten()
    return pos
# end def


# Let us initiate a NexusStructure object that conforms to the above mappings
structure_init = NexusStructure(
    forward=forward,
    backward=backward,
    params=[1.54],
    elem=2 * ['C'],
    units='A'
)


# return a 1-item list of Nexus jobs: SCF relaxation
def scf_relax_job(structure: Structure, path, **kwargs):
    system = generate_physical_system(structure=structure, C=4)
    relax = generate_pyscf(
        template='pyscf_relax.py',
        system=system,
        identifier='relax',
        job=job(**pyscfjob),
        path=path,
        mole=obj(
            spin=4,
            verbose=4,
            ecp='ccecp',
            basis='ccpvdz',
            symmetry=False,
        ),
    )
    return [relax]
# end def


# LINE-SEARCH

# 1) Surrogate: relaxation
structure_relax = structure_init.copy()
relax_job = NexusGeometry(
    scf_relax_job,
    loader=XyzGeometry({'suffix': 'relax.xyz'})
)
relax_job.relax(
    structure_relax,
    path=base_dir + 'relax/',
    interactive=interactive
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
        net_spin=4,
        C=4
    )
    scf = generate_pyscf(
        template='pyscf_pes.py',
        system=system,
        identifier='scf',
        job=job(**pyscfjob),
        path=path,
        mole=obj(
            spin=4,
            verbose=4,
            ecp='ccecp',
            basis='ccpvdz',
            symmetry=False,
        ),
    )
    return [scf]
# end def


# Hessian based on the structural mappings
pes = NexusPes(
    func=PesFunction(scf_pes_job),
    args={'path': 'pbe'},
    loader=FilesLoader({'suffix': 'energy.dat'})
)
hessian = ParameterHessian(structure=structure_relax)
hessian.compute_fdiff(
    path=base_dir + 'fdiff',
    pes=pes,
    interactive=interactive
)
print('Hessian:')
print(hessian)

# 3) Surrogate: Optimize line-search
# Use a macro to generate a parallel line-search object that samples the
# surrogate PES around the minimum along the search directions
surrogate = TargetLineSearch(
    d=0,
    fit_kind='pf3',
    structure=structure_relax,
    hessian=hessian,
    R=0.4,
    M=11
)
surrogate.evaluate(
    pes=pes,
    path=base_dir + 'surrogate',
    interactive=interactive
)
surrogate.reset_interpolation(interpolate_kind='cubic')

# Set target parameter error tolerances (epsilon): 0.01 Angstrom accuracy
# Then, optimize the surrogate line-search to meet the tolerances given the line-search
surrogate.optimize(
    epsilon=0.02,
    fit_kind='pf3',
    M=7,
    N=400,
    noise_frac=0.01,
)


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

    # Center the structure for QMCPACK
    system = generate_physical_system(
        structure=structure,
        C=4,
        net_spin=4,
    )
    scf = generate_pyscf(
        template='pyscf_pes.py',
        system=system,
        identifier='scf',
        job=job(**pyscfjob),
        path=path + 'scf',
        mole=obj(
            spin=4,
            verbose=4,
            ecp='ccecp',
            basis='augccpvtz',
            symmetry=False,
        ),
        save_qmc=True,
    )
    c4q = generate_convert4qmc(
        identifier='c4q',
        path=path + 'scf',
        job=job(cores=1),
        dependencies=(scf, 'orbitals'),
    )
    opt = generate_qmcpack(
        system=system,
        path=path + 'opt',
        job=job(**optjob),
        dependencies=[(c4q, 'orbitals')],
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
        dependencies=[(c4q, 'orbitals'), (opt, 'jastrow')],
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
    return [scf, c4q, opt, dmc]
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
    samples=10,
    interactive=interactive
)
qmcpes.args['var_eff'] = var_eff

# Finally, perform line-search iteration based on surrogate settings and DMC PES
dmc_ls = LineSearch(**surrogate.to_settings())
dmc_ls.evaluate(
    path=base_dir + 'dmc_ls',
    pes=qmcpes,
    interactive=interactive
)

# Diagnose the line-search performance
print(dmc_ls)
