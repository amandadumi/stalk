# Nexus examples

STALK integrates with Nexus workflow manager to enable highly automated generation and
management of numerical simulations, which is especially useful for operating QMCPACK. A few
examples are provided to demonstrate how the Nexus integrations of STALK can be used to
create automated workflows. 

The Nexus documentation here is minimal and intended for motivating the example layout. For
installation instructions and advanced use cases the user is referred to the
[Nexus User Guide](https://nexus-workflows.readthedocs.io/en/latest/).

## Preparations

The following documentation assumes a directory hierarchy similar to this one. To avoid
cluttering the git repository, it is recommended to copy the example directory to a work directory
with enough disk space, e.g.:
```
rsync -av <stalk_repo>/examples/nexus <wrk_dir>/stalk_examples
```
where `<stalk_repo>` and `<wrk_dir>` are to be filled appropriately.

Enter `<wkr_dir>` and use the provided script to download required pseudopotentials:
```
cd <wrk_dir>
./download_pseudos.sh
```

### Choosing an example

To pick any of the examples (e.g. `benzene`), navigate to its subfolder and copy the
Nexus settings template to `nxs.py`, e.g.
```
cd benzene
cp ../nxs_template.py nxs.py
```
Or, if you're feeling confident cutting corners, use symbolic link:
```
cd benzene
ln -s ../nxs_template.py nxs.py
```
This helps you set up and harmonize Nexus between all examples.

### Setting up Nexus

To manage workflows with Nexus, information of the environment, or `machine`, must be
supplied by calling `settings` with proper arguments. The configuration is done and can be
edited per project in `nxs.py` (see above), with the preset template enabling 8 cores on the
local machine.

NOTE: The heavier examples with DMC, such as benzene, may take a few hours to complete on 8
local cores, and so it will be left as a decent exercise to seek and configure for bigger
resources. This is easy enough with Nexus.

### Software prerequisites

Make sure to have the following software installed and available (depending on the example)
* STALK (0.2.0)
* PySCF (2.8.0)
* Quantum Espresso (7.4.0)
* QMCPACK (4.0.0)

The version numbers in parentheses are recommendations, not systematically tested. Quantum
Espresso must be compiled with `-DQE_ENABLE_PW2QMCPACK=1` to create `pw2qmcpack.x`
converter.

Paths to the appropriate binaries are set in `nxs.py` for each job type. The `presub`
argument (although empty in the preset) allows, for example, to load required modules.
Diversity of the job settings (e.g. different QMC jobs and/or QMCPACK builds for
optimization and DMC) are easy to edit, but that goes beyond this documentation.

## Understanding the example

In each example, the parallel line-search workflow is divided into a few stages that
correspond to python scripts.

NOTE: The followin is definitely not the only way to lay out the project; the whole thing
could be contained in a single script, or a Jupyter notebook. The present layout is the
Author's choice and it has the following benefit: it is easy to inspect, troubleshoot and
work through the steps in order, one by one.

### params.py

This file is the most important and elaborate as it contains the core definitions frequently
used by the actionable scripts.

First, define the geometric problem by composing `forward` and `backward` mappings that map
an array of real-space positions into a reduced set of scalar parameters, and back. It also
sets the very essence of the numerical problem, so the parameter choices are important and
can only be changed later with care (meaning: the parameter Hessian and the line-searches
depend on the parametric definitions and may have to be done over).

At least the translational and rotational degrees of freedom are reduced. Consequently, the
backward mapping must make choices of origin and orientation that are agnostic of the
parameter array. Furthermore, the ground states and periodic structures are often symmetric,
which further reduces degrees of freedom and requires the symmetries to be assumed and
incorporated in the `backward` mapping functions.

In the absence of automatic tools, the mappings must be implemented manually. The examples
demonstrate different ways of doing this.

Second, this file also contains definitions of the (numerical) potential energy surfaces
(PESs). In the Nexus mode, this amounts to pairs of job-generating and job-analyzing
functions.

For example, a geometry relaxation job is a `NexusGeometry` object, which is constructed out
of
1. A function taking `Structure` and `str` (path) that returns a list of Nexus simulations
2. A loader class to analyze the simulation results.

NOTE: In the examples, the PySCF relaxation job is specifically instructed to write the
atomic positions to an XYZ file that will be accessed by the `XyzGeometry` loader. Without
this consistency, the relaxed geometry must be manually supplied between runs.

As seen in the DMC PES definitions, the job-generation function may be more complicated with
dependencies and dynamic parameters, such as the number of samples. The loader can also be
more complicated, doing rescaling and postprocessing, and it is possible to tailor one per
project by inheriting from the `PesLoader` class. Some of these topics will be covered
later. Generally, to understand Nexus simulation jobs and their configuration, we refer to
the [Nexus User Guide](https://nexus-workflows.readthedocs.io/en/latest/).

### run0_relax.py

With parameter mappings and the surrogate PES defined, we must find the corresponding
equilibrium. This is usually most efficiently done using build-in relaxation of the
surrogate method.

First, we use the parametric mappings to create a starting atomic structure. Then, we
supply a copy of the structure to the relaxation job, where it gets updated.

### run1_hessian.py

Second, we create a ParameterHessian (i.e. a Hessian in the parametric subspace) and
evaluate it around the equilibrium geometry by using a finite-difference method.

### run2_surrogate.py

Third, using conjugate directions from the parametric Hessian, we create a surrogate model
of the parallel line-search, which we optimize to find equilibrium at a given tolerance.

We use the parallel line-search framework to explore the surrogate PES widely around
the equilibrium, along the conjugate directions. This data is then used to simulate and
resample line-search fitting performance relative to the well-know target (the zero
displacement, as we are in the equilibrium). Hence, the class responsible is called
`TargetParallelLineSearch`.

With the surrogate PES data, we can anticipate and optimize the performance of a line-search
of any given type (or rather, a set of them). In other words, we may request a set of
line-search parameters maintaining that the total fitting errors (bias + noise) stay below a
given tolerance. The resulting parameters include the line-search fit kind (e.g. 3rd order
polynomial), grid size (e.g. 7), grid extent per direction ("window"), and how much
statistical noise is tolerated ("sigma").

Generally, larger window causes larger bias but tolerates higher noise, and so the optimum
(most noise allows) lies somewhere between "small" and "large" window. The optimizer
attempts to balance all conjugate directions simultaneously to minimize the global cost. As
a result, the surrogate model will obtains line-search parameters for each individual
direction.

NOTE: The line-search optimizer operates on real data and can be sensitive to different
sources of error and misproportions of parameters. For instance, it may prove hard to seek
very low tolerances, if the data is poor or too scarce near the equilibrium. The stability
of the optimizer is a work in progress.

### run3_scf_noise.py

Fourth, the surrogate model can be checked for sanity by using its parameters to perform a
mock-up line-search. In this surrogate line-search, we may derive the whole line-search
configuration from the optimized model, except shift the starting configuration and observe
it being recovered to the equilibrium.

To simulate statistical fluctuation, we may artificially add the target errorbar (sigma) to
the analyzed energies and their uncertainties.

### run4_dmc_ls.py

Finally, we'll use the surrogate model to coordinate parallel line-search iteration on the
DMC PES. This is similar to the above SCF+noise line-search, except that there is no need to
shift starting parameters or add artificial noise, because both effects arise naturally from
the QMC simulation.

However, we must first characterize the effective variance of the QMC simulation with the
given setup: we first calculate the QMC energy (at surrogate equilibrium) with an arbitrary
scale of Monte Carlo samples (`samples=10`). From the resulting variance, we can anticipate
the errorbar with a different number of samples, allowing us to target an errorbar (to meet
"sigma") by scaling `samples`. This information is supplied to the DMC PES function, where
it must be treated correctly.

With the effective variance in place, the DMC line-search can be carried out.

NOTE: The DMC simulation of the examples is simple and intended for demonstration purposes
only. In scientific applications, more attention may be in order to treat numerical effects
due to finite time-step, DMC walker population, finite-size effects, basis sets,
optimization, etc.

## Running the example

It is recommended to run the examples in order on the command line, starting from
```
python3 run0_relax.py
```
and moving on. However, one may always be bold an start from 
```
python3 run4_dmc_ls.py
```
and observe how the preliminary steps are carried out one by one.

At any stage, it is also possible to load and inspect the steps up to that point, e.g.
```
python> from run1_hessian import *
python> print(hessian)
```
This can be extremely helpful in troubleshooting.

## Making your own project

The examples are made to show how a STALK line-search project can be composed. While the
examples are expected to work nicely, new project always face issues and require tinkering
with functions and parameters.

A recommended way of composing a new project is therefore to copy a suitable example one
file at a time, while adapting it as desired. That is,
1. Copy `nxs.py` and `params.py` and write the parameter mappings and surrogate PES
2. Copy `run0_relax.py` and execute it to assert mappings consistency
3. Copy `run1_hessian.py` and execute it to study the Hessian
4. Copy `run2_surrogate.py` and execute it. Play around with optimizer to figure out
reasonable tolerances and bottlenecks.
5. Copy `run3_srg_ls.py` and execute it to make sure that the line-search works
6. Copy `run4_dmc_ls.py` and execute it to get the derired results.

NOTE: Various effects may ensue, such as SCF convergence issues, that require changes to
simulation parameters and even the parametric mappings. Thus, we must emphasize that while
the line-search project operates on numerical recipes (i.e. Nexus job-generators) that
appear concise and "final", it may take serious background work (or live development) to
come up with recipes that work robustly. Rather, one may end up treating problematic
data points individually, or even concluding that the data is unattainable in some regions
of the parametric phase space.

Anyway, good luck!
