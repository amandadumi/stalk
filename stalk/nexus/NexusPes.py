#!/usr/bin/env python3
'''A wrapper class for generating Nexus functions to produce and represent a PES.'''

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

from numpy import isnan, isscalar
from pickle import load

from nexus import run_project, bundle

from stalk.io.PesLoader import PesLoader
from stalk.nexus.NexusStructure import NexusStructure
from stalk.params.PesFunction import PesFunction
from stalk.util.EffectiveVariance import EffectiveVariance
from stalk.util.util import directorize


class NexusPes(PesFunction):
    loader = None
    disable_failed = False
    bundle_jobs = False

    def __init__(
        self,
        func,
        args={},
        loader=None,
        load_func=None,
        load_args={},
        disable_failed=False,
        bundle_jobs=False
    ):
        super().__init__(func, args)
        self.disable_failed = disable_failed
        self.bundle_jobs = bundle_jobs
        if isinstance(loader, PesLoader):
            self.loader = loader
        else:
            self.loader = PesLoader(load_func, load_args)
        # end if
    # end def

    # Override evaluation function to support job submission and analysis
    def evaluate(
        self,
        structure: NexusStructure,
        sigma=0.0,
        add_sigma=False,
        eqm_jobs=None,
        path='',
        interactive=False,
        **kwargs
    ):
        job_path = self._job_path(path, structure.label)
        # TODO: try to load first, to assess whether to regenerate or not
        self._evaluate_structure(
            structure,
            job_path,
            sigma=sigma,
            eqm_jobs=eqm_jobs,
            **kwargs
        )
        if interactive:
            self._prompt([structure])
        # end if
        run_project(structure._jobs)
        res = self.loader.load(job_path)
        # Treat failure
        if isnan(res.value) and self.disable_failed:
            structure.enabled = False
        # end if
        if add_sigma:
            res.add_sigma(sigma)
        # end if
        structure.value = res.value
        structure.error = res.error
    # end def

    # Override evaluation function to support parallel job submission and analysis
    def evaluate_all(
        self,
        structures: list[NexusStructure],
        sigmas=None,
        add_sigma=False,
        path='',
        interactive=False,
        **kwargs
    ):
        if sigmas is None:
            sigmas = len(structures) * [0.0]
        # end if
        eqm_jobs = None
        # Try to find eqm
        for structure, sigma in zip(structures, sigmas):
            if structure.label == 'eqm':
                job_path = directorize(path) + structure.label
                eqm_jobs = self._evaluate_structure(
                    structure,
                    job_path,
                    sigma=sigma,
                    **kwargs,
                )
                break
            # end if
        # end for
        jobs = []
        for structure, sigma in zip(structures, sigmas):
            if structure.label == 'eqm':
                # Do not generate eqm jobs twice
                if structure.jobs is not None:
                    jobs += structure.jobs
                # end if
                continue
            # end if
            # Make a copy structure for job generation
            job_path = self._job_path(path, structure.label)
            if not structure.analyzed:
                self._evaluate_structure(
                    structure,
                    job_path,
                    sigma=sigma,
                    eqm_jobs=eqm_jobs,
                    **kwargs,
                )
                jobs += structure.jobs
            # end if
        # end for
        # TODO: try to load first, to assess whether to regenerate or not
        if interactive:
            self._prompt(structures)
        # end if
        if self.bundle_jobs:
            run_project(bundle(jobs))
        else:
            run_project(jobs)
        # end if

        # Then, load
        for structure, sigma in zip(structures, sigmas):
            job_path = self._job_path(path, structure.label)
            res = self.loader.load(job_path)
            # Treat failure
            if isnan(res.value) and self.disable_failed:
                structure.enabled = False
            # end if
            if add_sigma:
                res.add_sigma(sigma)
            # end if
            structure.value = res.value
            structure.error = res.error
        # end for
    # end def

    def _job_path(self, path, label):
        return f'{directorize(path)}{label}/'
    # end def

    def _evaluate_structure(
        self,
        structure: NexusStructure,
        job_path,
        eqm_jobs=None,
        sigma=0.0,
        **kwargs
    ):
        # Do not redo jobs
        if structure.generated:
            return
        # end if
        eval_args = self.args.copy()
        # Override with kwargs
        eval_args.update(**kwargs)
        jobs = self.func(
            structure.get_nexus_structure(),
            directorize(job_path),
            sigma=sigma,
            eqm_jobs=eqm_jobs,
            **eval_args
        )
        structure.sigma = sigma
        structure.jobs = jobs
    # end def

    def _prompt(self, structures: list[NexusStructure]):
        new_job_strs = []
        for structure in structures:
            if structure.generated:
                for job in structure.jobs:
                    sim_path = '{}/sim_{}/sim.p'.format(job.path, job.identifier)
                    finished = False
                    try:
                        with open(sim_path, mode='rb') as f:
                            sim = load(f)
                            finished = sim.finished
                        # end with
                    except (FileNotFoundError, AttributeError):
                        pass
                    # end try
                    if not finished:
                        job_str = '  {}'.format(job.path)
                        if hasattr(job, "samples") and isscalar(job.samples):
                            job_str += f' ({job.samples}x samples)'
                        # end if
                        new_job_strs.append(job_str)
                    # end if
                # end for
            # end if
        # end for
        if len(new_job_strs) > 0:
            print("About to submit the following jobs:")
            for job_str in new_job_strs:
                print(job_str)
            # end for
            proceed = input("Proceed (Y/n)? ")
            if proceed == 'n':
                exit("Submission cancelled by user.")
            # end if
        # end if
    # end def

    def get_var_eff(
        self,
        structure: NexusStructure,
        path='path',
        samples=10,
        interactive=False,
    ):
        self.evaluate(
            structure,
            path=path,
            sigma=None,
            samples=samples,
            interactive=interactive,
        )
        var_eff = EffectiveVariance(samples, structure.error)
        return var_eff
    # end def

    def relax(
        *args,
        **kwargs
    ):
        msg = "Relaxation not implemented in NexusPes class, use NexusGeometry instead"
        raise NotImplementedError(msg)
    # end def

# end class
