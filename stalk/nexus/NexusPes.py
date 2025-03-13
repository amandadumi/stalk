#!/usr/bin/env python3

from nexus import run_project

from stalk.io.PesLoader import PesLoader
from stalk.nexus.NexusStructure import NexusStructure
from stalk.params.PesFunction import PesFunction
from stalk.util.EffectiveVariance import EffectiveVariance
from stalk.util.util import directorize

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


class NexusPes(PesFunction):
    '''A wrapper class for generating Nexus functions to produce and represent a PES.'''
    loader = None

    def __init__(
        self,
        func,
        args={},
        loader=None,
        load_func=None,
        load_args={},
    ):
        super().__init__(func, args)
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
        run_project(jobs)

        # Then, load
        for structure, sigma in zip(structures, sigmas):
            job_path = self._job_path(path, structure.label)
            res = self.loader.load(job_path)
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
        structure.jobs = jobs
    # end def

    def _prompt(self, structures: list[NexusStructure]):
        print("About to submit the following jobs:")
        for structure in structures:
            if structure.generated:
                for job in structure.jobs:
                    print('  {} {}'.format(job.path, job.identifier))
                # end for
            # end if
        # end for
        proceed = input("Proceed (Y/n)? ")
        if proceed == 'n':
            exit("")
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
