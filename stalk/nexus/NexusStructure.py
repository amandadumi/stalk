#!/usr/bin/env python3

from stalk.io.PesLoader import PesLoader
from stalk.util import EffectiveVariance
from structure import Structure
from nexus import run_project
from simulation import Simulation

from stalk.io.GeometryLoader import GeometryLoader
from stalk.params.ParameterStructure import ParameterStructure
from stalk.util.util import directorize

from .NexusGenerator import NexusGenerator


class NexusStructure(ParameterStructure):
    # List of Nexus jobs to reproduce PES
    _jobs: None | list[Simulation] = None
    _job_path = ''

    @property
    def jobs(self):
        return self._jobs
    # end def

    @jobs.setter
    def jobs(self, jobs):
        if jobs is None or len(jobs) == 0:
            self._jobs = None
        else:
            for job in jobs:
                if not isinstance(job, Simulation):
                    raise TypeError("Nexus job must be inherited from Simulation class!")
                # end if
            # end for
            self._jobs = jobs
        # end if
    # end def

    @property
    def generated(self):
        return self.jobs is not None
    # end def

    @property
    def analyzed(self):
        return self.generated and self.value is not None
    # end def

    @property
    def finished(self):
        return self.generated and all(job.finished for job in self._jobs)
    # end def

    def generate_jobs(
        self,
        pes,
        path='',
        sigma=None,
        eqm_jobs=None,
    ):
        if not isinstance(pes, NexusGenerator):
            raise TypeError('The pes must be inherited from NexusGenerator class.')
        # end if
        self._job_path = self._make_job_path(path)
        jobs = pes.generate(
            self.get_nexus_structure(),
            self._job_path,
            sigma=sigma,
            eqm_jobs=eqm_jobs
        )
        self._jobs = jobs
    # end def

    def analyze_pes(self, loader, sigma=0.0):
        if not isinstance(loader, PesLoader):
            raise TypeError('The loader must be inherited from PesFunction class.')
        # end if
        if not self.generated:
            raise AssertionError('The pes jobs must be generated before analyzing.')
        # end if
        # Sigma will be added automatically
        if self.enabled:
            res = loader.load(self._job_path, sigma=sigma)
            self.value = res.get_value()
            self.error = res.get_error()
        else:
            print("The point in " + self._job_path + " has been disabled. Not analyzing.")
        # end if
    # end def

    def _make_job_path(self, path):
        return '{}{}'.format(directorize(path), self.label)
    # end def

    def relax(
        self,
        pes=None,
        pes_func=None,
        pes_args={},
        path='relax',
        loader=None,
        loader_args={},
        **kwargs,
    ):
        if not isinstance(pes, NexusGenerator):
            # Generate jobs
            pes = NexusGenerator(pes_func, pes_args)
        # end if
        # Make a copy structure for job generation
        relax_jobs = pes.generate(self.get_nexus_structure(), directorize(path))
        # Run project
        run_project(relax_jobs)

        # Load results
        if isinstance(loader, GeometryLoader):
            # returns pos, axes
            pos, axes = loader.load(path, **loader_args).get_result()
            self.set_position(pos, axes)
        # end if
    # end def

    def get_nexus_structure(
        self,
        kshift=(0, 0, 0),
        kgrid=(1, 1, 1),
        **kwargs
    ):
        kwargs.update({
            'elem': self.elem,
            'pos': self.pos,
            'units': self.units,
        })
        if self.axes is not None:
            kwargs.update({
                'axes': self.axes,
                'kshift': kshift,
                'kgrid': kgrid,
            })
        # end if
        return Structure(**kwargs)
    # end def

    def get_var_eff(
        self,
        pes=None,
        pes_func=None,
        pes_args={},
        loader=None,
        loader_args={},
        samples=10,
        path='path'
    ):
        if not isinstance(pes, NexusGenerator):
            pes = NexusGenerator(pes_func, pes_args)
        # end if
        jobs = pes.generate(self.get_nexus_structure(), path, sigma=None, samples=samples)
        run_project(jobs)

        Err = loader.load(path, **loader_args).get_error()
        var_eff = EffectiveVariance(samples, Err)
        return var_eff
    # end def

    def reset_value(self):
        super().reset_value()
        # Reset jobs upon value change
        self._jobs = None
    # end def

    def copy(
        self,
        **kwargs
        # params=None, params_err=None, label=None, pos=None, axes=None, offset=None
    ):
        tmp_jobs = self._jobs
        tmp_job_path = self._job_path
        # Put jobs lists aside during copy
        self._jobs = None
        self._jobs_path = ''
        result = super().copy(**kwargs)
        # Recover jobs lists
        self._jobs = tmp_jobs
        self._job_path = tmp_job_path
        return result
    # end def

# end class
