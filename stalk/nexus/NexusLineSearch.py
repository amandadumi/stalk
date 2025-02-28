#!/usr/bin/env python3

from nexus import run_project

from stalk.io.PesLoader import PesLoader
from stalk.ls.LineSearch import LineSearch
from stalk.nexus.NexusGenerator import NexusGenerator
from stalk.nexus.NexusStructure import NexusStructure


class NexusLineSearch(LineSearch):
    # List of NexusStructure instances (overrides _grid of LineSearchGrid)
    _grid: list[NexusStructure] = []

    @property
    def generated(self):
        return len(self) > 0 and all([point.generated for point in self._grid if point.enabled])
    # end def

    @property
    def analyzed(self):
        return len(self) > 0 and all([point.analyzed for point in self._grid if point.enabled])
    # end def

    @property
    def jobs(self):
        jobs = []
        for point in self._grid:
            if isinstance(point, NexusStructure) and point.generated:
                jobs += point._jobs
            # end if
        # end for
        return jobs
    # end def

    @property
    def enabled_jobs(self):
        jobs = []
        for point in self._grid:
            if isinstance(point, NexusStructure) and point.generated and point.enabled:
                jobs += point._jobs
            # end if
        # end for
        return jobs
    # end def

    def __init__(
        self,
        pes=None,
        pes_func=None,
        pes_args={},
        path='',
        loader=None,
        load_func=None,
        load_args={},
        add_sigma=False,
        **ls_args
        # structure=None, hessian=None, d=None, sigma=0.0, offsets=None, M=7, W=None
        # R=None, values=None, errors=None, fraction=0.025, sgn=1, fit_kind='pf3'
        # fit_func=None, fit_args={}, N=200, Gs=None
    ):
        # Omitting pes from LineSearch constructor leads to omission of evaluation
        super().__init__(**ls_args)
        try:
            self.evaluate(
                path=path,
                pes=pes,
                pes_func=pes_func,
                pes_args=pes_args
            )
            # If jobs were generated OK, run Nexus project
            if self.generated:
                run_project(self.enabled_jobs)
                self.analyze_jobs(
                    loader=loader,
                    load_func=load_func,
                    load_args=load_args,
                    add_sigma=add_sigma,
                )
            # end if
        except TypeError:
            # If pes is not yet provided, do not go on
            pass
        # end try
    # end def

    # Instead of a list of results, evaluate function returns a list of jobs to run
    def evaluate(
        self,
        path='',
        pes=None,
        pes_func=None,
        pes_args={}
    ):
        pes = NexusGenerator(pes, pes_func, pes_args)
        self._generate_eqm_jobs(pes, path=path)
        eqm_jobs = None
        eqm_point = self.find_point(0.0)
        if eqm_point is not None and isinstance(eqm_point, NexusStructure):
            eqm_jobs = eqm_point._jobs
        # end if
        # Generate jobs for non-eqm points (that may use eqm_jobs as dependency)
        for point in self._grid:
            if point is eqm_point:
                continue
            # end if
            point.generate_jobs(
                pes=pes,
                path=path,
                sigma=self.sigma,
                eqm_jobs=eqm_jobs
            )
        # end for
    # end def

    def _generate_eqm_jobs(
        self,
        pes: NexusGenerator,
        path='',
    ):
        # Try to find eqm point and generate jobs
        eqm_point = self.find_point(0.0)
        if eqm_point is not None and isinstance(eqm_point, NexusStructure) and not eqm_point.generated:
            eqm_point.generate_jobs(
                pes=pes,
                path=path,
                sigma=self.sigma
            )
        # end if
    # end def

    def analyze_jobs(
        self,
        loader=None,
        load_func=None,
        load_args={},
        add_sigma=False
    ):
        loader = PesLoader(loader, load_func, load_args)
        # Add sigma if add_sigma flag is on
        sigma = self.sigma if add_sigma else 0.0
        for point in self._grid:
            point.analyze_pes(loader, sigma=sigma)
        # end for
        # Search and store minimum
        self._search_and_store()
    # end def

# end class
