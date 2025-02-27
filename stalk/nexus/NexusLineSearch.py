#!/usr/bin/env python3

from nexus import run_project

from stalk.ls.LineSearch import LineSearch
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
        path='',
        loader=None,
        add_sigma=False,
        postpone_analyze=False,
        **ls_args
        # structure=None, hessian=None, d=None, sigma=0.0, offsets=None, M=7, W=None
        # R=None, values=None, errors=None, fraction=0.025, sgn=1, fit_kind='pf3'
        # fit_func=None, fit_args={}, N=200, Gs=None
    ):
        super().__init__(**ls_args)
        if pes is not None:
            # pes type is checked in NexusStructure class
            self.evaluate_pes(
                pes,
                path=path,
                loader=loader,
                add_sigma=add_sigma,
                postpone_analyze=postpone_analyze
            )
        # end if
    # end def

    def evaluate_pes(
        self,
        pes,
        path='',
        loader=None,
        add_sigma=False,
        postpone_analyze=False,
    ):
        self.generate_jobs(pes, path=path)
        # Next, the jobs need running either here or externally
        if not postpone_analyze:
            run_project(self.enabled_jobs)
            self.analyze_jobs(loader, add_sigma=add_sigma)
        # end if
    # end def

    def generate_jobs(
        self,
        pes,
        path='',
    ):
        self.generate_eqm_jobs(pes, path=path)
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

    def generate_eqm_jobs(
        self,
        pes,
        path='',
    ):
        # Try to find eqm point and generate jobs
        eqm_point = self.find_point(0.0)
        if eqm_point is not None and isinstance(eqm_point, NexusStructure):
            eqm_point.generate_jobs(
                pes=pes,
                path=path,
                sigma=self.sigma
            )
        # end if
    # end def

    def analyze_jobs(
        self,
        loader,
        add_sigma=False
    ):
        sigma = self.sigma if add_sigma else 0.0
        for point in self._grid:
            point.analyze_pes(loader, sigma=sigma)
        # end for
    # end def

# end class
