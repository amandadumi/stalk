#!/usr/bin/env python3

from stalk.io.GeometryLoader import GeometryLoader
from stalk.nexus.NexusStructure import NexusStructure
from stalk.util.FunctionCaller import FunctionCaller
from stalk.util.util import directorize
from nexus import run_project

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


class NexusGeometry(FunctionCaller):
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
        # Geometry jobs can use the same format as PesFunction
        super().__init__(func, args)
        if isinstance(loader, GeometryLoader):
            self.loader = loader
        else:
            self.loader = GeometryLoader(load_func, load_args)
        # end if
    # end def

    def relax(
        self,
        structure: NexusStructure,
        path='relax',
        **kwargs,
    ):
        # Override with kwargs
        eval_args = self.args.copy()
        eval_args.update(**kwargs)
        # Generate relaxation jobs
        jobs = self.func(
            structure.get_nexus_structure(),
            directorize(path),
            **eval_args
        )
        structure.jobs = jobs
        # Run project
        run_project(jobs)

        # Load results and update the structure
        res = self.loader.load(path)
        structure.set_position(res.get_pos(), res.get_axes())
    # end def

# end class
