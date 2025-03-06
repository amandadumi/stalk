#!/usr/bin/env python3
"""NexusHessian class inherits ParameterHessian with Nexus functions.
"""

from stalk.nexus.NexusStructure import NexusStructure
from stalk.util import directorize
from stalk.io.PesLoader import PesLoader
from stalk.params.ParameterHessian import ParameterHessian
from stalk.nexus.NexusGenerator import NexusGenerator

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


class NexusHessian(ParameterHessian):
    """NexusHessian class inherits ParameterHessian with Nexus functions.
    """

    def _evaluate_energies(
        self,
        structure_list: list[NexusStructure],
        path='fdiff',
        pes=None,
        pes_func=None,
        pes_args={},
        loader=None,
        load_func=None,
        load_args={},
        **kwargs,
    ):
        # Generate jobs
        if not isinstance(pes, NexusGenerator):
            # Checks are made in the wrapper class
            pes = NexusGenerator(pes_func, pes_args)
        # end if
        jobs = []
        for s in structure_list:
            dir = '{}{}'.format(directorize(path), s.label)
            # Make a copy structure for job generation
            jobs += pes.evaluate(s.get_nexus_structure(), dir)
        # end for
        from nexus import run_project
        run_project(jobs)

        loader = PesLoader(loader, load_func, load_args)
        for s in structure_list:
            dir = '{}{}'.format(directorize(path), s.label)
            E = loader.load(path=dir).get_value()
            s.value = E
        # end for
    # end def

# end class
