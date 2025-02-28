#!/usr/bin/env python3
'''Generic base class for sampling a PES in iterative batches
'''

from nexus import run_project

from stalk.io.PesLoader import PesLoader
from stalk.nexus.NexusGenerator import NexusGenerator
from stalk.nexus.NexusStructure import NexusStructure
from stalk.pls.PesSampler import PesSampler
from stalk.util.util import directorize

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# A class for managing sampling of the PES
class NexusSampler(PesSampler):

    def __init__(
        self,
        path=None,
        pes=None,
        pes_func=None,
        pes_args={},
        loader=None,
        load_func=None,
        load_args={},
    ):
        super().__init__(path=path)
        self.pes = NexusGenerator(pes, pes_func, pes_args)
        self.loader = PesLoader(loader, load_func, load_args)
    # end def

    # Override evaluation function to support parallel job submission and analysis
    def _evaluate_energies(
        self,
        structures: list[NexusStructure],
        sigmas: list[float],
        add_sigma=False
    ):
        jobs = []
        for structure, sigma in zip(structures, sigmas):
            dir = '{}{}'.format(directorize(self.path), structure.label)
            # Make a copy structure for job generation
            jobs += self.pes.evaluate(
                structure.get_nexus_structure(),
                dir,
                sigma=sigma
            )
        # end for
        run_project(jobs)

        for structure, sigma in zip(structures, sigmas):
            dir = '{}{}'.format(directorize(self.path), structure.label)
            res = self.loader.load(path=dir)
            if add_sigma:
                res.add_sigma(sigma)
            # end if
            structure.value = res.get_value()
            structure.error = res.get_error()
        # end for
    # end def

# end class
