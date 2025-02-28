#!/usr/bin/env python3

from simulation import Simulation
from structure import Structure

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


class NexusGenerator():
    '''A wrapper class for generating Nexus functions to produce and represent a PES.'''
    _func = None
    _args = None

    @property
    def func(self):
        return self._func
    # end def

    @func.setter
    def func(self, func):
        if callable(func):
            self._func = func
        else:
            raise TypeError("The PES function must be callable!")
        # end if
    # end

    @property
    def args(self):
        return self._args
    # end def

    @args.setter
    def args(self, args):
        if isinstance(args, dict):
            self._args = args
        elif args is None:
            self._args = {}
        else:
            raise TypeError("The PES arguments must be a dictionary")
        # end if
    # end

    def __init__(self, pes=None, pes_func=None, pes_args={}):
        '''A Nexus PES function is constructed from the job-generating function and arguments.'''
        if isinstance(pes, NexusGenerator):
            self.func = pes.func
            self.args = pes.args
        elif not callable(pes_func):
            # Allow construction with: NexusGenerator(pes_func, pes_args)
            self.func = pes
            self.args = pes_func
        else:
            self.func = pes_func
            self.args = pes_args
        # end if
    # end def

    def evaluate(self, structure, path, **kwargs):
        '''Return a list of Nexus jobs provided "structure" and "path" arguments.'''
        if not isinstance(structure, Structure):
            raise TypeError('Must use Structure class with NexusGenerator')
        # end if
        args = self.args.copy()
        args.update(kwargs)
        jobs = self.func(structure, path, **args)
        for job in jobs:
            if not isinstance(job, Simulation):
                raise TypeError("The nexus generator must return a list of Nexus Simulations!")
            # end if
        # end for
        return jobs
    # end def

# end class
