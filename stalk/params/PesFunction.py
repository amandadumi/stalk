#!/usr/bin/env python3

from stalk.params.PesResult import PesResult

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


class PesFunction():
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
        '''A PES function is constructed from the job-generating function and arguments.'''
        if isinstance(pes, PesFunction):
            self.func = pes.func
            self.args = pes.args
        elif callable(pes):
            # Allow construction with: PesFunction(pes_func, pes_args)
            self.func = pes
            self.args = pes_func
        else:
            self.func = pes_func
            self.args = pes_args
        # end if
    # end def

    def evaluate(self, structure, **kwargs):
        eval_args = self.args.copy()
        # Override with kwargs
        eval_args.update(**kwargs)
        value, error = self.func(structure, **eval_args)
        return PesResult(value, error)
    # end def

# end class
