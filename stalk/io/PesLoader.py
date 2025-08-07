#!/usr/bin/env python3

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

from stalk.params.PesResult import PesResult
from stalk.util.FunctionCaller import FunctionCaller


class PesLoader(FunctionCaller):

    def load(self, structure, sigma=0.0, **kwargs):
        args = self.args.copy()
        args.update(kwargs)
        res = self._load(structure, **args)
        # If a non-zero, artificial errorbar is requested, add it to result
        res.add_sigma(sigma)
        return res
    # end def

    def _load(self, structure, **kwargs):
        res = self.func(structure, **kwargs)
        if not isinstance(res, PesResult):
            raise AssertionError('The _load method must return a PesResult.')
        # end if
        return res
    # end def

# end class
