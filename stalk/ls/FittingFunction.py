#!/usr/bin/env python3
'''Class for fitting for curve minimum
'''

from numpy import random, array

from stalk.ls.FittingResult import FittingResult
from stalk.ls.LineSearchGrid import LineSearchGrid
from stalk.util import get_fraction_error

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


class FittingFunction():
    _result_type = FittingResult
    func = None
    args = {}

    def __init__(
        self,
        func=None,
        args={}
    ):
        if callable(func):
            self.func = func
            self.args = args
        else:
            raise ValueError("The fitting function must be callable.")
        # end if
    # end def

    def find_minimum(
        self,
        grid,
        sgn=1,
    ):
        if not isinstance(grid, LineSearchGrid):
            raise ValueError("Fitting function input must be inherited from LineSearchGrid.")
        # end if
        res = self._eval_function(grid.valid_offsets, grid.valid_values * sgn)
        return res
    # end def

    def _eval_function(self, offsets, values):
        x0, y0, fit = self.func(offsets, values, **self.args)
        return self._result_type(x0, y0, fit=fit)
    # end def

    def find_noisy_minimum(
        self,
        grid,
        sgn=1,
        N=200,
        Gs=None,
        fraction=0.025
    ):
        if not isinstance(grid, LineSearchGrid):
            raise ValueError("Fitting function input must be inherited from LineSearcGrid.")
        # end if
        result = self._eval_function(grid.valid_offsets, grid.valid_values * sgn)
        # If errors present, resample errorbars; if not, errors default to 0
        if grid.noisy:
            x0s, y0s = self.get_distribution(
                grid,
                sgn=sgn,
                N=N,
                Gs=Gs
            )
            result.x0_err = get_fraction_error(x0s - result.x0, fraction=fraction)[1]
            result.y0_err = get_fraction_error(y0s - result.y0, fraction=fraction)[1]
        # end if
        result.fraction = fraction
        return result
    # end def

    def get_distribution(
        self,
        grid,
        sgn=1,
        N=200,
        Gs=None,
    ):
        if not isinstance(grid, LineSearchGrid):
            raise ValueError("Fitting function input must be inherited from LineSearcGrid.")
        # end if
        if Gs is None:
            if isinstance(N, int) and N > 0:
                Gs = random.randn(N, len(grid.valid_errors))
            else:
                raise ValueError("Must provide either N > 0 or an array of G displacements")
            # end if
        # end if
        x0_distribution = []
        y0_distribution = []
        fit_distribution = []
        for G in Gs:
            values = sgn * grid.valid_values + grid.valid_errors * G
            result_this = self._eval_function(grid.valid_offsets, values)
            x0_distribution.append(result_this.x0)
            y0_distribution.append(result_this.y0)
            fit_distribution.append(result_this.fit)
        # end for
        return array(x0_distribution), array(y0_distribution)
    # end def

    def get_x0_distribution(self, grid, **kwargs):
        return self.get_distribution(grid, **kwargs)[0]
    # end def

    def get_y0_distribution(self, grid, **kwargs):
        return self.get_distribution(grid, **kwargs)[1]
    # end def

# end class
