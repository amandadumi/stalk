#!/usr/bin/env python3
'''Generic classes for 1-dimensional line-searches
'''

from numpy import linspace
from stalk.ls.FittingFunction import FittingFunction
from stalk.ls.FittingResult import FittingResult
from stalk.ls.LineSearchGrid import LineSearchGrid
from stalk.util import get_min_params

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# Class for line-search along direction in abstract context
class LineSearchBase(LineSearchGrid):

    fit_func: FittingFunction
    fit_res: FittingResult | None
    sgn: int = 1

    def __init__(
        self,
        offsets=None,
        values=None,
        errors=None,
        fraction=0.025,
        sgn=1,
        fit_kind='pf3',
        fit_func=None,
        fit_args={},
        N=200,
        Gs=None,
    ):
        LineSearchGrid.__init__(self, offsets)
        self.sgn = sgn
        self.set_fit_func(
            fit_func=fit_func,
            fit_args=fit_args,
            fit_kind=fit_kind
        )
        self.fit_res = None
        if values is not None:
            self.values = values
            if errors is not None:
                self.errors = errors
            # end if
            self._search_and_store(N=N, Gs=Gs, fraction=fraction)
        # end if
    # end def

    def set_fit_func(self, **kwargs):
        self.fit_func = self.get_fit_func(**kwargs)
    # end def

    def get_fit_func(
        self,
        fit_func=None,
        fit_args={},
        fit_kind='pf3'
    ):
        if isinstance(fit_func, FittingFunction):
            # If fitting function is good as is
            return fit_func
        elif callable(fit_func):
            # Try to initiate
            return FittingFunction(fit_func, args=fit_args)
        else:
            # Try to infer from fit_kind
            if hasattr(fit_kind, "__iter__") and 'pf' in fit_kind:
                return FittingFunction(
                    get_min_params,
                    args={'pfn': int(fit_kind[2:])}
                )
            else:
                raise TypeError('Fit kind {} not recognized'.format(fit_kind))
            # end if
        # end fi
    # end def

    def _search_and_store(self, N=200, Gs=None, fraction=0.025):
        """Perform line-search with the preset values and settings, saving the result to self."""
        self.fit_res = self.search_with_error(N=200, Gs=None, fraction=fraction)
    # end def

    def search_with_error(
        self,
        # The following kwargs can be used to override presets
        sgn=None,
        grid=None,
        fraction=None,
        fit_func=None,
        N=200,
        Gs=None,
    ):
        # It is possible to override grid to get alternative results. Used in resampling
        # alternative errorbars
        grid = grid if isinstance(grid, LineSearchGrid) else self
        sgn = sgn if isinstance(sgn, int) else self.sgn
        fit_func = fit_func if isinstance(sgn, FittingFunction) else self.fit_func
        fraction = fraction if isinstance(fraction, float) else self.fraction

        return fit_func.find_noisy_minimum(grid, N=N, Gs=Gs, fraction=fraction)
    # end def

    def search(
        self,
        # The following kwargs can be used to override presets
        sgn=None,
        grid=None,
        fraction=None,
        fit_func=None
    ):
        # It is possible to override grid to get alternative results. Used in resampling
        # alternative errorbars
        grid = grid if isinstance(grid, LineSearchGrid) else self
        sgn = sgn if isinstance(sgn, int) else self.sgn
        fit_func = fit_func if isinstance(fit_func, FittingFunction) else self.fit_func
        fraction = fraction if isinstance(fraction, float) else self.fraction

        return fit_func.find_minimum(grid, sgn=sgn)
    # end def

    def reset_search(self, x0=0.0, y0=0.0):
        self.fit_res.x0 = x0
        self.fit_res.x0_err = 0.0
        self.fit_res.y0 = y0
        self.fit_res.y0_err = 0.0
        return self
    # end def

    @property
    def fraction(self):
        return None if self.fit_res is None else self.fit_res.fraction
    # end def

    @property
    def x0(self):
        return None if self.fit_res is None else self.fit_res.x0
    # end def

    @property
    def x0_err(self):
        return None if self.fit_res is None else self.fit_res.x0_err
    # end def

    @property
    def y0(self):
        return None if self.fit_res is None else self.fit_res.y0
    # end def

    @property
    def y0_err(self):
        return None if self.fit_res is None else self.fit_res.y0_err
    # end def

    def _make_offsets_R(self, R, M):
        if R < 1e-6:
            raise ValueError("R must be larger than 1e-6")
        # end if
        offsets = linspace(-R, R, M)
        return offsets
    # end def

    def __str__(self):
        string = self.__class__.__name__
        string += str(self.offsets)
        if self.fit_res.x0 is None:
            string += '\n  x0: not set'
        else:
            x0_err = '' if self.fit_res.x0_err is None else ' +/- {: <8f}'.format(
                self.fit_res.x0_err)
            string += '\n  x0: {: <8f} {:s}'.format(self.fit_res.x0, x0_err)
        # end if
        if self.fit_res.y0 is None:
            string += '\n  y0: not set'
        else:
            y0_err = '' if self.fit_res.y0_err is None else ' +/- {: <8f}'.format(
                self.fit_res.y0_err)
            string += '\n  y0: {: <8f} {:s}'.format(self.fit_res.y0, y0_err)
        # end if
        return string
    # end def


# end class
