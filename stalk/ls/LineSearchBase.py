#!/usr/bin/env python3
'''Generic classes for 1-dimensional line-searches
'''

from numpy import array, random, polyval, polyder

from stalk.ls.LineSearchGrid import LineSearchGrid
from stalk.util import get_min_params, get_fraction_error

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# Class for line-search along direction in abstract context
class LineSearchBase(LineSearchGrid):

    fraction = None
    fit_kind = None
    func = None
    func_p = None
    x0 = None
    x0_err = None
    y0 = None
    y0_err = None
    sgn = 1
    fit = None

    def __init__(
        self,
        grid=None,
        values=None,
        errors=None,
        fraction=0.025,
        sgn=1,
        **kwargs,
    ):
        LineSearchGrid.__init__(self, grid)
        self.fraction = fraction
        self.set_func(**kwargs)
        self.sgn = sgn
        if values is not None:
            self.values = values
            if errors is not None:
                self.errors = errors
            # end if
            self.search()
        # end if
    # end def

    def set_func(
        self,
        fit_kind='pf3',
        **kwargs
    ):
        self.func, self.func_p = self._get_func(fit_kind)
        self.fit_kind = fit_kind
    # end def

    def get_func(self, fit_kind=None):
        if fit_kind is None:
            return self.func, self.func_p
        else:
            return self._get_func(fit_kind)
        # end if
    # end def

    def _get_func(self, fit_kind):
        if 'pf' in fit_kind:
            func = self._pf_search
            func_p = int(fit_kind[2:])
        else:
            raise ('Fit kind {} not recognized'.format(fit_kind))
        # end if
        return func, func_p
    # end def

    def search(self, **kwargs):
        """Perform line-search with the preset values and settings, saving the result to self."""
        res = self._search_with_error(
            self.valid_grid,
            self.valid_values * self.sgn,
            self.valid_errors,
            fit_kind=self.fit_kind,
            fraction=self.fraction,
            **kwargs)
        self.x0 = res[0]
        self.y0 = res[2]
        self.x0_err = res[1]
        self.y0_err = res[3]
        self.fit = res[4]
        self.analyzed = True
    # end def

    def _search(
        self,
        grid,
        values,
        fit_kind=None,
        **kwargs,
    ):
        func, func_p = self.get_func(fit_kind)
        return self._search_one(grid, values, func, func_p, **kwargs)
    # end def

    def _search_one(
        self,
        grid,
        values,
        func,
        func_p=None,
        **kwargs,
    ):
        return func(grid, values, func_p)  # x0, y0, fit
    # end def

    def _search_with_error(
        self,
        grid,
        values,
        errors,
        fraction=None,
        fit_kind=None,
        **kwargs,
    ):
        func, func_p = self.get_func(fit_kind)
        x0, y0, fit = self._search_one(grid, values, func, func_p, **kwargs)
        fraction = fraction if fraction is not None else self.fraction
        # resample for errorbars
        if errors is not None:
            x0s, y0s = self._get_distribution(
                grid, values, errors, func=func, func_p=func_p, **kwargs)
            ave, x0_err = get_fraction_error(x0s - x0, fraction=fraction)
            ave, y0_err = get_fraction_error(y0s - y0, fraction=fraction)
        else:
            x0_err, y0_err = 0.0, 0.0
        # end if
        return x0, x0_err, y0, y0_err, fit
    # end def

    def _pf_search(
        self,
        grid,
        values,
        pfn,
        **kwargs,
    ):
        return get_min_params(grid, values, pfn, **kwargs)
    # end def

    def reset(self):
        self.x0, self.x0_err, self.y0, self.y0_err, self.fit = None, None, None, None, None
    # end def

    def get_x0(self, err=True):
        assert self.x0 is not None, 'x0 must be computed first'
        if err:
            return self.x0, self.x0_err
        else:
            return self.x0
        # end if
    # end def

    def get_y0(self, err=True):
        assert self.y0 is not None, 'y0 must be computed first'
        if err:
            return self.y0, self.y0_err
        else:
            return self.y0
        # end if
    # end def

    def get_hessian(self, x=None):
        x = x if x is not None else self.x0
        if self.fit is None:
            return None
        else:
            return polyval(polyder(polyder(self.fit)), x)
        # end if
    # end def

    def get_force(self, x=None):
        x = x if x is not None else 0.0
        if self.fit is None:
            return None
        else:
            return -polyval(polyder(self.fit), x)
        # end if
    # end def

    def get_distribution(self, fit_kind=None, **kwargs):
        func, func_p = self.get_func(fit_kind)
        return self._get_distribution(
            self.valid_grid,
            self.valid_values * self.sgn,
            self.valid_errors,
            func=func,
            func_p=func_p,
            **kwargs
        )
    # end def

    def get_x0_distribution(self, errors=None, N=100, **kwargs):
        if errors is None:
            return array(N * [self.get_x0(err=False)])
        # end if
        return self.get_distribution(errors=errors, **kwargs)[0]
    # end def

    def get_y0_distribution(self, errors=None, N=100, **kwargs):
        if errors is None:
            return array(N * [self.get_y0(err=False)])
        # end if
        return self.get_distribution(errors=errors, **kwargs)[1]
    # end def

    # TODO: refactor to support generic fitting functions
    def val_data(self, xdata):
        if self.fit is None:
            return None
        else:
            return polyval(self.fit, xdata)
        # end def
    # end def

    # must have func, func_p in **kwargs
    def _get_distribution(self, grid, values, errors, Gs=None, N=100, **kwargs):
        if Gs is None:
            Gs = random.randn(N, len(errors))
        # end if
        x0s, y0s, pfs = [], [], []
        for G in Gs:
            x0, y0, pf = self._search_one(grid, values + errors * G, **kwargs)
            x0s.append(x0)
            y0s.append(y0)
            pfs.append(pf)
        # end for
        return array(x0s, dtype=float), array(y0s, dtype=float)
    # end def

    def __str__(self):
        string = self.__class__.__name__
        if self.fit_kind is not None:
            string += '\n  fit_kind: {:s}'.format(self.fit_kind)
        # end if
        string += str(self.grid)
        if self.x0 is None:
            string += '\n  x0: not set'
        else:
            x0_err = '' if self.x0_err is None else ' +/- {: <8f}'.format(
                self.x0_err)
            string += '\n  x0: {: <8f} {:s}'.format(self.x0, x0_err)
        # end if
        if self.y0 is None:
            string += '\n  y0: not set'
        else:
            y0_err = '' if self.y0_err is None else ' +/- {: <8f}'.format(
                self.y0_err)
            string += '\n  y0: {: <8f} {:s}'.format(self.y0, y0_err)
        # end if
        return string
    # end def


# end class
