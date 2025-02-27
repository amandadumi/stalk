#!/usr/bin/env python

from pytest import raises

from stalk.ls.FittingFunction import FittingFunction
from stalk.ls.FittingResult import FittingResult
from stalk.ls.LineSearchBase import LineSearchBase
from stalk.util.util import get_min_params, match_to_tol

from ..assets.fitting_pf2 import generate_exact_pf2

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# Test LineSearchBase class
def test_LineSearchBase():

    # test empty/defaults
    ls = LineSearchBase()
    assert ls.sgn == 1
    assert ls.x0 is None
    assert ls.x0_err is None
    assert ls.y0 is None
    assert ls.y0_err is None
    assert ls.fraction is None
    assert isinstance(ls.fit_func, FittingFunction)
    assert ls.fit_func.func is get_min_params
    assert ls.fit_func.args['pfn'] == 3

    # test initialization of different fit functions
    fit0 = FittingFunction(get_min_params, {'test': 0})
    ls.set_fit_func(fit_func=fit0)
    assert isinstance(ls.fit_func, FittingFunction)
    assert ls.fit_func is fit0
    assert ls.fit_func.args['test'] == 0

    ls.set_fit_func(fit_func=get_min_params, fit_args={'test': 1})
    assert isinstance(ls.fit_func, FittingFunction)
    assert ls.fit_func.func is get_min_params
    assert ls.fit_func.args['test'] == 1

    ls.set_fit_func(fit_kind='pf6')
    assert isinstance(ls.fit_func, FittingFunction)
    assert ls.fit_func.func is get_min_params
    assert ls.fit_func.args['pfn'] == 6

    with raises(TypeError):
        ls.set_fit_func(fit_kind='error')
    # end with

    # Test initialization with grid and values, no noise
    grid, ref = generate_exact_pf2(1.23, 2.34, N=5)
    ls_val = LineSearchBase(offsets=grid.offsets, values=grid.values, fit_kind='pf2')
    assert ls_val.fit_res.analyzed
    assert match_to_tol(ls_val.x0, ref.x0)
    assert match_to_tol(ls_val.y0, ref.y0)
    assert ls_val.x0_err == 0.0
    assert ls_val.y0_err == 0.0

    # Test initialization with grid, values and noise
    fraction = 0.1
    grid, ref = generate_exact_pf2(1.23, 2.34, N=5, error=0.1)
    ls_noisy = LineSearchBase(
        offsets=grid.offsets,
        values=grid.values,
        errors=grid.errors,
        fraction=fraction,
        fit_kind='pf2',
        N=10
    )
    assert ls_noisy.fit_res.analyzed
    assert match_to_tol(ls_noisy.x0, ref.x0)
    assert match_to_tol(ls_noisy.y0, ref.y0)
    assert ls_noisy.x0_err > 0.0
    assert ls_noisy.y0_err > 0.0
    assert ls_noisy.fraction == fraction
    # Test search
    # TODO: add coverage
    res = ls_noisy.search()
    assert isinstance(res, FittingResult)
    assert match_to_tol(res.x0, ref.x0)
    assert match_to_tol(res.y0, ref.y0)
    assert res.x0_err == 0.0
    assert res.y0_err == 0.0
    # Test search with error
    # TODO: add coverage
    res = ls_noisy.search_with_error()
    assert isinstance(res, FittingResult)
    assert match_to_tol(res.x0, ref.x0)
    assert match_to_tol(res.y0, ref.y0)
    assert res.x0_err > 0.0
    assert res.y0_err > 0.0

# end def
