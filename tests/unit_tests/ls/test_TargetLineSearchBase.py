#!/usr/bin/env python

from numpy import isnan, linspace
from scipy.interpolate import PchipInterpolator, CubicSpline
from pytest import raises

from stalk.ls.FittingFunction import FittingFunction
from stalk.ls.LineSearchGrid import LineSearchGrid
from stalk.util import match_to_tol
from stalk.ls import TargetLineSearchBase
from stalk.util.util import get_min_params

from ..assets.fitting_pf2 import generate_exact_pf2, generate_exact_pf3

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# test TargetLineSearchBase class
def test_TargetLineSearchBase():

    # Test init with zero input
    tls = TargetLineSearchBase()
    assert tls.bias_mix == 0.0
    assert tls.bias_order == 1
    assert tls.target_fit.x0 == 0.0
    assert tls.target_fit.y0 == 0.0
    assert tls.target_fit.x0_err == 0.0
    assert tls.target_fit.y0_err == 0.0
    assert not tls.valid_target

    # Test init with only offsets input
    offsets = linspace(-0.5, 0.5, 21)
    tls = TargetLineSearchBase(offsets=offsets)
    assert not tls.valid_target
    assert not tls.valid
    # Cannot reset interpolation while empty
    with raises(AssertionError):
        tls.reset_interpolation()
    # end with
    # Target evaluation results in None
    assert isnan(tls.evaluate_target([0.0]))
    assert isnan(tls.evaluate_target(0.0))
    # Bias assessment results in nan
    assert isnan(tls.compute_bias([LineSearchGrid()]))
    # Errorbar assessment results in nan, nan
    assert isnan(tls.compute_errorbar(LineSearchGrid())[0])
    assert isnan(tls.compute_errorbar(LineSearchGrid())[1])
    # Total error assessment results in nan
    assert isnan(tls.compute_error(LineSearchGrid()))
    # Cannot extrapolate error
    with raises(AssertionError):
        tls.bracket_target_bias()
    # end with

    # Test nominal init with reference potential data
    grid, ref = generate_exact_pf2(1.23, 2.34, N=21, error=0.1)
    bias_mix = 0.1
    interpolate_kind = 'pchip'
    fraction = 0.05
    fit_kind = 'pf2'
    N = 20
    tls = TargetLineSearchBase(
        offsets=grid.offsets,
        values=grid.values,
        bias_mix=bias_mix,
        interpolate_kind=interpolate_kind,
        fraction=fraction,
        fit_kind=fit_kind,
        N=N
    )
    assert tls.bias_mix == bias_mix
    # Test interpolant
    assert isinstance(tls._target_interp, PchipInterpolator)
    # Test Fitting function
    assert tls.fit_res.fraction == fraction
    assert isinstance(tls.fit_func, FittingFunction)
    assert tls.fit_func.func is get_min_params
    assert tls.fit_func.args['pfn'] == 2

    # Test evaluation
    match_to_tol(tls.evaluate_target(grid.offsets), grid.values)
    assert tls.evaluate_target(grid.offsets[4]) == grid.values[4]
    assert isnan(tls.evaluate_target(grid.offsets[0] - 1e-6))
    assert isnan(tls.evaluate_target(grid.offsets[-1] + 1e-6))
    # Test assessment of bias
    with raises(ValueError):
        tls.compute_bias(grid, bias_order=0)
    # end with
    bias = tls.compute_bias(grid)
    match_to_tol(bias, ref.x0 + bias_mix * ref.y0)
    # Test assessment of errorbars
    errorbar_x, errorbar_y = tls.compute_errorbar(grid)
    assert errorbar_x > 0.0
    assert errorbar_y > 0.0
    # Test assessment of total error
    error = tls.compute_error(grid)
    match_to_tol(error, bias + errorbar_x)

    # Test reset interpolation
    tls.reset_interpolation(interpolate_kind='cubic')
    assert isinstance(tls._target_interp, CubicSpline)
    with raises(ValueError):
        tls.reset_interpolation('error')
    # end with
    match_to_tol(tls.evaluate_target(grid.offsets), grid.values)
    assert tls.evaluate_target(grid.offsets[4]) == grid.values[4]
    assert isnan(tls.evaluate_target(grid.offsets[0] - 1e-6))
    assert isnan(tls.evaluate_target(grid.offsets[-1] + 1e-6))

    # Test bracket target bias
    biased_grid, ref = generate_exact_pf3(1.23, 262.3, N=11)
    tls = TargetLineSearchBase(
        offsets=biased_grid.offsets,
        values=biased_grid.values,
        fit_kind='pf2'
    )
    with raises(ValueError):
        tls.bracket_target_bias(bracket_fraction=0.0)
    # end with
    with raises(ValueError):
        tls.bracket_target_bias(bracket_fraction=1.0)
    # end with
    # At first the target biases are nonzero
    assert tls.target_fit.x0 != ref.x0
    assert tls.target_fit.y0 != ref.y0
    tls.bracket_target_bias(fit_kind='pf2')
    # After bracketing, the biases should be zero
    match_to_tol(tls.target_fit.x0, ref.x0)
    match_to_tol(tls.target_fit.y0, ref.y0)

# end def
