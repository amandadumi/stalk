#!/usr/bin/env python3
'''TargetLineSearch classes for the assessment and evaluation of fitting errors
'''

from numpy import array, isscalar, nan, where
from scipy.interpolate import CubicSpline, PchipInterpolator

from stalk.ls.FittingResult import FittingResult
from stalk.ls.LineSearchBase import LineSearchBase
from stalk.ls.LineSearchGrid import LineSearchGrid

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# Class for line-search with resampling and bias assessment against target
class TargetLineSearchBase(LineSearchBase):
    _target_interp = None  # Interpolant
    target_fit = FittingResult  # Target fit
    bias_mix = None
    bias_order = 1

    def __init__(
        self,
        offsets=None,
        values=None,
        errors=None,
        bias_mix=0.0,
        bias_order=1,
        interpolate_kind='cubic',
        **ls_args,
        # fraction=0.025, sgn=1, fit_kind='pf3', fit_func=None, fit_args={}, N=200, Gs=None
    ):
        LineSearchBase.__init__(
            self,
            offsets=offsets,
            values=values,
            errors=errors,
            **ls_args
        )
        self.bias_mix = bias_mix
        self.bias_order = bias_order
        self.target_fit = FittingResult(0.0, 0.0)
        if self.valid:
            self.reset_interpolation(interpolate_kind=interpolate_kind)
        # end if
    # end def

    @property
    def valid_target(self):
        return self._target_interp is not None
    # end def

    def reset_interpolation(
        self,
        interpolate_kind='cubic'
    ):
        if not self.valid:
            raise AssertionError("Must provide values before interpolation")
        # end if
        if interpolate_kind == 'pchip':
            self._target_interp = PchipInterpolator(
                self.valid_offsets,
                self.valid_values,
                extrapolate=True
            )
        elif interpolate_kind == 'cubic':
            self._target_interp = CubicSpline(
                self.valid_offsets,
                self.valid_values,
                extrapolate=True
            )
        else:
            raise ValueError("Could not recognize interpolate kind" + str(interpolate_kind))
        # end if
    # end def

    def evaluate_target(self, offsets):
        if isscalar(offsets):
            if self.valid_target:
                return self._evaluate_target_point(offsets)
            else:
                return nan
            # end if
        else:
            if self.valid_target:
                return array([self._evaluate_target_point(offset) for offset in offsets])
            else:
                return array(len(offsets) * [nan])
            # end if
        # end if
    # end def

    def _evaluate_target_point(self, offset):
        if offset < self._target_interp.x.min() or offset > self._target_interp.x.max():
            return nan
        else:
            return self._target_interp(offset)
        # end if
    # end def

    def compute_bias(
        self,
        grid: LineSearchGrid,
        bias_mix=None,
        bias_order=None,
        **ls_args  # sgn, fit_func
    ):
        if not self.valid_target:
            return nan
        # end if
        bias_mix = bias_mix if bias_mix is not None else self.bias_mix
        bias_order = bias_order if bias_order is not None else self.bias_order
        if bias_order <= 0:
            raise ValueError("bias_order must be 1 or more.")
        # end if
        bias_x, bias_y = self._compute_xy_bias(
            grid,
            bias_order=bias_order,
            x0_ref=self.target_fit.x0,
            y0_ref=self.target_fit.y0,
            **ls_args
        )
        bias_tot = abs(bias_x) + bias_mix * abs(bias_y)
        return bias_tot
    # end def

    def _compute_xy_bias(
        self,
        grid: LineSearchGrid,
        bias_order=1,
        x0_ref=0.0,
        y0_ref=0.0,
        **ls_args  # sgn, fit_func
    ):
        x0 = x0_ref
        x_min = self._target_interp.x.min()
        x_max = self._target_interp.x.max()
        # Repeat search 'bias_order' times to simulate how bias is self-induced
        for i in range(bias_order):
            offsets = grid.offsets + x0
            offsets[where(offsets < x_min)] = x_min
            offsets[where(offsets > x_max)] = x_max
            values = self.evaluate_target(offsets)
            res = self.search(
                grid=LineSearchGrid(offsets, values=values),
                **ls_args
            )
            x0 = res.x0
            y0 = res.y0
        # end for
        # Offset bias
        bias_x = x0 - x0_ref
        # Value bias
        bias_y = y0 - y0_ref
        return bias_x, bias_y
    # end def

    def compute_errorbar(
        self,
        grid: LineSearchGrid,
        **ls_args  # sgn, fraction, fit_func, N, Gs
    ):
        if not self.valid_target:
            return nan, nan
        # end if
        offsets = grid.valid_offsets
        values = self.evaluate_target(offsets)
        if values is not None:
            res = self.search_with_error(
                grid=LineSearchGrid(offsets, values=values, errors=grid.valid_errors),
                **ls_args
            )
            return res.x0_err, res.y0_err
        # end if
    # end def

    def compute_error(
        self,
        grid: LineSearchGrid,
        bias_mix=None,
        bias_order=None,
        N=200,
        Gs=None,
        fraction=None,
        **ls_args  # sgn, fraction, fit_func
    ):
        bias = self.compute_bias(
            grid,
            bias_mix=bias_mix,
            bias_order=bias_order,
            **ls_args
        )
        errorbar_x, errorbar_y = self.compute_errorbar(
            grid,
            N=N,
            Gs=Gs,
            fraction=fraction,
            **ls_args
        )
        return bias + errorbar_x
    # end def

    def bracket_target_bias(
        self,
        bracket_fraction=0.5,
        M=7,
        max_iter=10,
        **ls_args  # sgn=1, fit_kind='pf3', fit_func=None, fit_args={}
    ):
        if bracket_fraction <= 0 or bracket_fraction >= 1.0:
            raise ValueError("Must be 0 < bracket_fraction < 1")
        # end if
        if not self.valid_target:
            raise AssertionError("Target data is not valid yet.")
        # end if
        self.set_fit_func(**ls_args)

        R_this = self.valid_R_max
        bias_x = self.fit_res.x0
        for i in range(max_iter):
            offsets = self._make_offsets_R(R_this, M) + bias_x
            bias_x, bias_y = self._compute_xy_bias(
                LineSearchGrid(offsets),
                bias_order=1
            )
            R_this *= bracket_fraction
        # end for
        self.target_fit.x0 = bias_x
        self.target_fit.y0 = bias_y
    # end def

    def __str__(self):
        string = LineSearchBase.__str__(self)
        string += '\n  bias_mix: {:<4f}'.format(self.bias_mix)
        return string
    # end def

# end class
