#!/usr/bin/env python3
"""Wrapper that augments a fitting result with derivative information."""

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

from numpy import array

from stalk.ls.DerivativeResult import DerivativeResult
from stalk.ls.FittingFunction import FittingFunction


class DerivativeFit(FittingFunction):
    def __init__(self, fit_func, args={}):
        if not isinstance(fit_func, FittingFunction):
            raise TypeError("fit_func must be a FittingFunction")
        super().__init__(func=fit_func.func, args=fit_func.args)
        self.fit_func = fit_func

    @property
    def kind(self):
        return f"{self.fit_func.kind}-deriv"

    def _eval_function(self, offsets, values):
        base_res = self.fit_func._eval_function(offsets, values)
        res = DerivativeResult(
            base_res.x0,
            base_res.y0,
            x0_err=base_res.x0_err,
            y0_err=base_res.y0_err,
            fit=base_res.fit,
            fraction=base_res.fraction,
        )
        self._attach_derivatives(res)
        return res

    def _attach_derivatives(self, res):
        x0 = res.x0

        if hasattr(res, "get_force"):
            res.d0 = res.get_force(x0)
        elif hasattr(res.fit, "derivative"):
            res.d0 = -res.fit.derivative()(x0)

        if hasattr(res, "get_hessian"):
            res.curvature = res.get_hessian(x0)
        elif hasattr(res.fit, "derivative"):
            res.curvature = res.fit.derivative(nu=2)(x0)

        return res

    def __eq__(self, other):
        return isinstance(other, DerivativeFit) and self.fit_func == other.fit_func
