#!/usr/bin/env python3
"""Generic class for curve minimum and error bars"""

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


class DerivativeResult:
    # TODO: this implementation is a stub

    fraction = None
    x0 = None
    x0_err = 0.0
    y0 = None
    y0_err = 0.0
    fit = None
    x0_deriv = None  # optional derivative info at minimum
    y0_deriv = None

    def __init__(self, x0, y0, x0_err=0.0, y0_err=0.0, fit=None, fraction=0.025):
        self.fraction = fraction
        self.x0 = x0
        self.y0 = y0
        self.x0_err = x0_err
        self.y0_err = y0_err
        self.fit = fit

    # end def

    def get_force(self, x):
        if hasattr(self.fit, "derivative"):
            return -self.fit.derivative()(x)
        raise NotImplementedError

    def get_hessian(self, x):
        if hasattr(self.fit, "derivative"):
            return self.fit.derivative(nu=2)(x)
        raise NotImplementedError

    def get_values(self, x):
        if hasattr(self.fit, "__call__"):
            return self.fit(x)
        raise NotImplementedError
# end class
