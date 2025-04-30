#!/usr/bin/env python3
'''ErrorSurface class for containing resampled fitting errors'''

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

from bisect import bisect
from scipy.interpolate import LinearNDInterpolator
from numpy import array, argsort, append, where, isscalar
from numpy import argmin


class ErrorSurface():
    _E_mat = None  # Matrix of total errors
    _X_mat = None  # X-mesh
    _Y_mat = None  # Y-mesh
    X_res = None  # fractional X resolution
    Y_res = None  # fractional Y resolution

    def __init__(
        self,
        X_res=0.1,
        Y_res=0.1
    ):
        # Initialize with zero error at the origin
        self._E_mat = array([[0.0]])
        self._X_mat = array([[0.0]])
        self._Y_mat = array([[0.0]])
        self.X_res = X_res
        self.Y_res = Y_res
    # end def

    @property
    def Xs(self):
        return self._X_mat[0]
    # end def

    @property
    def Ys(self):
        return self._Y_mat[:, 0]
    # end def

    @property
    def X_mat(self):
        return self._X_mat
    # end def

    @property
    def Y_mat(self):
        return self._Y_mat
    # end def

    @property
    def E_mat(self):
        return self._E_mat
    # end def

    @property
    def T_mat(self):
        return self._X_mat >= self._Y_mat
    # end def

    def insert_row(self, y, row):
        if len(row) != len(self.Xs):
            raise ValueError(f"Cannot add row with len={len(row)} to data with len={len(self.Xs)}")
        # end if
        if not isscalar(y) or (y < 0.0):
            raise ValueError("Cannot add y < 0.0.")
        # end if
        X_mat = append(self._X_mat, [self.Xs], axis=0)
        Y_mat = append(self._Y_mat, [len(self.Xs) * [y]], axis=0)
        E_mat = append(self._E_mat, [row], axis=0)
        idx = argsort(Y_mat[:, 0])
        self._X_mat = X_mat[idx]
        self._Y_mat = Y_mat[idx]
        self._E_mat = E_mat[idx]
    # end def

    def insert_col(self, x, col):
        if len(col) != len(self.Ys):
            raise ValueError(f"Cannot add row with len={len(col)} to data with len={len(self.Ys)}")
        # end if
        if not isscalar(x) or (x < 0.0):
            raise ValueError("Cannot add x < 0.0.")
        # end if
        X_mat = append(self._X_mat, array([len(self.Ys) * [x]]).T, axis=1)
        Y_mat = append(self._Y_mat, array([self.Ys]).T, axis=1)
        E_mat = append(self._E_mat, array([col]).T, axis=1)
        idx = argsort(X_mat[0])
        self._X_mat = X_mat[:, idx]
        self._Y_mat = Y_mat[:, idx]
        self._E_mat = E_mat[:, idx]
    # end def

    def argmax_y(self, epsilon):
        """Return indices to the highest point in E matrix that is lower than epsilon"""
        if (epsilon <= 0.0):
            raise ValueError("Cannot optimize to epsilon <= 0.0")
        # end if
        xi, yi = 0, 0
        X = self.X_mat[0]
        for i in range(len(self.E_mat), 0, -1):  # from high to low
            err = where((self.E_mat[i - 1] < epsilon) & (self.T_mat[i - 1]))
            if len(err[0]) > 0:
                yi = i - 1
                # xi = err[0][argmax(E[i - 1][err[0]])]
                # take the middle
                xi = err[0][argmin(
                    abs(X[err[0]] - (X[err[0][0]] + X[err[0][-1]]) / 2))]
                break
            # end if
        # end for
        return xi, yi
    # end def

    def evaluate_surface(self, x, y):
        new_x, new_y, val = None, None, None
        xi = bisect(self.Xs, x)
        yi = bisect(self.Ys, y)

        if xi < len(self.Xs):
            xi_prev = xi - 1
            rdx = (self.Xs[xi] - self.Xs[xi_prev]) / self.Xs[xi]
            # If relative difference is bigger than resolution, request new row
            if rdx > self.X_res:
                new_x = (self.Xs[xi_prev] + self.Xs[xi]) / 2
            # end if
        else:
            # revert to maximum value
            xi, xi_prev = xi - 1, xi - 1
        # end if
        if yi < len(self.Ys):
            yi_prev = yi - 1
            rdy = self.Ys[yi] - self.Ys[yi_prev] / self.Ys[yi]
            # If relative difference is bigger than resolution, request new col
            if rdy > self.Y_res:
                new_y = (self.Ys[yi_prev] + self.Ys[yi]) / 2
            # end if
        else:
            new_y = max([y, self.Ys[yi - 1] * (1 + self.Y_res)])
        # end if

        # If no new data is requested, return value by interpolation
        if new_x is None and new_y is None:
            pts = [
                [self.Xs[xi], self.Ys[yi]],
                [self.Xs[xi_prev], self.Ys[yi]],
                [self.Xs[xi], self.Ys[yi_prev]],
                [self.Xs[xi_prev], self.Ys[yi_prev]]
            ]
            vals = [
                self.E_mat[yi, xi],
                self.E_mat[yi, xi_prev],
                self.E_mat[yi_prev, xi],
                self.E_mat[yi_prev, xi_prev]
            ]
            val_int = LinearNDInterpolator(pts, vals)
            val = val_int([x, y])
        # end if
        return new_x, new_y, val
    # end def

# end class
