#!/usr/bin/env python3

from numpy import linspace

from stalk.ls.LineSearchGrid import LineSearchGrid
from stalk.ls.FittingResult import FittingResult


def generate_exact_pf2(x_offset, y_offset, N, error=None):
    if N % 2 == 0:
        R = N / 2.0
    else:
        R = (N - 1) / 2.0
    # end if
    offsets = linspace(-R, R, N)
    offsets = offsets
    values = (offsets - x_offset)**2 + y_offset

    grid = LineSearchGrid(offsets)
    grid.values = values
    if error is not None:
        grid.errors = len(grid) * [error]
    # end if

    # TODO: x0_err, y0_err not considered if errors > 0
    # TODO: exact fit not considered
    ref = FittingResult(x0=x_offset, y0=y_offset)

    return grid, ref
# end def


def generate_exact_pf3(x_offset, y_offset, N, c=0.01, error=None):
    if N % 2 == 0:
        R = N / 2.0
    else:
        R = (N - 1) / 2.0
    # end if
    offsets = linspace(-R, R, N)
    offsets = offsets
    values = c * (offsets - x_offset)**3 + (offsets - x_offset)**2 + y_offset

    grid = LineSearchGrid(offsets)
    grid.values = values
    if error is not None:
        grid.errors = len(grid) * [error]
    # end if

    ref = FittingResult(x0=x_offset, y0=y_offset)

    return grid, ref
# end def
