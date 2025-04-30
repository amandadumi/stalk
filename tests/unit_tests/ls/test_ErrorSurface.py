#!/usr/bin/env python3

from numpy import abs

from pytest import raises
from stalk.ls.ErrorSurface import ErrorSurface
from stalk.util.util import match_to_tol

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# test ErrorSurface class
def test_ErrorSurface():

    # Utility function to simulate error monotonous with w, sigma parameters
    def errors(x, y):
        return abs(y) + x**2
    # end def

    es = ErrorSurface()
    assert len(es.Xs) == 1
    assert len(es.Ys) == 1
    assert es.X_res == 0.1
    assert es.Y_res == 0.1
    assert es.X_mat.shape[0] == 1
    assert es.X_mat.shape[1] == 1
    assert es.Y_mat.shape[0] == 1
    assert es.Y_mat.shape[1] == 1
    assert es.E_mat.shape[0] == 1
    assert es.E_mat.shape[1] == 1
    assert es.T_mat.shape[0] == 1
    assert es.T_mat.shape[1] == 1
    assert es.X_mat[0, 0] == 0.0
    assert es.Y_mat[0, 0] == 0.0
    assert es.E_mat[0, 0] == 0.0
    assert es.T_mat[0, 0]

    # Add x-point to end
    x0 = 1.0
    errs0 = [errors(x0, y) for y in es.Ys]
    es.insert_col(x0, errs0)
    assert match_to_tol(es.E_mat[:, 1], errs0)
    assert match_to_tol(es.Xs, [0.0, x0])
    assert match_to_tol(es.Ys, [0.0])
    # Add x-point to middle
    x1 = 0.5
    errs1 = [errors(x1, y) for y in es.Ys]
    es.insert_col(x1, errs1)
    assert match_to_tol(errs1, es.E_mat[:, 1])
    assert match_to_tol(es.Xs, [0.0, x1, x0])
    assert match_to_tol(es.Ys, [0.0])

    # Add y-point to end
    y2 = 1.0
    errs2 = [errors(x, y2) for x in es.Xs]
    es.insert_row(y2, errs2)
    assert match_to_tol(errs2, es.E_mat[-1, :])
    assert match_to_tol(es.Xs, [0.0, x1, x0])
    assert match_to_tol(es.Ys, [0.0, y2])
    # Add y-point to middle
    y3 = 0.4
    errs3 = [errors(x, y3) for x in es.Xs]
    es.insert_row(y3, errs3)
    assert match_to_tol(errs3, es.E_mat[1, :])
    assert match_to_tol(es.Xs, [0.0, x1, x0])
    assert match_to_tol(es.Ys, [0.0, y3, y2])

    # Check internal consistency
    for Y_row in es.Y_mat.T:
        assert match_to_tol(Y_row, es.Ys)
    # end for
    for X_col in es.X_mat:
        assert match_to_tol(X_col, es.Xs)
    # end for

    # Cannot add negative or inconsistent data
    with raises(ValueError):
        y = -0.1
        errs = [errors(x, y) for x in es.Xs]
        es.insert_col(y, errs)
    # end with
    with raises(ValueError):
        y = 0.1
        errs = [errors(x, y) for x in es.Xs][:-1]
        es.insert_col(y, errs)
    # end with
    with raises(ValueError):
        x = -0.1
        errs = [errors(x, y) for x in es.Ys]
        es.insert_row(x, errs)
    # end with
    with raises(ValueError):
        x = 0.1
        errs = [errors(x, y) for x in es.Ys][:-1]
        es.insert_row(x, errs)
    # end with

    # Test y-maximization
    # Cannot optimize to sub-zero epsilon
    with raises(ValueError):
        es.argmax_y(-1e-9)
    # end with

    # Small-epsilon limit is in the origin
    xi, yi = es.argmax_y(1e-9)
    assert xi == 0
    assert yi == 0

    # High-epsilon limit is in the opposite corner
    xi, yi = es.argmax_y(1e9)
    assert xi == len(es.Xs) - 1
    assert yi == len(es.Ys) - 1

    # Medium-epsilon is inside E_mat
    xi, yi = es.argmax_y(1.2)
    assert xi == 1
    assert yi == 1

    # Test surface evaluation
    new_x, new_y, val = es.evaluate_surface(1.0, 1.01)
    assert new_x is None  # x cannot be increased
    assert new_y == 1.1  # y is increased by Y_res
    assert val is None  # val is not computed when new y is requested

    new_x, new_y, val = es.evaluate_surface(0.2, 1.2)
    assert new_x == 0.25  # x is between 0 and 0.5
    assert new_y == 1.2  # new y is the requested y
    assert val is None  # val is not computed when new x and y are requested

    errsx = [errors(x, 1.05) for x in es.Xs]
    es.insert_row(1.05, errsx)
    errsy = [errors(0.52, y) for y in es.Ys]
    es.insert_col(0.52, errsy)
    new_x, new_y, val = es.evaluate_surface(0.51, 1.0)
    assert new_x is None  # no new x is requested
    assert new_y is None  # no new y is requested
    assert match_to_tol(val, errors(0.51, 1.0), 1e-2)

# end def
