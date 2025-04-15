#!/usr/bin/env python3

from pytest import raises
from stalk.util.FunctionCaller import FunctionCaller

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


def test_FunctionCaller():

    # Minimal function for testing vs callable
    def func(x, arg=0):
        return x + arg, x + 1
    # end def

    # Test degraded
    with raises(TypeError):
        # Cannot init empty
        FunctionCaller()
    # end with
    with raises(TypeError):
        # func must be callable
        FunctionCaller(func=[])
    # end with
    with raises(TypeError):
        # args must be dict
        FunctionCaller(pes_func=func, pes_args=[])
    # end with

    # Test nominal
    args = {"arg": 5}
    pf = FunctionCaller(func, args=args)
    assert pf.args is args
    assert pf.func is func

    # Test copy constructor
    pf_copy = FunctionCaller(pf)
    assert pf_copy.args is args
    assert pf_copy.func is func

# end def
