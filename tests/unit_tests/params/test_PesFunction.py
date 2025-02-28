#!/usr/bin/env python3

from pytest import raises
from stalk.params.PesFunction import PesFunction
from stalk.params.PesResult import PesResult

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


def test_PesFunction():

    # Minimal function for testing vs callable
    def pes_func(x, arg=0):
        return x + arg, x + 1
    # end def

    # Alternative function
    def pes_func_alt(x, arg=0):
        return x + arg, x + 2
    # end def

    # Test degraded
    with raises(TypeError):
        # Cannot init empty
        PesFunction()
    # end with
    # Test degraded
    with raises(TypeError):
        # pes_func must be callable
        PesFunction(pes_func=[])
    # end with
    with raises(TypeError):
        # pes_args must be dict
        PesFunction(pes_func=pes_func, pes_args=[])
    # end with

    # Test nominal
    args = {"arg": 5}
    pf = PesFunction(pes_func=pes_func, pes_args=args)
    assert pf.args is args
    assert pf.func is pes_func

    # Test copy constructor
    pf_copy = PesFunction(pf)
    assert pf_copy.args is args
    assert pf_copy.func is pes_func

    # Copy takes precedence
    pf_copy2 = PesFunction(pf, pes_func_alt, {})
    assert pf_copy2.args is args
    assert pf_copy2.func is pes_func

    # Test evaluation (using simple summation)
    base = 11.0
    res = pf.evaluate(base)
    assert isinstance(res, PesResult)
    assert res.get_value() == pes_func(base, **args)[0]
    assert res.get_error() == pes_func(base, **args)[1]

    # Test evaluation with overriding kwargs
    new_args = {"arg": 7}
    res2 = pf.evaluate(base, **new_args)
    assert isinstance(res2, PesResult)
    assert res2.get_value() == pes_func(base, **new_args)[0]
    assert res2.get_error() == pes_func(base, **new_args)[1]

    # Test alternative constructor
    pf_alt = PesFunction(pes_func, args)
    assert pf_alt.args is args
    assert pf_alt.func is pes_func

# end def
