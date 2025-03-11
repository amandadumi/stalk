#!/usr/bin/env python3

from pytest import raises
from stalk.params.ParameterSet import ParameterSet
from stalk.params.PesFunction import PesFunction
from stalk.util.util import match_to_tol

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


def test_PesFunction():

    # Minimal function for testing vs callable
    def pes_func(s: ParameterSet, arg=0, sigma=0.0):
        return sum(s.params), arg
    # end def

    # Test degraded
    with raises(TypeError):
        # Cannot init empty
        PesFunction()
    # end with
    with raises(TypeError):
        # pes_func must be callable
        PesFunction(pes_func=[])
    # end with
    with raises(TypeError):
        # pes_args must be dict
        PesFunction(pes_func=pes_func, pes_args=[])
    # end with

    # Test nominal
    args = {"arg": 5.0}
    pf = PesFunction(func=pes_func, args=args)
    assert pf.args is args
    assert pf.func is pes_func

    # Test copy constructor
    pf_copy = PesFunction(pf)
    assert pf_copy.args is args
    assert pf_copy.func is pes_func

    # Test evaluation add_sigma=False
    params = [1.0, 2.0, 3.0]
    sigma = 3.5
    s = ParameterSet(params)
    pf.evaluate(s, add_sigma=False, sigma=sigma)
    assert s.value == sum(params)
    assert match_to_tol(s.error, args['arg'])

    pf.evaluate(s, add_sigma=True, sigma=sigma)
    assert s.value != sum(params)
    assert match_to_tol(s.error, (args['arg']**2 + sigma**2)**0.5)

# end def
