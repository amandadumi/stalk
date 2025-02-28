#!/usr/bin/env python

from pytest import raises

from stalk.params.PesResult import PesResult
from stalk.io.PesLoader import PesLoader

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# Test PesLoader class
def test_PesLoader(tmp_path):

    # Test empty init
    pl = PesLoader()
    assert len(pl.args) == 0
    with raises(NotImplementedError):
        # __load__ not implemented in abstract class
        pl.load('path')
    # end with

    with raises(TypeError):
        # args must be dict or None
        PesLoader(args=[])
    # end with

    # Test return by with a custom loader function
    def test_loader(path, arg=0):
        return PesResult(float(len(path)) + arg)
    # end def

    # Manually replace load method for testing
    args = {'arg': 2}
    pl = PesLoader(test_loader, args)

    path = "12345"
    res = pl.load(path)
    assert isinstance(res, PesResult)
    assert res.get_value() == 7.0
    assert res.get_error() == 0.0

    # Test overriding arg
    res2 = pl.load(path, arg=3)
    assert isinstance(res2, PesResult)
    assert res2.get_value() == 8.0
    assert res2.get_error() == 0.0

    # Test loading add_sigma > 0
    sigma = 1.23
    res_sigma = pl.load(path, sigma=sigma)
    assert isinstance(res_sigma, PesResult)
    assert res_sigma.get_value() != 5.0
    assert res_sigma.get_error() == sigma

    # Test copy constructor
    pl_copy = PesLoader(pl)
    assert pl_copy.__load__ is pl.__load__
    assert pl_copy.args is pl.args

# end def
