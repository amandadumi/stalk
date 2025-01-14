from pytest import raises
from numpy import isnan
from stalk.params.PesResult import PesResult


def test_PesResult():
    # Test degraded
    # Cannot init empty
    with raises(TypeError):
        PesResult()
    # end with

    val = 1.0
    err = 2.0
    err_default = 0.0
    sigma = 3.0

    # test nominal (no error)
    res0 = PesResult(val)
    assert res0.get_value() == val
    assert res0.get_error() == err_default
    res0.add_sigma(sigma)
    assert res0.get_error() == sigma
    # Only test that value has been changed but not by how much
    assert res0.get_value() != val

    # Test nominal (with error)
    res1 = PesResult(val, err)
    assert res1.get_value() == val
    assert res1.get_error() == err
    # Add zero sigma and expect no effect
    res1.add_sigma(0.0)
    assert res1.get_value() == val
    assert res1.get_error() == err
    assert res1.get_result()[0] == val
    assert res1.get_result()[1] == err
    res1.add_sigma(sigma)
    assert res1.get_error() == (err**2 + sigma**2)**0.5
    # Only test that value has been changed but not by how much
    assert res1.get_value() != val

    # Test degraded (Nan value)
    res2 = PesResult(None, err)
    assert isnan(res2.get_value())
    assert res2.get_error() == 0.0

    # Test degraded (Nan error, sigma)
    res2 = PesResult(val, None)
    assert res2.get_value() == val
    assert res2.get_error() == 0.0
    with raises(ValueError):
        res2.add_sigma([])
    # end with
    with raises(ValueError):
        res2.add_sigma(-1e-9)
    # end with

    # Test degraded (Nan value/error)
    res2 = PesResult(None, None)
    assert isnan(res2.get_value())
    assert res2.get_error() == 0.0

# end def
