#!/usr/bin/env python

import io
import sys
from pytest import raises

from stalk.params import Parameter

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# Test Parameter class
def test_Parameter():

    # Cannot construct without a value
    with raises(TypeError):
        Parameter()
    # end with

    # Cannot construct with non-scalar
    with raises(ValueError):
        Parameter([])
    # end with

    # test empty
    p = Parameter(0.0)
    assert p.value == 0.0
    assert p.error == 0.0
    assert p.unit == ''
    assert p.label == 'p'

    # test nominal
    value = 1.0
    error = 2.0
    label = 'label'
    unit = 'unit'
    p = Parameter(value, error, label=label, unit=unit)
    assert p.value == value
    assert p.error == error
    assert p.label == label
    assert p.unit == unit

    # cannot shift with non-scalar
    with raises(ValueError):
        p.shift([])
    # end with

    # test shifting
    shift = 3.0
    p.shift(shift)
    assert p.value == value + shift
    assert p.error == 0.0

    # test printing
    test_stdout = io.StringIO()
    sys.stdout = test_stdout
    print(p)
    sys.stdout = sys.__stdout__
    param_str = test_stdout.getvalue()
    param_str_ref = '     label: 4.000000   unit  \n'
    assert param_str == param_str_ref
# end def
