#!/usr/bin/env python

from numpy import array
from pytest import raises
from stalk.util import match_to_tol
from stalk.params import ParameterStructure

from ..assets.h2 import backward_H2_alt, forward_H2_alt, pos_H2, forward_H2, backward_H2, elem_H2
from ..assets.gese import params_GeSe, forward_GeSe, backward_GeSe, elem_GeSe, pos_GeSe, axes_GeSe

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# Test ParameterStructure class
def test_ParameterStructure_open():

    # Test empty/default initialization
    s_empty = ParameterStructure()
    assert s_empty.forward_func is None
    assert s_empty.forward_args == {}
    assert s_empty.backward_func is None
    assert s_empty.backward_args == {}
    assert s_empty.pos is None
    assert s_empty.axes is None
    assert s_empty.elem is None
    assert s_empty.dim == 3
    assert s_empty.params is None
    assert s_empty.params_err is None
    assert s_empty.value is None
    assert s_empty.error == 0.0
    assert s_empty.label is None
    assert s_empty.unit is None
    assert not s_empty.consistent
    assert not s_empty.periodic
    assert s_empty.tol == 1e-7

    # Test nominal initialization (using H2 1-parameter model, pos init)
    value = 1.0
    error = 2.0
    tol = 0.1
    fwd_args = {'fwd': 1}
    bck_args = {'bck': 2}
    label = 'H2'
    unit = 'unit'
    s_H2 = ParameterStructure(
        forward=forward_H2,
        backward=backward_H2,
        forward_args=fwd_args,
        backward_args=bck_args,
        pos=pos_H2,
        elem=elem_H2,
        value=value,
        error=error,
        label=label,
        tol=tol,
        unit=unit
    )
    assert s_H2.forward_func == forward_H2
    assert s_H2.forward_args == fwd_args
    assert s_H2.backward_func == backward_H2
    assert s_H2.backward_args == bck_args
    match_to_tol(s_H2.pos, pos_H2, tol)
    assert s_H2.axes is None
    for el, el_ref in zip(s_H2.elem, elem_H2):
        assert el == el_ref
    # end for
    assert s_H2.dim == 3
    match_to_tol(s_H2.params, [1.4], tol)
    match_to_tol(s_H2.params_err, [0.0], tol)
    assert s_H2.value == value
    assert s_H2.error == error
    assert s_H2.label == label
    assert s_H2.unit == unit
    assert s_H2.consistent
    assert not s_H2.periodic
    assert s_H2.tol == tol

    # test setting position
    with raises(AssertionError):
        # Dimensions must be right
        s_H2.set_position(pos_H2[:, :-1])
    # end with
    # translate=False
    s_H2.set_position(pos_H2 + 2, translate=False)
    match_to_tol(s_H2.params, [1.4], tol)
    match_to_tol(s_H2.pos, pos_H2 + 2, tol)
    # translate=True
    s_H2.set_position(pos_H2 + 2, translate=False)
    match_to_tol(s_H2.params, [1.4], tol)
    match_to_tol(s_H2.pos, pos_H2, tol)

    # setting axes causes TypeError
    with raises(TypeError):
        s_H2.copy().set_axes([1.0, 2.0, 3.0])
    # end with

    # test setting of alternative forward func (params are divided by factor)
    factor = 3
    s_H2.set_forward_func(forward_H2_alt, {'factor': factor})
    match_to_tol(s_H2.params, [1.4 / factor], tol)
    match_to_tol(s_H2.params_err, [0.0], tol)
    # Not consistent...
    assert not s_H2.consistent
    # ...until matching mapping is provided
    s_H2.set_backward_func(backward_H2_alt, {'factor': factor})
    match_to_tol(s_H2.pos, pos_H2, tol)
    assert s_H2.consistent


# end def


def test_ParameterStructure_periodic():

    value = 1.0
    error = 2.0
    tol = 0.1
    fwd_args = {'fwd': 1}
    bck_args = {'bck': 2}
    label = 'GeSe'
    unit = 'unit'
    s_GeSe = ParameterStructure(
        forward=forward_GeSe,
        backward=backward_GeSe,
        forward_args=fwd_args,
        backward_args=bck_args,
        params=params_GeSe,
        value=value,
        error=error,
        label=label,
        tol=tol,
        unit=unit
    )
    assert s_GeSe.periodic

# end def
