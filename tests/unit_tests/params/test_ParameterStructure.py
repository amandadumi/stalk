#!/usr/bin/env python

from numpy import array
from pytest import raises
from stalk.util import match_to_tol

from ..assets.h2o import pos_H2O, elem_H2O, forward_H2O, backward_H2O
from ..assets.h2 import pos_H2, forward_H2, backward_H2, elem_H2
from ..assets.gese import params_GeSe, forward_GeSe, backward_GeSe, elem_GeSe

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# Test ParameterStructureBase class
def test_ParameterStructure():
    from stalk.params import ParameterStructure

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
    assert match_to_tol(s_H2.pos, pos_H2, tol)
    assert s_H2.axes is None
    for el, el_ref in zip(s_H2.elem, elem_H2):
        assert el == el_ref
    # end for
    assert s_H2.dim == 3
    # assert s_empty.params is None
    # assert s_empty.params_err is None
    assert s_H2.value == value
    assert s_H2.error == error
    assert s_H2.label == label
    assert s_H2.unit == unit
    assert s_H2.consistent
    assert s_H2.tol == tol
    
    return
    # test inconsistent pos vector
    with raises(AssertionError):
        s_H2.set_position([0.0, 0.0])
    # end with
    
    # test premature backward mapping
    assert s.map_backward()[0] is None
    assert s.map_backward()[1] is None
    # cannot be consistent without mapping functions
    assert not s.check_consistency()

    # test backward mapping
    s.init_params([1.4])
    s.set_backward_func(backward_H2)  # pos should now be computed automatically
    assert match_to_tol(s.pos, pos_H2, tol=1e-5)

    assert not s.check_consistency()  # still not consistent, without forward mapping
    # test premature forward mapping
    assert s.map_forward() is None

    s.set_forward_func(forward_H2)  # set forward mapping
    assert match_to_tol(s.pos, [0.0, 0.0, 0.7, 0.0, 0.0, -0.7], tol=1e-5)
    # set another pos
    s.set_position([0.0, 0.0, 0.0, 0.0, 0.0, 1.6])
    # params computed automatically
    assert match_to_tol(s.params, 1.6, tol=1e-5)
    assert s.check_consistency()  # finally consistent
    # also consistent at another point
    assert s.check_consistency(params=[1.3])
    assert s.check_consistency(pos=pos_H2)  # also consistent at another point
    # consistent set of arguments
    assert s.check_consistency(pos=pos_H2 * 0.5, params=[0.7])
    # inconsistent set of arguments
    assert not s.check_consistency(pos=pos_H2 * 0.5, params=[1.4])

    # test H2O (open; 2 parameters)
    s = ParameterStructure(pos=pos_H2O, forward=forward_H2O, elem=elem_H2O)
    params_ref = [0.95789707432, 104.119930724]
    assert match_to_tol(s.params, params_ref, tol=1e-5)

    # add backward mapping
    s.set_backward_func(backward_H2O)
    pos_ref = [[0., 0.000000, 0.00000],
               [0., 0.755450, 0.58895],
               [0., -0.75545, 0.58895]]
    assert match_to_tol(s.pos, pos_ref, tol=1e-5)
    assert s.check_consistency()

    # test another set of parameters
    s.set_params([1.0, 120.0])
    pos2_ref = [[0., 0.00000000, 0.],
                [0., 0.86602540, 0.5],
                [0., -0.8660254, 0.5]]
    assert match_to_tol(s.params, [1.0, 120.0], tol=1e-5)
    assert match_to_tol(s.pos, pos2_ref, tol=1e-5)

    jac_ref = array('''
    0.          0.
    0.          0.
    0.          0.
    0.          0.
    0.8660254   0.00436329
    0.5        -0.00755752
    0.          0.
   -0.8660254  -0.00436329
    0.5        -0.00755752
    '''.split(), dtype=float).reshape(-1, 2)
    assert match_to_tol(jac_ref, s.jacobian())

    # test periodic structure
    s = ParameterStructure(
        forward_GeSe,  # forward
        backward_GeSe,  # backward
        None,  # pos
        None,  # axes
        elem_GeSe,  # elem
        params_GeSe,
        None,  # params_err
        value=-10.0,  # value
        error=0.1,  # error
        label='GeSe test',  # label
        unit='crystal',  # unit
        dim=3,  # dim
    )
    pos_orig = s.pos
    s.shift_params([0.1, 0.1, 0.0, -0.1, 0.05])
    params_ref = [4.360000, 4.050000, 0.414000, 0.456000, 0.610000]
    pos_ref = array('''
    0.414000 0.250000 0.456000
    0.914000 0.750000 0.544000
    0.500000 0.250000 0.390000
    0.000000 0.750000 0.610000
    '''.split(), dtype=float)
    axes_ref = array('''
    4.360000 0.000000 0.000000
    0.000000 4.050000 0.000000
    0.000000 0.000000 20.000000
    '''.split(), dtype=float)
    assert match_to_tol(s.params, params_ref)
    assert match_to_tol(s.pos, pos_ref)
    assert match_to_tol(s.axes, axes_ref)
    dpos = pos_orig.flatten() - pos_ref
    s.shift_pos(dpos)
    params_ref2 = [4.360000, 4.050000, 0.414000, 0.556000, 0.560000]
    assert match_to_tol(s.params, params_ref2)
    assert match_to_tol(s.pos, pos_orig)
# end def