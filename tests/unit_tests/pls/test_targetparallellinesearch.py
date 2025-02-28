#!/usr/bin/env python

from numpy import array
from pytest import raises
from stalk.params import PesFunction
from stalk.pls.TargetParallelLineSearch import TargetParallelLineSearch
from stalk.util import match_to_tol

from ..assets.h2o import pes_H2O, get_structure_H2O, get_hessian_H2O
from ..assets.helper import Gs_N200_M7

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# test TargetParallelLineSearch class
def test_TargetParallelLineSearch():

    # test empty init
    with raises(TypeError):
        TargetParallelLineSearch()
    # end with
    structure = get_structure_H2O()
    hessian = get_hessian_H2O()
    srg = TargetParallelLineSearch(
        pes_func=pes_H2O,
        structure=structure,
        hessian=hessian,
        window_frac=0.2,
    )
    assert srg.setup
    assert srg.evaluated
    assert not srg.optimized

# end def
