#!/usr/bin/env python

from stalk.io import XyzGeometry
from stalk.nexus.NexusStructure import NexusStructure
from stalk.params.ParameterStructure import ParameterStructure
from stalk.util.util import match_to_tol
from ..assets.test_jobs import nxs_generic_pes
from ..assets.h2o import backward_H2O, elem_H2O, forward_H2O, pes_H2O, pos_H2O

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# Test NexusStructure class
def test_NexusStructure(tmp_path):

    # empty init (matches ParameterStructure)
    s = NexusStructure()
    assert isinstance(s, ParameterStructure)

    # Meaningful init
    s = NexusStructure(
        label='label',
        pos=pos_H2O,
        elem=elem_H2O,
        forward=forward_H2O,
        backward=backward_H2O
    )
    # Original energy
    E_original = pes_H2O(s)
    # Shifted energy
    s.shift_params([0.1, -0.1])
    E_shifted = pes_H2O(s)
    assert E_original < E_shifted

    s.relax(
        path=str(tmp_path),
        pes_func=nxs_generic_pes,
        pes_args={'pes_variable': 'relax_h2o'},
        loader=XyzGeometry()
    )
    E_relax = pes_H2O(s)
    match_to_tol(E_original, E_relax, 1e-7)
    match_to_tol(s.pos, pos_H2O)

# end def
