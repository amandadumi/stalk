#!/usr/bin/env python

from structure import Structure
from stalk.nexus.NexusPes import NexusPes
from stalk.nexus.NexusStructure import NexusStructure
from stalk.params.ParameterStructure import ParameterStructure
from stalk.util.util import match_to_tol
from ..assets.test_jobs import nxs_generic_pes, TestLoader
from ..assets.h2o import backward_H2O, elem_H2O, forward_H2O, pes_H2O, pos_H2O

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# Test NexusStructure class
def test_NexusStructure(tmp_path):

    # empty init (matches ParameterStructure)
    s = NexusStructure()
    assert isinstance(s, ParameterStructure)
    assert not s.analyzed
    assert not s.generated
    assert not s.finished
    assert s.jobs is None
    assert isinstance(s.get_nexus_structure(), Structure)

    # Meaningful init
    s = NexusStructure(
        label='label',
        pos=pos_H2O,
        elem=elem_H2O,
        forward=forward_H2O,
        backward=backward_H2O,
        units='A'
    )
    assert isinstance(s.get_nexus_structure(), Structure)

    # 1: Test generation of jobs
    pes = NexusPes(
        nxs_generic_pes,
        args={'pes_variable': 'h2o'},
        loader=TestLoader()
    )
    pes.evaluate(s, path=str(tmp_path) + '/nosigma', sigma=0.0)
    assert s.generated
    assert len(s.jobs) == 1
    assert s.finished
    assert s.finished
    assert s.analyzed
    E_original = pes_H2O(pos_H2O)[0]
    assert match_to_tol(s.value, E_original)
    assert match_to_tol(s.error, 0.0)

    # 2b: Test analyzing of jobs (sigma)
    sigma = 0.1
    pes.evaluate(s, path=str(tmp_path) + '/sigma', sigma=sigma, add_sigma=True)
    # The value must have shifted
    assert not match_to_tol(s.value, E_original)
    assert match_to_tol(s.error, 0.1)

    # Test copy
    s_copy = s.copy()
    assert s_copy.jobs is None
    assert not s_copy.generated
    assert not s_copy.analyzed

# end def
