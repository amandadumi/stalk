#!/usr/bin/env python

from pytest import raises
from stalk.io import XyzGeometry
from stalk.nexus.NexusGenerator import NexusGenerator
from stalk.nexus.NexusStructure import NexusStructure
from stalk.params.ParameterStructure import ParameterStructure
from stalk.util.util import match_to_tol
from ..assets.test_jobs import nxs_generic_pes, TestLoader
from ..assets.h2o import backward_H2O, elem_H2O, forward_H2O, pes_H2O, pos_H2O

from nexus import Structure, run_project

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
    assert s._job_path == ''
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
    path = 'path'
    assert s._make_job_path(path) == 'path/label'
    assert isinstance(s.get_nexus_structure(), Structure)

    # 1: Test generation of jobs
    with raises(TypeError):
        # pes must be NexusGenerator
        s.generate_jobs(pes=[])
    # end with
    with raises(AssertionError):
        # Cannot load before generated
        s.analyze_pes(loader=TestLoader())
    # end with
    s.generate_jobs(
        pes=NexusGenerator(nxs_generic_pes, {'pes_variable': 'h2o'}),
        path=str(tmp_path)
    )
    assert s.generated
    assert len(s.jobs) == 1
    assert not s.finished
    assert not s.analyzed
    run_project(s.jobs)
    assert s.finished

    # 2a: Test analyzing of jobs (no sigma)
    s.analyze_pes(loader=TestLoader())
    assert s.generated
    assert len(s.jobs) == 1
    assert s.finished
    assert s.analyzed
    E_original = pes_H2O(pos_H2O)[0]
    assert match_to_tol(s.value, E_original)
    assert match_to_tol(s.error, 0.0)

    # 2b: Test analyzing of jobs (sigma)
    sigma = 0.1
    s.analyze_pes(loader=TestLoader(), sigma=sigma)
    # The value must have shifted
    assert not match_to_tol(s.value, E_original)
    assert match_to_tol(s.error, 0.1)

    # Test copy
    s_copy = s.copy()
    assert s_copy.jobs is None
    assert not s_copy.generated
    assert not s_copy.analyzed

    # Test relax and shift
    s = NexusStructure(
        label='label',
        pos=pos_H2O,
        elem=elem_H2O,
        forward=forward_H2O,
        backward=backward_H2O,
        units='A'
    )
    E_original = pes_H2O(pos_H2O)[0]

    params_orig = s.params.copy()
    s.shift_params([0.2, -0.2])
    assert s.value is None
    assert s.error == 0.0
    assert not s.analyzed
    assert not s.generated
    assert not s.finished
    s.relax(
        path=str(tmp_path),
        pes_func=nxs_generic_pes,
        pes_args={'pes_variable': 'relax_h2o'},
        loader=XyzGeometry()
    )
    E_relax = pes_H2O(s)[0]
    assert match_to_tol(E_original, E_relax, 1e-7)
    assert match_to_tol(s.params, params_orig)

# end def
