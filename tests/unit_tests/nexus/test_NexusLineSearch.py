from pytest import raises
import numpy as np

from stalk.nexus.NexusLineSearch import NexusLineSearch
from stalk.params.ParameterHessian import ParameterHessian
from stalk.util.util import match_to_tol
from stalk.nexus.NexusGenerator import NexusGenerator
from stalk.nexus.NexusStructure import NexusStructure
from nexus import run_project
from tests.unit_tests.assets.test_jobs import TestLoader, nxs_generic_pes
from ..assets.h2o import backward_H2O, elem_H2O, forward_H2O, pes_H2O, pos_H2O, hessian_H2O


def test_NexusLineSearch(tmp_path):

    # Test empty init
    ls = NexusLineSearch()
    assert not ls.generated
    assert not ls.analyzed
    assert len(ls.jobs) == 0
    assert len(ls.enabled_jobs) == 0
    with raises(TypeError):
        ls.evaluate()
    # end with

    # Test nominal init
    s = NexusStructure(
        label='label',
        pos=pos_H2O,
        elem=elem_H2O,
        forward=forward_H2O,
        backward=backward_H2O,
        units='A'
    )
    d = 1
    path = str(tmp_path) + '/path'
    hessian = ParameterHessian(hessian_H2O)
    M = 3
    R = 0.2
    fit_kind = 'pf2'
    pes = NexusGenerator(nxs_generic_pes, {'pes_variable': 'h2o'})
    ls = NexusLineSearch(
        structure=s,
        hessian=hessian,
        d=d,
        M=M,
        R=R,
        fit_kind=fit_kind,
        pes=pes,
        path=path,
        loader=TestLoader()
    )
    assert ls.generated
    assert ls.analyzed
    assert len(ls.jobs) == M
    assert len(ls.enabled_jobs) == M
    offsets_ref = np.linspace(-R, R, M)
    d_ref = hessian.get_directions()[d]
    assert match_to_tol(ls.offsets, offsets_ref)
    for offset, params, value in zip(offsets_ref, ls.get_shifted_params(), ls.values):
        params_ref = s.params + offset * d_ref
        assert match_to_tol(params, params_ref)
        assert match_to_tol(value, pes_H2O(backward_H2O(params_ref))[0], 1e-6)
    # end for
    assert match_to_tol(ls.errors, M * [0.0])

    # Test addition of two points, one enabled, one disabled
    enabled_shift = -0.3
    disabled_shift = 0.4
    ls.add_shift(enabled_shift)
    ls.add_shift(disabled_shift)
    ls.disable_value(disabled_shift)
    offsets_ref2 = [enabled_shift]
    for offset in offsets_ref:
        offsets_ref2 += [offset]
    # end for
    offsets_ref2 += [disabled_shift]
    match_to_tol(ls.offsets, offsets_ref2)
    assert not ls.generated
    assert not ls.analyzed
    # The number of jobs stays the same until generated again
    assert len(ls.jobs) == M
    assert len(ls.enabled_jobs) == M
    # Generating: changing path is recorded to old jobs, too
    ls.evaluate(pes=pes, path=path + '/new_round')
    assert len(ls.jobs) == M + 2
    assert len(ls.enabled_jobs) == M + 1
    assert ls.generated
    assert not ls.analyzed
    run_project(ls.enabled_jobs)
    # Then analyze all jobs
    ls.analyze_jobs(loader=TestLoader())
    assert ls.analyzed
    for offset, params, value in zip(offsets_ref2, ls.get_shifted_params(), ls.values):
        params_ref = s.params + offset * d_ref
        assert match_to_tol(params, params_ref)
        if offset == disabled_shift:
            assert value is None
        else:
            assert match_to_tol(value, pes_H2O(backward_H2O(params_ref))[0], 1e-6)
        # end if
    # end for
    assert match_to_tol(ls.errors, (M + 2) * [0.0])

# end def
