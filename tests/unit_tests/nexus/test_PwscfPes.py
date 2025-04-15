#!/usr/bin/env python3

from numpy import isnan

from stalk.nexus.PwscfPes import PwscfPes
from stalk.util.util import match_to_tol


def test_PwscfPes(tmp_path):

    # Test with empty args / defaults
    pes = PwscfPes()
    assert pes.func is None
    assert len(pes.args) == 0

    # default suffix: scf.in
    E_ref = -22.74988263  # See tests/unit_tests/assets/pwscf_pes/scf.out
    res = pes.load('tests/unit_tests/assets/pwscf_pes')
    assert match_to_tol(res.value, E_ref)
    assert match_to_tol(res.error, 0.0)

    # failing output file
    res2 = pes.load('tests/unit_tests/assets/pwscf_pes', suffix='scf_failed.out')
    assert isnan(res2.value)
    assert match_to_tol(res2.error, 0.0)

# end def
