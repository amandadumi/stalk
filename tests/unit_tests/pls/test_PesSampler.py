#!/usr/bin/env python

from os import path
from pytest import raises

from ..assets.h2o import pes_H2O

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# Test PesSampler class
def test_PesSampler(tmp_path):
    from stalk.pls.PesSampler import PesSampler

    # Test empty init
    with raises(TypeError):
        # Cannot init with empty pes
        PesSampler()
    # end with

    save_path = str(tmp_path)
    ps = PesSampler(path=save_path, pes_func=pes_H2O)
    assert ps.pes.func is pes_H2O
    assert ps.path == save_path

    # Test write to disk (default)
    ps.write_to_disk()
    assert path.exists(save_path + '/data.p')

    # Test loading the image
    ps_load = PesSampler(load=save_path + '/data.p')
    assert ps_load.path == save_path
    assert ps_load.pes.func is pes_H2O

# end def
