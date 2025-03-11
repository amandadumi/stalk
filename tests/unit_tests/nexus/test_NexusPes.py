#!/usr/bin/env python3

from pytest import raises
from stalk.util.util import match_to_tol
from structure import Structure
from stalk.nexus.NexusPes import NexusPes
from stalk.nexus.NexusStructure import NexusStructure
from nexus import run_project
from simulation import Simulation

from ..assets.test_jobs import TestAnalyzer, nxs_generic_pes
from ..assets.h2o import pes_H2O, pos_H2O, elem_H2O

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# Dummy simulation class to suppress actual identifier check-ups in testing
class DummySimulation(Simulation):
    def __init__(self):
        pass
    # end def
# end class


def generator(structure, path, arg0='', arg1=''):
    # Test that the structure is nexus-friendly
    assert isinstance(structure, Structure)
    # Return something simple to test
    sim = DummySimulation()
    sim.data = path + arg0 + arg1
    return [sim]
# end def


def test_NexusPes(tmp_path):

    s = NexusStructure(
        label='label',
        pos=pos_H2O,
        elem=elem_H2O,
        units='A'
    )
    path = str(tmp_path)

    # Test empty (should fail)
    with raises(TypeError):
        NexusPes()
    # end with

# end def
