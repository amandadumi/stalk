#!/usr/bin/env python3

from pytest import raises
from structure import Structure
from simulation import Simulation
from stalk.nexus.NexusGeometry import NexusGeometry
from stalk.nexus.NexusStructure import NexusStructure

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

from ..assets.test_jobs import TestAnalyzer, nxs_generic_pes
from ..assets.h2o import pes_H2O, pos_H2O, elem_H2O


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


def test_NexusGeometry(tmp_path):

    s = NexusStructure(
        label='label',
        pos=pos_H2O,
        elem=elem_H2O,
        units='A'
    )
    path = str(tmp_path)

    # Test empty (should fail)
    with raises(TypeError):
        NexusGeometry()
    # end with

    # TODO: write the actual test

# end def
