from pytest import raises
from stalk.util.util import match_to_tol
from structure import Structure
from stalk.nexus.NexusGenerator import NexusGenerator
from stalk.nexus.NexusStructure import NexusStructure
from nexus import run_project

from ..assets.test_jobs import TestAnalyzer, nxs_generic_pes
from ..assets.h2o import pes_H2O, pos_H2O, elem_H2O


def generator(structure, path, arg0='', arg1=''):
    # Test that the structure is nexus-friendly
    assert isinstance(structure, Structure)
    # Return something to test
    return [path + arg0 + arg1]
# end def


def test_NexusGenerator(tmp_path):

    s = NexusStructure(
        label='label',
        pos=pos_H2O,
        elem=elem_H2O
    )
    path = 'path'

    # Test empty (should fail)
    with raises(TypeError):
        NexusGenerator()
    # end with

    # Test nominal function call, no args
    gen_noargs = NexusGenerator(generator)
    assert gen_noargs.generate(s.get_nexus_structure(), path)[0] == 'path'

    # Test nominal function call, args
    gen_args = NexusGenerator(generator, {'arg0': '_arg0', 'arg1': '_arg1'})
    assert gen_args.generate(s.get_nexus_structure(), path)[0] == 'path_arg0_arg1'
    # Test nominal, arg overridden
    assert gen_args.generate(s.get_nexus_structure(), path, arg1='_over')[0] == 'path_arg0_over'

    # Test job generation
    gen_dummy = NexusGenerator(nxs_generic_pes, {'pes_variable': 'h2o'})
    jobs = gen_dummy.generate(s.get_nexus_structure(), str(tmp_path) + "/" + path)
    run_project(jobs)
    analyzer = TestAnalyzer(jobs[0])
    analyzer.analyze()
    energy_ref = pes_H2O(pos_H2O)
    # Assert that the results match
    match_to_tol(analyzer.value, energy_ref[0])
    match_to_tol(analyzer.error, energy_ref[1])

# end def
