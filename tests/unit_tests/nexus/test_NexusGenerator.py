from pytest import raises
from structure import Structure
from stalk.nexus.NexusGenerator import NexusGenerator
from stalk.nexus.NexusStructure import NexusStructure


def generator(structure, path, arg0='', arg1=''):
    # Test that the structure is nexus-friendly
    assert isinstance(structure, Structure)
    # Return something to test
    return [path + arg0 + arg1]
# end def


def test_NexusGenerator():

    s = NexusStructure(label='label')
    path = '_path'

    # Test empty (should fail)
    with raises(TypeError):
        NexusGenerator()
    # end with

    # Test nominal, no args
    gen_noargs = NexusGenerator(generator)
    assert gen_noargs.generate(s.get_nexus_structure(), path)[0] == '_path'

    # Test nominal, args
    gen_args = NexusGenerator(generator, {'arg0': '_arg0', 'arg1': '_arg1'})
    assert gen_args.generate(s.get_nexus_structure(), path)[0] == '_path_arg0_arg1'
    # Test nominal, arg overridden
    assert gen_args.generate(s.get_nexus_structure(), path, arg1='_over')[0] == '_path_arg0_over'

# end def
