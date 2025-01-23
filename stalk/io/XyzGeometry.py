from numpy import loadtxt, savetxt, array

from stalk.params.ParameterStructure import ParameterStructure

from stalk.params.GeometryResult import GeometryResult
from .GeometryWriter import GeometryWriter
from .GeometryLoader import GeometryLoader


class XyzGeometry(GeometryLoader, GeometryWriter):

    def __load__(self, path, suffix='relax.xyz', c_pos=1.0):
        el, x, y, z = loadtxt('{}/{}'.format(path, suffix), dtype=str, unpack=True, skiprows=2)
        pos = array([x, y, z], dtype=float).T * c_pos
        return GeometryResult(pos, axes=None, elem=el)
    # end def

    def __write__(self, structure, path, suffix='structure.xyz', c_pos=1.0):
        assert isinstance(structure, ParameterStructure)
        output = []
        header = str(len(structure.elem)) + '\n'

        fmt = '{:<10f}'
        for el, pos in zip(structure.elem, structure.pos * c_pos):
            output.append([el, fmt.format(pos[0]), fmt.format(pos[1]), fmt.format(pos[2])])
        # end for
        savetxt('{}/{}'.format(path, suffix), array(output), header=header, fmt='%s', comments='')
    # end def

# end class
