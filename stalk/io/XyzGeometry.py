#!/usr/bin/env python3

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

from numpy import loadtxt, savetxt, array

from stalk.params.ParameterStructure import ParameterStructure
from stalk.io.GeometryWriter import GeometryWriter
from stalk.io.GeometryLoader import GeometryLoader
from stalk.params.GeometryResult import GeometryResult
from stalk.util.util import PL


class XyzGeometry(GeometryLoader, GeometryWriter):

    def __init__(
        self,
        args={}
    ):
        self.args = args
    # end def

    def _load(self, path, suffix='relax.xyz', c_pos=1.0):
        el, x, y, z = loadtxt(
            PL.format(path, suffix),
            dtype=str,
            unpack=True,
            skiprows=2
        )
        pos = array([x, y, z], dtype=float).T * c_pos
        return GeometryResult(pos, axes=None, elem=el)
    # end def

    def __write__(self, structure, path, suffix='structure.xyz', c_pos=1.0):
        # TODO: branch for ParameterSet
        assert isinstance(structure, ParameterStructure)
        output = []
        header = str(len(structure.elem)) + '\n'

        fmt = '{:< 10f}'
        for el, pos in zip(structure.elem, structure.pos * c_pos):
            output.append([el, fmt.format(pos[0]), fmt.format(pos[1]), fmt.format(pos[2])])
        # end for
        savetxt(
            PL.format(path, suffix),
            array(output),
            header=header,
            fmt='%s',
            comments=''
        )
    # end def

# end class
