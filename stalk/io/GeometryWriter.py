#!/usr/bin/env python3

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


class GeometryWriter():
    args = None

    def __init__(self, args={}):
        assert isinstance(args, dict), 'Args must be inherited from dictionary.'
        self.args = args
    # end def

    def write(self, structure, path, **kwargs):
        '''The Geometry writer must accept a "structure" and a "path" to output file
        '''
        args = self.args.copy()
        args.update(kwargs)
        self.__write__(structure, path, **args)
    # end def

    def __write__(self, structure=None, path=None, *args, **kwargs):
        raise NotImplementedError(
            "Implement __load__ function in inherited class.")
    # end def

# end class
