#!/usr/bin/env python3

from stalk.params.GeometryResult import GeometryResult
from stalk.util.FunctionCaller import FunctionCaller


class GeometryLoader(FunctionCaller):

    def load(self, path, **kwargs):
        '''The Geometry loader must accept a "path" to input file and return GeometryResult.
        '''
        args = self.args.copy()
        args.update(kwargs)
        res = self._load(path=path, **args)
        return res
    # end def

    def _load(self, path=None, **kwargs):
        res = self.func(path=path, **kwargs)
        if not isinstance(res, GeometryResult):
            raise AssertionError('The _load method must return a GeometryResult.')
        # end if
        return res
    # end def

# end class
