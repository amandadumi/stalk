#!/usr/bin/env python3

import warnings
from numpy import nan

from numpy import isscalar, isnan

from stalk.params.GeometryResult import GeometryResult
from stalk.params.ParameterSet import ParameterSet
from stalk.util.ArgsContainer import ArgsContainer
from stalk.util.util import get_filename

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


class GeometryLoader(ArgsContainer):

    def load(self, path, c_pos=None, only_warn=False, **kwargs) -> GeometryResult:
        '''The Geometry loader must accept a "path" to input file and return GeometryResult.
        '''
        # Hot update of args
        args = self.get_updated(kwargs)
        # Keep for backward compatibility
        if isscalar(c_pos):
            args['scale'] = c_pos**-1
        # end if
        scale = args.pop('scale', 1.0)

        filename = get_filename(path, args)
        if filename is None:
            if only_warn:
                warnings.warn(f'Could not find result in {path}')
                return GeometryResult(nan)
            else:
                raise FileNotFoundError(f'Could not find result in {path}')
            # end if
        else:
            res = self._load(filename, **args)
        # end if
        # Rescale to model units
        res.rescale(scale)
        return res
    # end def

    def load_or_relax(
        self,
        path,
        relax_func: callable,
        structure: ParameterSet,
        relax_args={},
        **kwargs
    ) -> ParameterSet:
        res = self.load(path, **kwargs)
        if not isnan(res.pos):
            return structure.copy(pos=res.pos, axes=res.axes)
        # end i
        if not callable(relax_func):
            raise TypeError('The relax_func must be callable and return GeometryResult!')
        # end if
        # Try to relax
        relax_func(structure, **relax_args)
        # Then, try to load again
        res = self.load(path, **kwargs)
        return structure.copy(pos=res.pos, axes=res.axes)
    # end def

    # The actual loading function must be overridden and return a GeometryResult object
    def _load(self, filename, **kwargs) -> GeometryResult:
        raise NotImplementedError("Implement _load(filename) function in inherited class.")
    # end def

# end class
