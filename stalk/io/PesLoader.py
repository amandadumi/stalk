#!/usr/bin/env python3

import warnings
from numpy import nan

from stalk.io.util import load_energy
from stalk.params.ParameterSet import ParameterSet
from stalk.params.PesResult import PesResult
from stalk.util.ArgsContainer import ArgsContainer
from stalk.util.util import get_filename

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


class PesLoader(ArgsContainer):

    def load(
        self,
        arg: ParameterSet | str,
        sigma=0.0,
        only_warn=True,  # Only warn instead of raising exception
        **kwargs
    ) -> PesResult:
        if isinstance(arg, str):
            structure = ParameterSet()
            structure.file_path = arg
        else:
            structure = arg
        # end if
        # Hot update of args
        args = self.get_updated(kwargs)
        scale = args.pop('scale', 1.0)

        filename = get_filename(structure.file_path, args)
        if filename is None:
            if only_warn:
                warnings.warn(f'Could not find result in {structure.file_path}')
                return PesResult(nan)
            else:
                raise FileNotFoundError(f'Could not find result in {structure.file_path}')
            # end if
        else:
            res = self._load(filename, **args)
        # end if
        # Rescale to model units
        res.rescale(scale)
        # If a non-zero, artificial errorbar is requested, add it to result
        res.add_sigma(sigma)
        return res
    # end def

    def _load(self, structure, **kwargs) -> PesResult:
        res = load_energy(structure)
        return res
    # end def

# end class
