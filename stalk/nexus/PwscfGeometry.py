#!/usr/bin/env python3

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

import warnings
from nexus import PwscfAnalyzer
from pathlib import Path

from stalk.util.util import PL
from stalk.io.GeometryLoader import GeometryLoader
from stalk.params.GeometryResult import GeometryResult


class PwscfGeometry(GeometryLoader):

    def __init__(self, args={}):
        self._func = None
        self.args = args
    # end def

    def _load(self, path, suffix='relax.in', c_pos=1.0, **kwargs):
        input_file = Path(PL.format(path, suffix))
        # Testing existence here, because Nexus will shut down everything upon failure
        if input_file.exists():
            ai = PwscfAnalyzer(str(input_file), **kwargs)
            ai.analyze()
        else:
            warnings.warn(f"PwscfGeometry loader could not find {str(input_file)}. Returning None.")
            return GeometryResult(None, None)
        # end if

        if not hasattr(ai, "structures") or len(ai.structures) == 0:
            warnings.warn(f"PwscfGeometry loader could not find structures in {str(input_file)}")
            return GeometryResult(None, None)
        # end if
        pos = ai.structures[len(ai.structures) - 1].positions * c_pos
        try:
            axes = ai.structures[len(ai.structures) - 1].axes * c_pos
        except AttributeError:
            # In case axes is not present in the relaxation
            axes = None
        # end try
        return GeometryResult(pos, axes)
    # end def

# end class
