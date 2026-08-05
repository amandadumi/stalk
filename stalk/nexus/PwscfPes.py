#!/usr/bin/env python3

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

import warnings
from pathlib import Path

import numpy as np
from nexus import PwscfAnalyzer

from stalk.io.PesLoader import PesLoader
from stalk.params.ParameterSet import ParameterSet
from stalk.params.PesResult import PesResult
from stalk.util.util import PL


class PwscfPes(PesLoader):

    def __init__(self, args={}):
        self._func = None
        self.args = args

    # end def

    def _load(self, structure: ParameterSet, suffix="scf.in", **kwargs):
        input_file = Path(PL.format(structure.file_path, suffix))
        # Testing existence here, because Nexus will shut down everything upon failure
        if input_file.exists():
            ai = PwscfAnalyzer(str(input_file), **kwargs)
            ai.analyze()
        else:
            warnings.warn(
                f"PwscfPes loader could not find {str(input_file)}. Returning None."
            )
            return PesResult(np.nan)
        # end if

        if not hasattr(ai, "E") or ai.E == 0.0:
            # Analysis has failed
            warnings.warn(
                f"PwscfPes loader could not find energy in {str(input_file)}. Returning None."
            )
            E = np.nan
        else:
            E = ai.E
        # end if
        Err = 0
        res = PesResult(E, Err)
        # Try to attach stress if available
        stress = None
        if hasattr(ai, "stress"):
            stress = np.array(ai.stress)[:, 0:3]
        elif hasattr(ai, "sigma"):
            stress = ai.sigma

        # end if

        if stress is not None:
            res.stress = stress

        return res

    # end def


# end class
