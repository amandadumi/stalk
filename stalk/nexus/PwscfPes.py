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
        print("")
        print("[PwscfPes] Attempting load")
        print(f"  label:      {getattr(structure, 'label', None)}")
        print(f"  file_path:  {getattr(structure, 'file_path', None)}")
        print(f"  suffix:     {suffix}")
        print(f"  full path:  {input_file}")
        print(f"  exists:     {input_file.exists()}")
        print("")

        # Testing existence here, because Nexus will shut down everything upon failure
        if input_file.exists():
            ai = PwscfAnalyzer(str(input_file), **kwargs)
            ai.analyze()
        else:
            warnings.warn(
                f"PwscfPes loader could not find {str(input_file)}. Returning PesResult(nan)."
            )
            return PesResult(np.nan)
        # end if

        if not hasattr(ai, "E") or ai.E == None:
            # Analysis has failed
            warnings.warn(
                f"PwscfPes loader could not find energy in {str(input_file)}. Returning  PesResult(nan)."
            )
            E = np.nan
        else:
            try:
                E = float(ai.E)
            except Exception:
                warnings.warn(
        f"PwscfPes loader could not parse energy in {str(input_file)}. Returning NaN."
                )
                E = np.nan
        # end if
        Err = 0
        res = PesResult(E, Err)
        print("[PwscfPes] Analyzer stress diagnostics")
        print(f"  has ai.stress: {hasattr(ai, 'stress')}")
        if hasattr(ai, "stress"):
            print(f"  ai.stress: {ai.stress}")
            try:
                print(f"  np.array(ai.stress).shape: {np.array(ai.stress).shape}")
            except Exception as e:
                print(f"  could not array(ai.stress): {e}")

        print(f"  has ai.sigma: {hasattr(ai, 'sigma')}")
        if hasattr(ai, "sigma"):
            print(f"  ai.sigma: {ai.sigma}")
            try:
                print(f"  np.array(ai.sigma).shape: {np.array(ai.sigma).shape}")
            except Exception as e:
                print(f"  could not array(ai.sigma): {e}")
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
