#!/usr/bin/env python3

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

import warnings
from numpy import nan

from nexus import PwscfAnalyzer

from stalk.params.PesResult import PesResult
from stalk.io.PesLoader import PesLoader


class PwscfPes(PesLoader):

    def __init__(
        self,
        args: dict = {},  # Keep 'args' for backward compatibility
        suffix='scf.in',

        **kwargs
    ):
        my_args = {'suffix': suffix}
        my_args.update(**args, **kwargs)
        super().__init__(**my_args)
    # end def

    def _load(self, filename: str, **kwargs) -> PesResult:
        ai = PwscfAnalyzer(filename, **kwargs)
        ai.analyze()
        if not hasattr(ai, "E") or ai.E == 0.0:
            # Analysis has failed
            warnings.warn(f"PwscfPes loader could not find energy in {filename}. Returning None.")
            E = nan
        else:
            E = ai.E
        # end if
        Err = 0.0
        return PesResult(E, Err)
    # end def

# end class

class PwscfEnthalpy(PesLoader):

    def __init__(self, args={}):
        self._func = None
        self.args = args
        self.target_pressure = None #expecting this to enter as GPa
        if 'target_pressure' in self.args:
            self.target_pressure = self.args['target_pressure']
            print(f' target_pressure set to {self.target_pressure} kbar')


    # end def

    def _load(self, filename: str, **kwargs) -> PesResult:
        ai = PwscfAnalyzer(filename, **kwargs)
        ai.analyze()
        
        if not hasattr(ai, "E") or ai.E == 0.0:
            # Analysis has failed
            warnings.warn(f"PwscfPes loader could not find energy in {str(input_file)}. Returning None.")
            E = nan
        else:
            E = ai.E # In units of Ry
        # end if

            #convert units
        if self.target_pressure is not None:
            P=self.target_pressure * (1/(14710*10)) # ryd/bohr3 per gpa
        # end if

        if not hasattr(ai, "volume") or ai.volume == 0.0:
            # Analysis has failed
            warnings.warn(f"PwscfPes loader could not find volume in {str(input_file)}. Returning None.")
            V = nan
        else:
            V = ai.volume # in units of bohr^3
            # convert units
            
        # end if
        Err = 0.0
        print(f"Energy:{E}\nVolume: {V}\nPressure:{P}\n\n")

        H = E+(P*V)
        return PesResult(H, Err)
    # end def

# end class
