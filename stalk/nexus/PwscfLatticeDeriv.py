#!/usr/bin/env python3

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

import warnings
from numpy import nan
from pathlib import Path

from nexus import PwscfAnalyzer

from stalk.params.ParameterSet import ParameterSet
from stalk.params.PesResult import PesResult
from stalk.util.util import PL
from stalk.io.PesLoader import PesLoader


class PwscfPes(PesLoader):

    def __init__(self, args={}):
        self._func = None
        self.args = args
    # end def

    def _load(self, structure: ParameterSet, suffix='scf.in', **kwargs):
        input_file = Path(PL.format(structure.file_path, suffix))
        # Testing existence here, because Nexus will shut down everything upon failure
        if input_file.exists():
            ai = PwscfAnalyzer(str(input_file), **kwargs)
            ai.analyze()
        else:
            warnings.warn(f"PwscfPes loader could not find {str(input_file)}. Returning None.")
            return PesResult(nan)
        # end if

        if not hasattr(ai, "E") or ai.E == 0.0:
            # Analysis has failed
            warnings.warn(f"PwscfPes loader could not find energy in {str(input_file)}. Returning None.")
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
    # end def

    def _load(self, structure: ParameterSet, suffix='scf.in', **kwargs):
        input_file = Path(PL.format(structure.file_path, suffix))
        # Testing existence here, because Nexus will shut down everything upon failure
        if input_file.exists():
            ai = PwscfAnalyzer(str(input_file), **kwargs)
            ai.analyze()
        else:
            warnings.warn(f"PwscfPes loader could not find {str(input_file)}. Returning None.")
            return PesResult(nan)
        # end if
        
        if not hasattr(ai, "E") or ai.E == 0.0:
            # Analysis has failed
            warnings.warn(f"PwscfPes loader could not find energy in {str(input_file)}. Returning None.")
            E = nan
        else:
            E = ai.E # In units of Ry
        # end if

        if not hasattr(ai, "pressure") or ai.pressure == 0.0:
            # Analysis has failed
            warnings.warn(f"PwscfPes loader could not find pressure in {str(input_file)}. Returning None.")
            P = nan
        else:
            P = ai.pressure # in units of kbar
            #convert units
            P *= 6.7978e-6 # Ry/bohr3 per Pa
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


class PwscfEnthalpyLatticeDeriv(PesLoader):
    def __init__(self,args={}):
        self.func = None
        self.args = args
        self.target_pressure = None #expecting this to enter as GPa
        if 'target_pressure' in self.args:
            self.target_pressure = self.args['target_pressure']

    def _load(self, structure: ParameterSet, suffix='scf.in', **kwargs):
            input_file = Path(PL.format(structure.file_path, suffix))
            # Testing existence here, because Nexus will shut down everything upon failure
            if input_file.exists():
                ai = PwscfAnalyzer(str(input_file), **kwargs)
                ai.analyze()
            else:
                warnings.warn(f"PwscfPes loader could not find {str(input_file)}. Returning None.")
                return PesResult(nan)
            # end if
            
            if not hasattr(ai, "E") or ai.E == 0.0:
                # Analysis has failed
                warnings.warn(f"PwscfPes loader could not find energy in {str(input_file)}. Returning None.")
                E = nan
            else:
                E = ai.E # In units of Ry
            # end if
            """
            if not hasattr(ai, "pressure") or ai.pressure == 0.0:
                # Analysis has failed
                warnings.warn(f"PwscfPes loader could not find pressure in {str(input_file)}. Returning None.")
                P = nan
            else:
                P = ai.pressure # in units of kbar
                #convert units
                P *= 6.7978e-6 # Ry/bohr3 per Pa
            # end if
            """
            if self.target_pressure is not None:
                P=self.target_pressure * (1/14710) # ryd/bohr3 per gpa
            else:
                warnings.warn("target pressure not set")
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
            stress = ai.stress
            lattice = ai.lattice
            lattice_inv = np.linalg.inv(lattice)
            P_identity = np.identity(P)

            dHdl = -V *((stress-P)@lattice_inv.T)


            return PesResult(H, Err)


    # end class
