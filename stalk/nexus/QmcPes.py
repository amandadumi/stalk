#!/usr/bin/env python3

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

import warnings
from numpy import nan, array
from pathlib import Path

from nexus import QmcpackAnalyzer

from stalk.nexus.NexusStructure import NexusStructure
from stalk.params.PesResult import PesResult
from stalk.io.PesLoader import PesLoader
from stalk.util.util import PL


class QmcPes(PesLoader):

    def __init__(self, args={}):
        self._func = None
        self.args = args
    # end def

    def _load(
        self,
        structure: NexusStructure,
        qmc_idx=1,
        suffix='dmc/dmc.in.xml',
        term='LocalEnergy',
        twist_averaging=False,
        twist_weights=None,
        **kwargs  # e.g. equilibration=None
    ) -> PesResult:
        input_file = Path(PL.format(structure.file_path, suffix))
        # Testing existence here, because Nexus will shut down everything upon failure
        if input_file.exists():
            ai = QmcpackAnalyzer(str(input_file), **kwargs)
            ai.analyze()
        else:
            warnings.warn(f"QmcPes loader could not find {str(input_file)}. Returning NaN.")
            return PesResult(nan)
        # end if

        if twist_averaging and self._check_bundled(ai):
            return self._perform_twist_averaging(ai, qmc_idx, term, twist_weights)
        else:
            if not hasattr(ai, "qmc") or len(ai.qmc) < qmc_idx or not hasattr(ai.qmc[qmc_idx], "scalars"):
                # Analysis has failed
                warnings.warn(f"QmcPes loader could not find energy in {str(input_file)}. Returning NaN.")
                return PesResult(nan)
            else:
                return self._analyze_energy_term(ai.qmc[qmc_idx].scalars, term)
            # end if
    # end def

    def _check_bundled(self, ai: QmcpackAnalyzer):
        if not hasattr(ai, "bundled_analyzers") or ai.bundled_analyzers is None:
            warnings.warn("QmcpackAnalyzer could not find twist bundles. Reverting to non-twist energy.")
            return False
        else:
            return True
        # end if
    # end def

    def _analyze_energy_term(self, scalars, label) -> PesResult:
        LE = scalars[label]
        value = LE.mean
        error = LE.error
        return PesResult(value, error)
    # end def

    def _perform_twist_averaging(self, ai: QmcpackAnalyzer, qmc_idx, label, twist_weights):
        if twist_weights is None:
            twist_weights = array(len(ai.bundled_analyzers) * [1])
        # end if
        weighted_sum = 0.0
        weighted_error2 = 0.0
        weight = 0.0
        for analyzer, w in zip(ai.bundled_analyzers, twist_weights):
            res = self._analyze_energy_term(analyzer.qmc[qmc_idx].scalars, label)
            weighted_sum += w * res.value
            weighted_error2 += w * res.error**2
            weight += w
        # end for
        weighted_sum /= weight
        weighted_error = weighted_error2**0.5 / weight
        return PesResult(weighted_sum, weighted_error)
    # end def

# end class

class QmcEnthalpy(PesLoader):

    def __init__(self, args={}):
        self._func = None
        self.args = args
    # end def

    def _load(
        self,
        structure: NexusStructure,
        qmc_idx=1,
        suffix='dmc/dmc.in',
        term='LocalEnergy',
        twist_averaging=False,
        twist_weights=None,
        **kwargs  # e.g. equilibration=None
    ):
        input_file = Path(PL.format(structure.file_path, suffix))
        # Testing existence here, because Nexus will shut down everything upon failure
        if input_file.exists():
            ai = QmcpackAnalyzer(str(input_file), **kwargs)
            ai.analyze()
        else:
            warnings.warn(f"QmcPes loader could not find {str(input_file)}. Returning None.")
            return PesResult(nan)
        # end if
        
        if twist_averaging and self._check_bundled(ai):
            return self._perform_twist_averaging(ai, qmc_idx, term, twist_weights)
        else:
            if not hasattr(ai, "qmc") or len(ai.qmc) < qmc_idx or not hasattr(ai.qmc[qmc_idx], "scalars"):
                # Analysis has failed
                warnings.warn(f"QmcPes loader could not find energy in {str(input_file)}. Returning None.")
                return PesResult(nan)
            else:
                P = self._analyze_stress_term(ai.qmc[qmc_idx].scalars)
                print(f'pressure: {P}')
                E,E_err =  self._analyze_energy_term(ai.qmc[qmc_idx].scalars, term)
                print(f'energy: {E}')
                V = self._analyze_volume_term(ai.info.system.structure)
                print(f'volume: {V}')
                H = E + (P*V)
                print(f'Enthalpy: {H}')
                return PesResult(H,E_err) 
        # end if
    # end def
    
    def _check_bundled(self, ai: QmcpackAnalyzer):
        if not hasattr(ai, "bundled_analyzers") or ai.bundled_analyzers is None:
            warnings.warn("QmcpackAnalyzer could not find twist bundles. Reverting to non-twist energy.")
            return False
        else:
            return True

    def _analyze_volume_term(self, structure):
        import numpy as np
        axes = structure.axes
        a = axes[0,:]
        b = axes[1,:]
        c = axes[2,:]
        return np.abs(np.dot(a,np.cross(b,c)))
        

    def _analyze_stress_term(self, scalars):
        force_00 = scalars['force_0_0']
        value_00 = force_00.mean
        error_00 = force_00.error
        
        force_11 = scalars['force_1_1']
        value_11 = force_11.mean
        error_11 = force_11.error
        
        force_22 = scalars['force_2_2']
        value_22 = force_22.mean
        error_22 = force_22.error
        print('stress values') 
        print(value_00, value_11,value_22)
        P = (1/3)*(value_00 + value_11 + value_22)
        return P

    def _analyze_energy_term(self, scalars, label='LocalEnergy'):
        LE = scalars[label]
        value = LE.mean
        error = LE.error
        return value,error

    #
    def _perform_twist_averaging(self, ai: QmcpackAnalyzer, qmc_idx, label, twist_weights):
        if twist_weights is None:
            twist_weights = array(len(ai.bundled_analyzers) * [1])
        # end if
        weighted_sum = 0.0
        weighted_error2 = 0.0
        weight = 0.0
        for analyzer, w in zip(ai.bundled_analyzers, twist_weights):
            e_res = self._analyze_energy_term(analyzer.qmc[qmc_idx].scalars, label)
            v_res = self._analyze_energy_term(analyzer.qmc[qmc_idx].scalars, label)
            p_res = self._analyze_stress_term(analyzer.qmc[qmc_idx].scalars)
            e_weighted_sum += w * e_res.value
            v_weighted_sum += w * v_res
            p_weighted_sum += w * p_res
            e_weighted_error2 += w * e_res.error**2
            weight += w
        # end for
        e_weighted_sum /= weight
        v_weighted_sum /= weight
        p_weighted_sum /= weight
        e_weighted_error = e_weighted_error2**0.5 / weight
        h_weighted_sum = e_weighted_sum + (p_weighted_sum*v_weighted_sum)
        return PesResult(h_weighted_sum, e_weighted_error)
    # end def

 #end def

# end class
