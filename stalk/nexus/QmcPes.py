#!/usr/bin/env python3

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

import warnings
import numpy as np
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
        stress_hf_label_format=None,
        stress_pulay_label_format=None,
        stress_index_base=0,
        stress_scale=1.0,
        stress_sign=1.0,
        symmetrize_stress=True,
        debug_scalars=False,
        **kwargs  # e.g. equilibration=None
    ) -> PesResult:
        input_file = Path(PL.format(structure.file_path, suffix))
        # Testing existence here, because Nexus will shut down everything upon failure
        if input_file.exists():
            ai = QmcpackAnalyzer(str(input_file), **kwargs)
            ai.analyze()
        else:
            warnings.warn(f"QmcPes loader could not find {str(input_file)}. Returning NaN.")
            return PesResult(np.nan)
        # end if

        print("[QmcPes] Available scalar terms:")
        print(list(ai.qmc[qmc_idx].scalars.keys()))

        print("[QmcPes] qmc object attributes:")
        print(ai.qmc[qmc_idx].__dict__.keys())

        if twist_averaging and self._check_bundled(ai):
            res =  self._perform_twist_averaging(
                    ai, 
                    qmc_idx, 
                    term, 
                    twist_weights,
                    stress_hf_label_format=stress_hf_label_format,
                    stress_pulay_label_format=stress_pulay_label_format,
                    stress_index_base=stress_index_base,
                    stress_scale=stress_scale,
                    stress_sign=stress_sign,
                    symmetrize_stress=symmetrize_stress,
                    debug_scalars=debug_scalars,
                    )
            return res
            

        # Non-twist case.
        qmc = self._get_qmc_block(ai, qmc_idx, input_file=input_file)
        if qmc is None:
            return PesResult(np.nan)

        if debug_scalars:
            print("[QmcPes] Available scalar terms:")
            print(list(qmc.scalars.keys()))
            print("[QmcPes] qmc object attributes:")
            print(qmc.__dict__.keys())

        res = self._analyze_energy_term(qmc.scalars, term)

        stress = self._extract_stress_from_scalars(
            qmc.scalars,
            hf_label_format=stress_hf_label_format,
            pulay_label_format=stress_pulay_label_format,
            index_base=stress_index_base,
            scale=stress_scale,
            sign=stress_sign,
            symmetrize=symmetrize_stress,
        )

        if stress is not None:
            res.stress = stress

        return res
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
   




    def _perform_twist_averaging(self, 
            ai: QmcpackAnalyzer, 
            qmc_idx, 
            label, 
            twist_weights,
            stress_hf_label_format=None,
            stress_pulay_label_format=None,
            stress_index_base=1,
            stress_scale=1.0,
            stress_sign=1.0,
            symmetrize_stress=True,
            debug_scalars=False,
            ):

        analyzers = ai.bundled_analyzers
        n_twists = len(analyzers)

        
        if twist_weights is None:
            twist_weights = np.ones(n_twists, dtype=float)
        else:
            twist_weights = np.array(twist_weights, dtype=float)


        if len(twist_weights) != n_twists:
            raise ValueError(
                f"twist_weights length {len(twist_weights)} does not match "
                f"number of bundled analyzers {n_twists}."
                )

        # end if
        weighted_energy = 0.0
        weighted_error2 = 0.0
        total_weight = 0.0

        weighted_stress = None
        stress_weight = 0.0
        missing_stress = 0

        #for analyzer, w in zip(ai.bundled_analyzers, twist_weights):
        for itwist, (analyzer, weight) in enumerate(zip(analyzers, twist_weights)):
            qmc = self._get_qmc_block(analyzer, qmc_idx)
            if qmc is None:
                warnings.warn(f"Skipping twist {itwist}: qmc block unavailable.")
                continue

            if debug_scalars and itwist == 0:
                print("[QmcPes] qmc object attributes for first twist:")
                print(qmc.__dict__.keys())
                print("[QmcPes] Available scalar terms for first twist:")
                print(list(qmc.scalars.keys()))

            res = self._analyze_energy_term(qmc.scalars, label)
            
            weighted_energy += w * res.value
            weighted_error2 += w * res.error**2
            total_weight += w

            stress_this = self._extract_stress_from_scalars(
                qmc.scalars,
                hf_label_format=stress_hf_label_format,
                pulay_label_format=stress_pulay_label_format,
                index_base=stress_index_base,
                scale=stress_scale,
                sign=stress_sign,
                symmetrize=symmetrize_stress,
            )

            if stress_this is None:
                missing_stress += 1
            else:
                if weighted_stress is None:
                    weighted_stress = np.zeros((3, 3), dtype=float)
                weighted_stress += weight * stress_this
                stress_weight += weight
        if total_weight <= 0.0:
            warnings.warn("No valid twist data found. Returning NaN.")
            return PesResult(np.nan)
        # end for
        energy_weighted_sum /= total_weight
        weighted_error = weighted_error2**0.5 / total_weightA

        result = PesResult(energy, error)
        if weighted_stress is not None and stress_weight > 0.0:
            result.stress = weighted_stress / stress_weight

            if missing_stress > 0:
                warnings.warn(
                    f"Stress was parsed for only {n_twists - missing_stress} of "
                    f"{n_twists} twists. Averaging available stress components only."
                )

        return result
    # end def


    def _extract_stress_from_scalars(
        self,
        scalars,
        hf_label_format=None,
        pulay_label_format=None,
        index_base=0,
        scale=1.0,
        sign=1.0,
        symmetrize=True,
    ):
        """
        Extract a 3x3 stress tensor from QMCPACK scalar labels.

        Example labels with label_format='force_{}_{}':

            force_0_0, force_0_1, ..., force_2_2

        Parameters
        ----------
        scalars
            QMCPACK analyzer scalar object/dict.
        hf_label_format : str or None
            Format string for Hellmann-Feynman tensor labels. Example: 'force_{}_{}'.
            If None, stress extraction is skipped.
        pulay_label_format : str or None
            Format string for pulay tensor labels. Example: 'force_{}_{}'.
            If None, stress extraction is skipped.
        index_base : int
            Use 0 if labels are force_0_0 ... force_2_2.
            Use 1 if labels are force_1_1 ... force_3_3.
        scale : float
            Multiplicative unit conversion factor.
        sign : float
            Sign convention factor. Use -1.0 if QMCPACK's reported quantity has
            the opposite sign from the stress convention used by ThermoResult.
        symmetrize : bool
            If True, replace tensor by 0.5 * (stress + stress.T).

        Returns
        -------
        stress : np.ndarray or None
            3x3 stress tensor, or None if unavailable.
        """
        if hf_label_format is None:
            return None
        if pulay_label_format is None:
            return None

        stress = np.zeros((3, 3), dtype=float)

        for i in range(3):
            #for j in range(3):
            j = i    
            hf_label = hf_label_format.format(i + index_base, j + index_base)
            pulay_label = pulay_label_format.format(i + index_base, j + index_base)
            print(hf_label)
            print(pulay_label)
            if hf_label not in scalars:
                warnings.warn(
                    f"QmcPes could not find stress scalar '{hf_label}'. "
                    "Stress will not be attached."
                )
                return None
            
            if pulay_label not in scalars:
                warnings.warn(
                    f"QmcPes could not find stress scalar '{pulay_label}'. "
                    "Stress will not be attached."
                )
                return None

            stress[i, j] = scalars[hf_label].mean + scalars[pulay_label].mean

        stress *= sign * scale

        if symmetrize:
            stress = 0.5 * (stress + stress.T)
        print(stress)
        return stress


    def _get_qmc_block(self, analyzer, qmc_idx, input_file=None):
        if not hasattr(analyzer, "qmc"):
            warnings.warn(
                f"QmcPes loader could not find qmc blocks"
                + (f" in {input_file}" if input_file is not None else "")
                + "."
            )
            return None

        if len(analyzer.qmc) <= qmc_idx:
            warnings.warn(
                f"QmcPes requested qmc_idx={qmc_idx}, but only found "
                f"{len(analyzer.qmc)} qmc blocks"
                + (f" in {input_file}" if input_file is not None else "")
                + "."
            )
            return None

        qmc = analyzer.qmc[qmc_idx]

        if not hasattr(qmc, "scalars"):
            warnings.warn(
                f"QmcPes qmc block {qmc_idx} has no scalars"
                + (f" in {input_file}" if input_file is not None else "")
                + "."
            )
            return None

        return qmc
# end class
