#!/usr/bin/env python3
"""ThermoLoader: wrapper for energy loaders that augments results with enthalpy."""

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

from numpy import isnan, isscalar
import numpy as np

from stalk.params.PesResult import PesResult
from stalk.params.ThermoResult import ThermoResult
from stalk.util.FunctionCaller import FunctionCaller


class ThermoLoader:
    def __init__(self, loader, args={}, pressure=None, use_enthalpy=True):
        self.loader = loader
        self.pressure = pressure
        self.use_enthalpy = use_enthalpy
        self.args = args

    # end def

    def load(self, structure, **kwargs):
        print(structure)
        res = self.loader.load(structure)
        if not isinstance(res, PesResult):
            raise AssertionError("ThermoLoader wrapped loader must return PesResult.")
        # Determine pressure and volume
        p = self.args.pop("pressure", None)
        if p is None:
            p = self.pressure

        v = self.args.pop("volume", None)
        if v is None:
            v = self._get_volume(structure)

        # Build thermo result
        thermo = ThermoResult(
            energy=res.value,
            energy_error=res.error,
            pressure=p,
            volume=v,
            use_enthalpy=self.use_enthalpy,
        )

        if self.use_enthalpy and p is not None and v is not None:
            thermo.compute_enthalpy(pressure=p, volume=v)
            dHdL = self.compute_enthalpy_gradient()
            

        return thermo

    def _get_volume(self, structure):
        if hasattr(structure, "volume") and isscalar(structure.volume):
            return structure.volume
        if hasattr(structure, "axes") and structure.axes is not None:
            return abs(np.linalg.det(np.array(structure.axes, dtype=float)))
        return None

    # end def

    def compute_enthalpy_gradient(L, stress, pressure):
        """
        Compute the enthalpy gradient with respect to the lattice matrix.

        Parameters
        ----------
        L : (3, 3) array_like
            Lattice matrix with lattice vectors as rows or columns, consistent
            with the stress/strain convention used in your derivation.
        stress : (3, 3) array_like
            Stress tensor sigma_{mu beta}.
        pressure : float
            External pressure P.

        Returns
        -------
        dH_dL : (3, 3) ndarray
            Gradient dH/dL_{mu nu}.
        """
        L = np.asarray(L, dtype=float)
        stress = np.asarray(stress, dtype=float)

        if L.shape != (3, 3):
            raise ValueError(f"L must be 3x3, got shape {L.shape}")
        if stress.shape != (3, 3):
            raise ValueError(f"stress must be 3x3, got shape {stress.shape}")

        omega = np.linalg.det(L)
        Linv = np.linalg.inv(L)

        # delta_{mu beta}
        I = np.eye(3)

        # A_{mu beta} = sigma_{mu beta} - P delta_{mu beta}
        A = stress - pressure * I

        # dH/dL_{mu nu} = -Omega * sum_beta A_{mu beta} * (L^{-1})_{nu beta}
        # This is equivalent to -Omega * A @ Linv.T
        dH_dL = -omega * A @ Linv.T

        return dH_dL


# end class
