#!/usr/bin/env python3
"""ThermoLoader: wrapper for energy loaders that augments results with enthalpy."""

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

import numpy as np
from numpy import isnan, isscalar

from stalk.params.PesResult import PesResult
from stalk.params.ThermoResult import ThermoResult
from stalk.util.FunctionCaller import FunctionCaller


class ThermoLoader:
    def __init__(
        self,
        loader,
        args={},
        backend=None,
        dhdp_from_dhdl_map=None,
        pressure=None,
        use_enthalpy=True,
        quantity=None,
    ):
        self.loader = loader
        self.args = args
        self.backend = backend
        self.pressure = pressure
        self.dhdp_from_dhdl = dhdp_from_dhdl_map

        if quantity is None:
            quantity = "enthalpy" if use_enthalpy else "energy"
        self.quantity = quantity 

        # handle pressure units:
        if self.backend == "pwscf":
            self._convert_pressure_pwscf()
        elif self.backend == "qmcpack":
            self._convert_pressure_qmcpack()
        else:
            raise ValueError("Unknown backend")

    # end def
    def _convert_pressure_pwscf(self):
        self.pressure *= (1 / 14710.5076) * (
            1 / 10
        )  # ryd/bohr3 per gpa; 10 kbar = 1 GPa

    def _convert_pressure_qmcpack(self):
        self.pressure *= (
            1 / (14710.5076 * 10)
        ) / 2  # ryd/bohr3 per kbar and then divided by to for Ha/Bohr3

    def load(self, structure, quantity= None, **kwargs):
        load_args = self.args.copy()
        load_args.update(kwargs)
        
        res = self.loader.load(structure)

        if not isinstance(res, PesResult):
            raise AssertionError("ThermoLoader wrapped loader must return PesResult.")

        quantity = quantity if quantity is not None else self.quantity

        # Determine pressure and volume
        p = load_args.pop("pressure", None)
        if p is None:
            p = self.pressure

        v = args.pop("volume", None)
        if v is None:
            v = self._get_volume(structure)

        # Build thermo result
        thermo = ThermoResult(
            energy=res.value,
            energy_error=res.error,
            pressure=p,
            volume=v,
        )

        dhdl = None
        dhdp = None
        if quantity == "enthalpy":
            L = structure.axes
            thermo._enthalpy, thermo.value = thermo.compute_enthalpy()
            if hasattr(res, "stress") and res.stress is not None:
                dH_dL = thermo.compute_enthalpy_gradient(L, res.stress, p)
            thermo.set_lattice_derivative(dhdl)

            if self.dhdp_from_dhdl_map is not None:
                dhdp = self.dhdp_from_dhdl_map(
                    dH_dL, L[0, 0], L[2, 2] / L[0, 0]
                )  # TODO: this is not general to other cells, only hcp
            elif self.use_enthalpy and p is None:
                raise ValueError("use_enthalpy is True, but pressure is not set")
            elif self.use_enthalpy and v is None:
                raise ValueError("use_enthalpy is True, but volume is not set")
            else:
                print("the stress wasnt parsed")
            thermo.dhdp = dhdp
            thermo.set_parameter_derivative(dH_dL)

            return thermo

    def _get_volume(self, structure):
        if hasattr(structure, "volume") and isscalar(structure.volume):
            return structure.volume
        if hasattr(structure, "axes") and structure.axes is not None:
            return abs(np.linalg.det(np.array(structure.axes, dtype=float)))
        return None

    # end def


# end class
