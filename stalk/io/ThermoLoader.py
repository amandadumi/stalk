#!/usr/bin/env python3
"""ThermoLoader: wrapper for energy loaders that augments results with enthalpy."""

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

from numpy import isnan, isscalar

from stalk.params.PesResult import PesResult
from stalk.params.ThermoResult import ThermoResult
from stalk.util.FunctionCaller import FunctionCaller


class ThermoLoader:
    def __init__(self, loader, pressure=None, use_enthalpy=True):
        self.loader = loader
        self.pressure = pressure
        self.use_enthalpy = use_enthalpy

    # end def

    def load(self, structure, **kwargs):
        res = self.loader.load(structure)
        if not isinstance(res, PesResult):
            raise AssertionError("ThermoLoader wrapped loader must return PesResult.")
        # Determine pressure and volume
        p = args.pop("pressure", None)
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
            use_enthalpy=self.use_enthalpy,
        )

        if self.use_enthalpy and p is not None and v is not None:
            thermo.compute_enthalpy(pressure=p, volume=v)

        return thermo

    def _get_volume(self, structure):
        if hasattr(structure, "volume") and isscalar(structure.volume):
            return structure.volume
        if hasattr(structure, "axes") and structure.axes is not None:
            return abs(np.linalg.det(np.array(structure.axes, dtype=float)))
        return None

    # end def


# end class
