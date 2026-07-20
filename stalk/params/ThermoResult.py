#!/usr/bin/env python3
"""PesResult represents a PES evaluation result as value+error pair (float/nan)"""

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

import numpy as np

from stalk.params.PesResult import PesResult


class ThermoResult(PesResult):
    energy: float
    energy_err: float
    enthalpy: float
    enthalpy_error: float
    pressure: float | None
    volume: float | None
    stress: float | None
    lattice: float | None
    dH_dL: np.ndarray | None
    dH_dL_err: np.ndarray | None
    dH_dz: np.ndarray | None
    dH_dz_err: np.ndarray | None
    dE_dz: np.ndarray | None
    dE_dz_err: np.ndarray | None

    def __init__(
        self,
        energy,
        energy_error=0.0,
        pressure=None,
        volume=None,
        lattice=None,
        stress=None,
        enthalpy=None,
        enthalpy_error=0.0,
        dH_dz=None,
        dH_dz_err=None,
        dH_dL=None,
        dH_dL_err=None,
        dE_dz=None,
        dE_dz_err=None,
        use_enthalpy=False,
    ):
        super().__init__(energy, error=energy_error)
        self._energy = self.value
        self._energy_error = self.error
        self._pressure = pressure
        self._volume = volume
        self._enthalpy = enthalpy if enthalpy is not None else np.nan
        self._enthalpy_error = enthalpy_error
        self._dH_dz = None if dH_dz is None else np.array(dH_dz, dtype=float)
        self._dH_dz_err = (
            None if dH_dz_err is None else np.array(dH_dz_err, dtype=float)
        )
        self._dH_dL = None if dH_dL is None else np.array(dH_dL, dtype=float)
        self._dH_dL_err = (
            None if dH_dL_err is None else np.array(dH_dL_err, dtype=float)
        )
        self._dE_dz = None if dE_dz is None else np.array(dE_dz, dtype=float)
        self._dE_dz_err = (
            None if dE_dz_err is None else np.array(dE_dz_err, dtype=float)
        )

        if use_enthalpy:
            self.value = self._enthalpy
            self.error = self._enthalpy_error

    # end def

    @property
    def energy(self):
        return self._energy

    @property
    def enthalpy(self):
        return self._enthalpy

    @property
    def enthalpy_error(self):
        return self._enthalpy_error

    # end def

    @property
    def energy_error(self):
        return self._energy_error

    # end def

    @property
    def pressure(self):
        return self._pressure

    # end def

    @property
    def volume(self):
        return self._volume

    # end def

    @property
    def dH_dz(self):
        return self._dH_dz

    # end def

    @property
    def dH_dz_err(self):
        return self._dH_dz_err

    # end def

    @property
    def dE_dz(self):
        return self._dE_dz

    # end def

    @property
    def dE_dz_err(self):
        return self._dE_dz_err

    # end def

    def use_quantity(self, quantity):
        if quantity == "energy":
            self.value = self.energy
            self.error = self.energy_error
        elif quantity == "enthalpy":
            if np.isnan(self.enthalpy):
                self.enthalpy = self.compute_enthalpy()
            self.value = self.enthalpy
            self.error = self.enthalpy_error
        else:
            raise ValueError(f"Unknown quantity: {quantity}")

    def compute_enthalpy(self):
        self._enthalpy = self._energy + (self._pressure * self._volume)
        return self.enthalpy

    def compute_enthalpy_gradient(self, L, stress, pressure):
        """
        Compute the enthalpy gradient with respect to the lattice matrix.

        Parameters
        ----------
        L : (3, 3) np.array_like
            Lattice matrix with lattice vectors as rows or columns, consistent
            with the stress/strain convention used in your derivation.
        stress : (3, 3) np.array_like
            Stress tensor sigma_{mu beta}.
        pressure : float
            External pressure P.

        Returns
        -------
        dH_dL : (3, 3)np.ndarray
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

    def set_directional_derivative(self, dH_dz, dH_dz_err=None):
        self._dH_dz = None if dH_dz is None else np.array(dH_dz, dtype=float)
        self._dH_dz_err = (
            None if dH_dz_err is None else np.array(dH_dz_err, dtype=float)
        )

    # end def

    def set_lattice_derivative(self, dH_dL, dH_dL_err=None):
        self._dH_dL = np.array(dH_dL, dtype=float)
        self._dH_dL_err = (
            None if dH_dL_err is None else np.array(dH_dL_err, dtype=float)
        )

    def set_parameter_derivative(self, dH_dp, dH_dp_err=None):
        self._dH_dp = None if dH_dp is None else np.array(dH_dp, dtype=float)
        self._dH_dp_err = (
            None if dH_dp_err is None else np.array(dH_dp_err, dtype=float)
        )

    def set_parameter_derivative(self, dH_dp, dH_dp_err=None):
        return

    def rescale(self, scale):
        super().rescale(scale)
        self._energy = self.value
        self._energy_error = self.error
        if not np.isnan(self._enthalpy):
            self._enthalpy /= scale
            self._enthalpy_error /= scale
        # end if

    # end def

    def add_sigma(self, sigma):
        super().add_sigma(sigma)
        self._energy = self.value
        self._energy_error = self.error
        if not np.isnan(self._enthalpy):
            self._enthalpy_error = (self._enthalpy_error**2 + sigma**2) ** 0.5
        # end if

    # end def


# end class
