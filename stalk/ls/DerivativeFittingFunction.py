def __init__(
        self,
        energy,
        energy_error=0.0,
        pressure=None,
        volume=None,
        enthalpy=None,
        enthalpy_error=0.0,
        dH_dz=None,
        dH_dz_err=None,
        dE_dz=None,
        dE_dz_err=None,
        use_enthalpy=False,
    ):
        super().__init__(energy, error=energy_error)
        self._energy = self.value
        self._energy_error = self.error
        self._pressure = pressure
        self._volume = volume
        self._enthalpy = enthalpy if enthalpy is not None else nan
        self._enthalpy_error = enthalpy_error
        self._dH_dz = None if dH_dz is None else array(dH_dz, dtype=float)
        self._dH_dz_err = None if dH_dz_err is None else array(dH_dz_err, dtype=float)
        self._dE_dz = None if dE_dz is None else array(dE_dz, dtype=float)
        self._dE_dz_err = None if dE_dz_err is None else array(dE_dz_err, dtype=float)

        if use_enthalpy:
            self.value = self.enthalpy
            self.error = self.enthalpy_error
    # end def

    @property
    def energy(self):
        return self._energy
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
    def enthalpy(self):
        if not isnan(self._enthalpy):
            return self._enthalpy
        if self.pressure is not None and self.volume is not None:
            return self.energy + self.pressure * self.volume
        return nan
    # end def

    @property
    def enthalpy_error(self):
        return self._enthalpy_error if self._enthalpy_error is not None else self.energy_error
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

    def compute_enthalpy(self, pressure=None, volume=None):
        if pressure is not None:
            self._pressure = pressure
        if volume is not None:
            self._volume = volume
        if self.pressure is None or self.volume is None:
            raise ValueError("Need both pressure and volume to compute enthalpy.")
        self._enthalpy = self.energy + self.pressure * self.volume
        self._enthalpy_error = self.energy_error
        return self._enthalpy
    # end def

    def set_directional_derivative(self, dH_dz, dH_dz_err=None, dE_dz=None, dE_dz_err=None):
        self._dH_dz = None if dH_dz is None else array(dH_dz, dtype=float)
        self._dH_dz_err = None if dH_dz_err is None else array(dH_dz_err, dtype=float)
        self._dE_dz = None if dE_dz is None else array(dE_dz, dtype=float)
        self._dE_dz_err = None if dE_dz_err is None else array(dE_dz_err, dtype=float)
    # end def

    def rescale(self, scale):
        super().rescale(scale)
        self._energy = self.value
        self._energy_error = self.error
        if not isnan(self._enthalpy):
            self._enthalpy /= scale
            self._enthalpy_error /= scale
        # end if
    # end def

    def add_sigma(self, sigma):
        super().add_sigma(sigma)
        self._energy = self.value
        self._energy_error = self.error
        if not isnan(self._enthalpy):
            self._enthalpy_error = (self._enthalpy_error**2 + sigma**2)**0.5
        # end if
    # end def

# end class

