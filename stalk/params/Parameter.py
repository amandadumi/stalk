#!/usr/bin/env python
"""Base class for representing an optimizable parameters.
"""


from numpy import isscalar

from stalk.util.util import FF, FLL, FU

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


class Parameter():
    """Base class for representing an optimizable parameter"""
    _value: float
    _error: float = 0.0
    label: str = ''
    unit: str = ''

    def __init__(
        self,
        value,
        error=0.0,
        label='p',
        unit='',
    ):
        self.value = value
        self.error = error
        self.label = label
        self.unit = unit
    # end def

    @property
    def value(self):
        return self._value
    # end def

    @value.setter
    def value(self, value):
        if isscalar(value):
            self._value = value
        else:
            raise ValueError("Value must be scalar!")
        # end if
    # end def

    @property
    def error(self):
        return self._error
    # end def

    @error.setter
    def error(self, error):
        if isscalar(error):
            self._error = error
        else:
            self._error = 0.0
        # end if
    # end def

    def shift(self, shift):
        if isscalar(shift):
            self.value += shift
            self.error = 0.0
        else:
            raise ValueError("Shift must be scalar!")
        # end if
    # end def

    def __str__(self):
        string = (FLL + FF).format(self.label, self.value)
        if self.error > 0:
            string += FU.format(self.error)
        # end if
        string += FLL.format(self.unit)
        return string
    # end def
# end class
