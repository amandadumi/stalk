#!/usr/bin/env python
"""Base class for representing an optimizable parameters.
"""


from numpy import isscalar

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


class Parameter():
    """Base class for representing an optimizable parameter"""
    _value: float
    _error: float = 0.0
    label: str = None
    unit: str = None

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

    def print_value(self):
        if self.error is None:
            print('{:<8.6f}             '.format(self.value))
        else:
            print('{:<8.6f} +/- {:<8.6f}'.format(self.value, self.error))
        # end if
    # end def

    def __str__(self):
        return '{:>10s}: {:<10f} {:6s}'.format(self.label, self.value, self.unit)
    # end def
# end class
