#!/usr/bin/env python3
'''Class to produce relative number of samples to meet an error target.'''

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

from numpy import isnan, nan, array, mean, isscalar


class EffectiveVariance():
    _errorbar_data = None

    def __init__(
        self,
        samples=None,
        errorbar=None
    ):
        self._errorbar_data = []
        if isscalar(samples) and isscalar(errorbar):
            self.add_errorbar_data(samples, errorbar)
        elif hasattr(samples, '__iter__') and hasattr(errorbar, '__iter__'):
            # Assume two lists were provided
            for s, e in zip(samples, errorbar):
                self.add_errorbar_data(s, e)
            # end for
        # else: do nothing
        # end if
    # end def

    def add_errorbar_data(self, samples, errorbar):
        if samples > 0 and errorbar > 0:
            self.errorbar_data.append([samples, errorbar])
        else:
            msg = 'Both the samples and errobar must be larger than 0, '
            msg += f'provided: samples={samples}, errorbar={errorbar}'
            raise ValueError(msg)
        # end if
    # end def

    @property
    def errorbar_data(self):
        return self._errorbar_data
    # end def

    @property
    def var_eff(self):
        if len(self.errorbar_data) == 0:
            var_eff = nan
        elif len(self.errorbar_data) == 1:
            var_eff = self.errorbar_data[0][0] * self.errorbar_data[0][1]**2
        else:
            data = array(self.errorbar_data).T
            var_eff = mean(data[0] * data[1]**2)
        # end if
        return var_eff
    # end

    def get_samples(self, errorbar, nodes=1):
        if errorbar <= 0:
            raise ValueError(f'The provided errorbar must be > 0, provided: {errorbar}')
        # end if
        if nodes < 1:
            raise ValueError(f'The number of nodes must be > 0, provided: {nodes}')
        # end if
        if isnan(self.var_eff):
            return -1
        else:
            samples = self.var_eff * errorbar**-2
            return max(1, int(samples / nodes))
        # end if
    # end def

    def get_errorbar(self, samples, nodes=1):
        if samples <= 0:
            raise ValueError(f'The provided samples must be > 0, provided: {samples}')
        # end if
        if nodes < 1:
            raise ValueError(f'The number of nodes must be > 0, provided: {nodes}')
        # end if
        return (nodes * self.var_eff / samples)**0.5
    # end def

    def __add__(self, other):
        if not isinstance(other, EffectiveVariance):
            raise TypeError('EffectiveVariance can only be added to EffectiveVariance')
        # end if
        new_samples = [data[0] for data in self.errorbar_data]
        new_samples += [data[0] for data in other.errorbar_data]
        new_errorbars = [data[1] for data in self.errorbar_data]
        new_errorbars += [data[1] for data in other.errorbar_data]
        return EffectiveVariance(new_samples, new_errorbars)
    # end def

    def __str__(self):
        if isnan(self.var_eff):
            return "Effective variance is not set."
        else:
            return f"Effective variance: {self.var_eff}"
        # end if
    # end def

# end class
