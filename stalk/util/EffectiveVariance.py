#!/usr/bin/env python3
'''Class to produce relative number of samples to meet an error target.'''

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


class EffectiveVariance():
    samples = None
    errorbar = None

    def __init__(
        self,
        samples,
        errorbar
    ):
        assert samples > 0, 'The number of samples must be > 0.'
        self.samples = samples
        assert errorbar > 0, 'The initial errorbar must be > 0.'
        self.errorbar = errorbar
    # end def

    def get_samples(self, errorbar):
        assert errorbar > 0, 'The requested errorbar must be > 0.'
        samples = self.samples * self.errorbar**2 * errorbar**-2
        return max(1, int(samples))
    # end def

    def get_errorbar(self, samples):
        assert samples > 0, 'The requested samples must be > 0'
        return self.errorbar * (float(self.samples) / samples)**0.5
    # end def

    def __str__(self):
        if self.errorbar is None or self.samples is None:
            return "Effective variances not set."
        else:
            return f"Effective variance: {(self.errorbar * self.samples)**2}"
        # end if
    # end def

# end class
