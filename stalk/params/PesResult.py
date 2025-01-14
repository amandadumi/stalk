from numpy import nan, isnan, random


class PesResult:
    '''Represents a PES evaluation result as value+error pair (float/nan)'''

    value = None
    error = None

    def __init__(self, value, error=0.0):
        if not isinstance(value, float):
            self.value = nan
            self.error = 0.0
        else:
            self.value = value
            if isinstance(error, float):
                self.error = error
            else:
                self.error = 0.0
            # end if
        # end if
    # end def

    def get_value(self):
        return self.value
    # end def

    def get_error(self):
        return self.error
    # end def

    def get_result(self):
        return self.get_value(), self.get_error()
    # end def

    def add_sigma(self, sigma):
        '''Add artificial white noise to the result for error resampling purposes.'''
        if isinstance(sigma, float) and sigma >= 0.0:
            self.error = (self.error**2 + sigma**2)**0.5
            if not isnan(self.value):
                self.value += sigma * random.randn(1)[0]
            # end if
        else:
            raise ValueError('Tried to add poor sigma: ' + str(sigma))
        # end if
    # end def

# end class
