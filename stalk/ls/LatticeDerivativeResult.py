
#!/usr/bin/env python3
'''Class for linear fit of the derivative of the enthalpy'''

__author__ = "Amanda Dumi"
__email__ = "aedumi@sandia.gov"
__license__ = "BSD-3-Clause"

from numpy import polyder, polyval

from stalk.ls import FittingResult


class LinearDerivativeResult(FittingResult):

    def get_hessian(self, x):
        pfh = polyder(polyder(self.fit))
        return polyval(pfh, x)
    # end def

    def get_force(self, x):
        pfd = -polyder(self.fit)
        return polyval(pfd, x)
    # end def

    def get_values(self, x):
        return polyval(self.fit, x)
    # end def

# end class
