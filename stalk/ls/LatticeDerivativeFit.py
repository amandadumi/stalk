from stalk.ls import FittingFunction
from stalk.ls.LatticeDerivativeResult import LatticeDerivativeResult


class LatticeDerivativeFit(FittingFunction):
    _n = None

    def __init__(self, n):
        self.n = n
    # end def

    @property
    def n(self):
        return self._n
    # end def

    @property
    def kind(self):
        return f'ldf{self.n}'
    # end def

    @n.setter
    def n(self, n):
        if not isinstance(n, int):
            raise TypeError(f'Polynomial degree n must be integer, provided: {n}')
        # end if
        if n < 2:
            raise ValueError(f'Polynomial degree n must be higher than 1, provided: {n}')
        # end if
        self._n = n
    # end def

    def _eval_function(self, offsets, values) -> LatticeDerivativeResult:
        if len(offsets) <= self.n:
            raise ValueError(f'Fitting of {self.n} degree polynomial to {len(offsets)} points is under-determined. Aborting.')
        # end if
        from scipy.optimize import brentq
        f_interp = interp1d(offsets, values, kind='linear')
# Find intervals where sign changes
        sign_changes = []
        for i in range(len(y_data) - 1):
            if y_data[i] * y_data[i+1] < 0:
                sign_changes.append((x_data[i], x_data[i+1]))
        if len(sign_changes>1):
            raise Warning('more than one root present')
        pfd = polyder(pf)
        r = roots(pfd)
        d = polyval(polyder(pfd), r)
        # filter real minima (maxima with sgn < 0)
        x_mins = r[where((r.imag == 0) & (d > 0))].real
        if len(x_mins) > 0:
            y_mins = polyval(pf, x_mins)
            imin = argmin(abs(x_mins))
        else:
            warnings.warn('The fit minimum not found inside grid but at the boundary!')
            x_mins = [min(offsets), max(offsets)]
            y_mins = polyval(pf, x_mins)
            imin = argmin(y_mins)  # pick the lowest/highest energy
        # end if
        y0 = y_mins[imin]
        x0 = x_mins[imin]
        res = PolynomialResult(x0, y0, fit=pf)
        return res
 
