from stalk.ls import FittingFunction
from stalk.ls.LatticeDerivativeResult import LatticeDerivativeResult


class LatticeDerivativeFit(FittingFunction):
    _n = None

    def __init__(self, n):
        self.n = n
    # end def

    @property
    def pressure_target(self):
        return self._pressure_target
    # end def

    @property
    def kind(self):
        return f'ldf{self.n}'
    # end def

    @n.setter
    def pressure(self, p):
        self._pressure_target = p
    # end def

    def _eval_function(self, offsets, values) -> LatticeDerivativeResult:
        if len(offsets) <= self.n:
        from scipy.optimize import brentq
        f_interp = interp1d(offsets, values, kind='linear')
        # Find intervals where sign changes
        sign_changes = []
        for i in range(len(y_data) - 1):
            if y_data[i] * y_data[i+1] < 0:
                sign_changes.append((x_data[i], x_data[i+1]))
        if len(sign_changes>1):
            raise Warning('more than one root present')
        roots = []
        for a, b in sign_changes:
            root = brentq(f_interp, a, b)
            roots.append(root)
        
        
        x_mins = r[where((r.imag == 0))].real
        if len(x_mins) > 0:
            y_mins = 0
        else:
            warnings.warn('The fit minimum not found inside grid but at the boundary!')
        # end if
        y0 = y_mins
        x0 = x_mins[0]
        res = PolynomialResult(x0, y0, fit=pf)
        return res
 
