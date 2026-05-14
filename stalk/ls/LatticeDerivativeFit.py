from stalk.ls import FittingFunction
from stalk.ls.LatticeDerivativeResult import LatticeDerivativeResult
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy

class LatticeDerivativeFit(FittingFunction):
    _n = None

    def __init__(self, interp_kind='linear'):
        self.interp_kind = interp_kind

    # end def

    @property
    def target_pressure(self):
        return self.target_pressure
    # end def

    @property
    def kind(self):
        return f'ldf{self.n}'
    # end def

    @target_pressure.setter
    def target_pressure(self, p):
        self.target_pressure= p
    # end def

    def _eval_function(self, offsets, values) -> LatticeDerivativeResult:
        if len(offsets) < 2:
            raise ValueError("Need at least 2 points to find a root.")
        
        # sort by offsets just in case
        idx = numpy.array(offsets).argsort()
        offsets = numpy.array(offsets)[idx]
        values = numpy.array(values)[idx]

        f_interp = interp1d(offsets, values, kind=self.interp_kind, fill_value='extrapolate')
        # Find intervals where sign changes
        roots = []
        for i in range(len(offsets) - 1):
            y1, y2 = values[i], values[i+1] 
            if y1 ==0.0:
                roots.append(offsets[i])
            if y1*y2 < 0:
                root = brentq(lambda z: float(f(z)), offsets[i], offsets[i + 1])
                roots.append(root)        
        if len(roots) == 0:
            raise ValueError("No root found in the provided interval")
        x0 = min(roots,key=abs)
        y0 = 0.0
        res = LatticeDerivativeResult(x0, y0, fit=f_interp)
        return res
 
