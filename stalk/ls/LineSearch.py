from numpy import array, linspace, polyval, sign, equal, isscalar
from matplotlib import pyplot as plt

from stalk.params.ParameterHessian import ParameterHessian
from stalk.params.PesFunction import PesFunction
from stalk.params.ParameterSet import ParameterSet
from stalk.ls.LineSearchBase import LineSearchBase


# Class for PES line-search in structure context
class LineSearch(LineSearchBase):
    _structure: ParameterSet | None = None  # The equilibrium structure
    _hessian: ParameterHessian | None = None  # The equilibrium full Hessian
    _sigma = 0.0  # Target errorbar
    _d: int | None = None  # direction count

    def __init__(
        self,
        structure=None,
        hessian=None,
        d=None,
        sigma=0.0,
        grid=None,
        M=7,
        W=None,
        R=None,
        **ls_args
        # values=None, errors=None, fraction=0.025, sgn=1
        # fit_kind='pf3', fit_func=None, fit_args={}, N=200, Gs=None
    ):
        LineSearchBase.__init__(self, grid=None, **ls_args)
        self.sigma = sigma
        if structure is not None:
            self.structure = structure
        # end if
        if hessian is not None:
            self.hessian = hessian
        # end if
        if d is not None:
            self.d = d
        # end if
        # Try to initialize grid based on available information
        try:
            self.initialize_grid(M=M, W=W, R=R, grid=grid)
        except ValueError:
            # Setting the grid later then
            pass
        # end try
    # end def

    @property
    def structure(self):
        return self._structure
    # end def

    @structure.setter
    def structure(self, structure):
        if isinstance(structure, ParameterSet):
            if structure.check_consistency():
                self._structure = structure
                # Empty grid when updating structure
                self._grid = []
            else:
                raise ValueError('Provided structure is not a consistent mapping')
            # end if
        else:
            raise ValueError('Provided structure is not a ParameterSet object')
        # end if
    # end def

    @property
    def sigma(self):
        return self._sigma
    # end def

    @sigma.setter
    def sigma(self, sigma):
        if isscalar(sigma) and sigma >= 0.0:
            self._sigma = sigma
        else:
            raise ValueError("Sigma must be >= 0.0")
        # end if
    # end def

    @property
    def d(self):
        return self._d
    # end def

    @d.setter
    def d(self, d):
        if self.hessian is not None:
            max_d = len(self.hessian)
        elif self.structure is not None:
            max_d = len(self.structure)
        else:
            max_d = 1e10
        # end if
        if isinstance(d, int) and d < max_d:
            self._d = d
        else:
            raise ValueError('d must be integer smaller than the Hessian/structure dimension')
        # end if
    # end def

    @property
    def hessian(self):
        return self._hessian
    # end def

    @hessian.setter
    def hessian(self, hessian):
        if isinstance(hessian, ParameterHessian):
            self._hessian = hessian
            Lambda = self.hessian.get_lambda(self.d)
            self.sgn = sign(Lambda)
            if self.structure is None:
                # Use Hessian structure if no other has been provided yet
                self.structure = hessian.structure
            # end if
        else:
            raise ValueError('Provided Hessian is not a ParameterHessian object')
        # end if
    # end def

    @property
    def Lambda(self):
        return None if self.hessian is None else abs(self.hessian.get_lambda(self.d))
    # end def

    @property
    def direction(self):
        if self.d is None:
            return 0.0
        # end if
        if self.hessian is not None:
            return self.hessian.get_directions(self.d)
        elif self.structure is not None:
            # Get pure parameter direction
            direction = len(self.structure) * [0.0]
            direction[self.d] += 1.0
            return array(direction)
        else:
            return 0.0
        # end if
    # end def

    def initialize_grid(self, M=7, W=None, R=None, grid=None):
        if grid is None:
            if M < 0:
                raise ValueError("Grid size M must be positive!")
            # end if
            if R is not None:
                offsets = self._make_grid_R(R, M=M)
            elif W is not None and self.hessian is not None:
                if self.hessian is None:
                    raise ValueError('Must set Hessian before using W to set grid.')
                else:
                    offsets = self._make_grid_W(W, M=M)
                # end if
            else:
                raise ValueError('Must provide grid, R or W to characterize grid.')
            # end if
        else:
            offsets = grid
        # end if

        if self.structure is None:
            # Reverting to LineSearchPoints
            self.grid = offsets
        else:
            # Using shifted parametric structures
            self.grid = [self._shift_structure(offset) for offset in offsets]
        # end if
    # end def

    def _make_grid_R(self, R, M):
        if R < 0:
            raise ValueError("R must be positive!")
        # end if
        R = max(R, 1e-4)
        grid = linspace(-R, R, M)
        return grid
    # end def

    def _make_grid_W(self, W, M):
        if W < 0:
            raise ValueError("W must be positive!")
        # end if
        R = self._W_to_R(max(W, 1e-4))
        return self._make_grid_R(R, M=M)
    # end def

    def _W_to_R(self, W):
        """Map W to R"""
        if self.Lambda is None:
            return None
        else:
            return (2 * W / self.Lambda)**0.5
        # end if
    # end def

    def _R_to_W(self, R):
        """Map R to W"""
        if self.Lambda is None:
            return None
        else:
            return 0.5 * self.Lambda * R**2
        # end if
    # end def

    @property
    def R_max(self):
        if len(self) > 0:
            return min([-self.offsets.min(), self.offsets.max()])
        else:
            return 0.0
        # end if
    # end def

    @property
    def W_max(self):
        return self._R_to_W(self.R_max)
    # end def

    def add_shift(self, shift):
        if self.structure is None:
            # Reverting to LineSearchPoint
            self.add_point(shift)
        else:
            structure = self._shift_structure(shift)
            self.add_point(structure)
        # end if
    # end def

    def _shift_structure(self, shift, roundi=4):
        shift_rnd = round(shift, roundi)
        params_this = self.structure.params
        if shift_rnd == 0.0:
            label = 'eqm'
            params = params_this.copy()
        else:
            sgn = '' if shift_rnd < 0 else '+'
            label = 'd{}_{}{}'.format(self.d, sgn, shift_rnd)
            params = params_this + shift * self.direction
        # end if
        structure = self.structure.copy(params=params, label=label, offset=shift)
        return structure
    # end def

    def evaluate_pes(
        self,
        pes_eval,
        add_sigma=False
    ):
        '''Evaluate the PES on the line-search grid using an evaluation function.'''
        assert isinstance(
            pes_eval, PesFunction), 'The evaluation function must be inherited from PesFunction class.'
        for point in self._grid:
            res = pes_eval.evaluate(point, sigma=self.sigma)
            if add_sigma:
                res.add_sigma(self.sigma)
            # end if
            point.value = res.get_value()
            point.error = res.get_error()
        # end for
        self._search_and_store()
    # end def

    def set_results(
        self,
        grid,
        values,
        errors=None,
        **kwargs
    ):
        grid = grid if grid is not None else self.grid
        if values is None or all(equal(array(values), None)):
            return False
        # end if
        if errors is None:
            errors = 0.0 * array(values)
        # end if
        self.values = values
        self.errors = errors
        return True
    # end def

    def get_shifted_params(self):
        if len(self) > 0:
            return array([structure.params for structure in self.grid if isinstance(structure, ParameterSet)])
        else:
            return None
        # end if
    # end def

    def plot(
        self,
        ax=None,
        figsize=(4, 3),
        color='tab:blue',
        linestyle='-',
        marker='.',
        return_ax=False,
        c_lambda=1.0,  # FIXME: replace with unit conversions
        **kwargs
    ):
        if ax is None:
            f, ax = plt.subplots()
        # end if
        xdata = self.grid
        ydata = self.values
        xmin = xdata.min()
        xmax = xdata.max()
        ymin = ydata.min()
        xlen = xmax - xmin
        xlims = [xmin - xlen / 8, xmax + xlen / 8]
        xllims = [xmin + xlen / 8, xmax - xlen / 8]
        xgrid = linspace(xlims[0], xlims[1], 201)
        xlgrid = linspace(xllims[0], xllims[1], 201)
        ydata = self.values
        edata = self.errors
        x0 = self.x0 if self.x0 is not None else 0
        y0 = self.y0 if self.x0 is not None else ymin
        x0e = self.x0_err
        y0e = self.y0_err
        # plot lambda
        if self.Lambda is not None:
            a = self.sgn * self.Lambda / 2 * c_lambda
            pfl = [a, -2 * a * x0, y0 + a * x0**2]
            stylel_args = {'color': color, 'linestyle': ':'}  # etc
            ax.plot(xlgrid, polyval(pfl, xlgrid), **stylel_args)
        # end if
        # plot the line-search data
        style1_args = {'color': color,
                       'linestyle': 'None', 'marker': marker}  # etc
        style2_args = {'color': color,
                       'linestyle': linestyle, 'marker': 'None'}
        if edata is None or all(equal(array(edata), None)):
            ax.plot(xdata, ydata, **style1_args)
        else:
            ax.errorbar(xdata, ydata, edata, **style1_args)
        # end if
        ax.errorbar(x0, y0, y0e, xerr=x0e, marker='x', color=color)
        ax.plot(xgrid, self.val_data(xgrid), **style2_args)
        if return_ax:
            return ax
        # end if
    # end def

    def __str__(self):
        string = LineSearchBase.__str__(self)
        if self.Lambda is not None:
            string += '\n  Lambda: {:<9f}'.format(self.Lambda)
        # end if
        if self.W is not None:
            string += '\n  W: {:<9f}'.format(self.W)
        # end if
        if self.R is not None:
            string += '\n  R: {:<9f}'.format(self.R)
        # end if
        return string
    # end def

# end class
