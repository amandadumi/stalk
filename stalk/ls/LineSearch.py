from numpy import array, linspace, polyval, sign, equal
from matplotlib import pyplot as plt

from stalk.params.PesFunction import PesFunction
from stalk.params.ParameterSet import ParameterSet
from stalk.ls.LineSearchBase import LineSearchBase


# Class for PES line-search in structure context
class LineSearch(LineSearchBase):
    d = None  # direction count
    W = None
    R = None
    M = None
    Lambda = None
    sigma = None

    def __init__(
        self,
        structure=None,
        hessian=None,
        d=0,
        sigma=0.0,
        grid=None,
        **kwargs,
    ):
        self.sigma = sigma if sigma is not None else 0.0
        self.d = d
        if structure is not None:
            self.set_structure(structure)
        # end if
        if hessian is not None:
            self.set_hessian(hessian)
            self.figure_out_grid(grid=grid, **kwargs)
            LineSearchBase.__init__(self, sgn=self.sgn, **kwargs)
        else:
            LineSearchBase.__init__(self, **kwargs)
        # end if
    # end def

    def set_structure(self, structure):
        assert isinstance(structure, ParameterSet), 'provided structure is not a ParameterSet object'
        assert structure.check_consistency(), 'Provided structure is not a consistent mapping'
        self.structure = structure
    # end def

    def set_hessian(self, hessian):
        self.hessian = hessian
        Lambda = hessian.get_lambda(self.d)
        self.Lambda = abs(Lambda)
        self.sgn = sign(Lambda)
        self.direction = hessian.get_directions(self.d)
    # end def

    def figure_out_grid(self, **kwargs):
        offsets, self.M = self._figure_out_grid(**kwargs)
        grid = []
        for offset in offsets:
            structure = self._shift_structure(offset)
            grid.append(structure)
        # end for
        self.grid = grid
        self.shifted = True
    # end def

    def _figure_out_grid(self, M=None, W=None, R=None, grid=None, **kwargs):
        if M is None:
            M = self.M if self.M is not None else 7  # universal default
        # end if
        if grid is not None:
            self.M = len(grid)
        elif R is not None:
            assert not R < 0, 'R cannot be negative, {} requested'.format(R)
            grid = self._make_grid_R(R, M=M)
            self.R = R
        elif W is not None:
            assert not W < 0, 'W cannot be negative, {} requested'.format(W)
            grid = self._make_grid_W(W, M=M)
            self.W = W
        else:
            raise AssertionError('Must characterize grid')
        # end if
        return grid, M
    # end def

    def _make_grid_R(self, R, M):
        R = max(R, 1e-4)
        grid = linspace(-R, R, M)
        return grid
    # end def

    def _make_grid_W(self, W, M):
        R = self._W_to_R(max(W, 1e-4))
        return self._make_grid_R(R, M=M)
    # end def

    def _W_to_R(self, W):
        """Map W to R"""
        R = (2 * W / self.Lambda)**0.5
        return R
    # end def

    def _R_to_W(self, R):
        """Map R to W"""
        W = 0.5 * self.Lambda * R**2
        return W
    # end def

    def add_shift(self, shift):
        structure = self._shift_structure(shift)
        self.add_point(structure)
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
        self.search()
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
        return array([structure.params for structure in self._grid if isinstance(structure, ParameterSet)])
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
