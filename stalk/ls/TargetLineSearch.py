#!/usr/bin/env python3
'''TargetLineSearch classes for the assessment and evaluation of fitting errors
'''

from numpy import array, argsort, isscalar, linspace, append, nan, isnan, where
from numpy import random, argmin, ndarray

from stalk.ls.FittingFunction import FittingFunction
from stalk.ls.LineSearchGrid import LineSearchGrid
from .LineSearch import LineSearch
from .TargetLineSearchBase import TargetLineSearchBase

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# Class for error scan line-search
class TargetLineSearch(TargetLineSearchBase, LineSearch):

    _Gs: None | ndarray = None  # N x M set of correlated random fluctuations for the grid
    _epsilon = None  # optimized target error
    _W_opt = None  # W to meet epsilon
    _sigma_opt = None  # sigma to meet epsilon
    _fit_func_opt = None  # Fit function to meet epsilon
    E_mat = None  # resampled W-sigma matrix of errors
    W_mat = None  # resampled W-mesh
    S_mat = None  # resampled sigma-mesh
    T_mat = None  # resampled trust-mesh (whether error is reliable)
    target_bias_order = 1
    target_bias_mix = 0.0

    def __init__(
        self,
        # kwargs related to LineSearch
        structure=None,
        hessian=None,
        d=None,
        # sigma=0.0
        offsets=None,
        M=7,
        W=None,
        R=None,
        pes=None,
        # kwargs related to TargetLineSearchBase
        bias_order=1,
        bias_mix=0.0,
        interpolate_kind='cubic',
        # Line-search kwargs
        **ls_args
        # values=None, errors=None, fraction=0.025, sgn=1
        # fit_kind='pf3', fit_func=None, fit_args={}, N=200, Gs=None
    ):
        # Set bias_mix and target_fit in TargetLineSearchBase
        TargetLineSearchBase.__init__(
            self,
            bias_mix=bias_mix,
            bias_order=bias_order
        )
        # The necessary init is done in LineSearch class
        LineSearch.__init__(
            self,
            structure=structure,
            hessian=hessian,
            d=d,
            M=M,
            W=W,
            R=R,
            offsets=offsets,
            pes=pes,
            **ls_args,
        )
        # Finally, attempt to reset interpolation
        if self.valid:
            self.reset_interpolation(interpolate_kind=interpolate_kind)
        # end if
    # end def

    @property
    def M(self):
        if self.Gs is not None:
            return self.Gs.shape[1]
        else:
            return len(self)
        # end if
    # end def

    @property
    def N(self):
        if self.Gs is not None:
            return self.Gs.shape[0]
        else:
            return 0
        # end if
    # end def

    @property
    def sigma_opt(self):
        return self._sigma_opt
    # end def

    @sigma_opt.setter
    def sigma_opt(self, sigma_opt):
        if isscalar(sigma_opt) and sigma_opt > 0.0:
            self._sigma_opt = sigma_opt
    # end def

    @property
    def W_opt(self):
        return self._W_opt
    # end def

    @W_opt.setter
    def W_opt(self, W_opt):
        if isscalar(W_opt) and W_opt > 0.0:
            self._W_opt = W_opt
        # end if
    # end def

    @property
    def fit_func_opt(self):
        return self._fit_func_opt
    # end def

    @fit_func_opt.setter
    def fit_func_opt(self, fit_func):
        if isinstance(fit_func, FittingFunction):
            self._fit_func_opt = fit_func
        # end if
    # end def

    @property
    def R_opt(self):
        return self._W_to_R(self.W_opt)
    # end def

    @property
    def Gs(self):
        return self._Gs
    # end def

    @property
    def Ws(self):
        if self.resampled:
            return self.W_mat[0]
        else:
            return None
        # end if
    # end def

    @property
    def sigmas(self):
        if self.resampled:
            return self.S_mat[:, 0]
        else:
            return None
        # end if
    # end def

    @property
    def resampled(self):
        return self.E_mat is not None
    # end def

    @property
    def optimized(self):
        return self.W_opt is not None and self.sigma_opt is not None
    # end def

    @Gs.setter
    def Gs(self, Gs):
        if isinstance(Gs, ndarray) and Gs.ndim == 2:
            if Gs.shape[0] < 2:
                raise AssertionError('Gs must have at least two rows')
            elif Gs.shape[1] < 3:
                raise AssertionError('Gs must have at least two columns')
            else:
                self._Gs = Gs
                # When new Gs is set
                self._reset_resampling()
            # end if
        else:
            raise ValueError('Gs must be a 2D array')
        # end if
    # end def

    def _reset_resampling(self):
        self.E_mat = None
        self.S_mat = None
        self.W_mat = None
        self.T_mat = None
    # end def

    def compute_bias_of(
        self,
        M=None,
        R=None,
        W=None,
        num_W=10,  # number of W points if no grids are provided
        **ls_args  # sgn, fraction, fit_func, N, Gs, bias_order, bias_mix
    ):
        M = M if M is not None else self.M

        Rs = []
        Ws = []
        if R is None:
            if W is None:
                # By default, compute bias for a range of W values
                for W in linspace(0.0, self.valid_W_max, num_W):
                    Ws.append(W)
                    Rs.append(self._W_to_R(W))
                # end for
            elif isscalar(W):
                Ws.append(W)
                Rs.append(self._W_to_R(W))
            else:
                Ws = W
                Rs = [self._W_to_R(W) for W in Ws]
            # end if
        else:
            if isscalar(R):
                Ws.append(self._R_to_W(R))
                Rs.append(R)
            else:
                Rs = R
                Ws = [self._R_to_W(R) for R in Rs]
            # end if
        # end if

        biases = []
        for W in Ws:
            offsets = self.figure_out_adjusted_offsets(M=M, W=W)
            values = self.evaluate_target(offsets)
            bias = self.compute_bias(
                grid=LineSearchGrid(offsets=offsets, values=values),
                **ls_args
            )
            biases.append(bias)
        # end for
        return array(Ws), array(Rs), array(biases)
    # end def

    def figure_out_adjusted_offsets(
        self,
        **grid_args  # M, R, W, offsets
    ):
        if self.fit_res is None:
            raise AssertionError('Must have fit_res before adjusting offsets')
        # end if
        return self.figure_out_offsets(**grid_args) + self.target_fit.x0
    # end def

    # X: grid of W values; Y: grid of sigma values; E: grid of total errors
    #   if Gs is not provided, use M and N
    def generate_error_surface(
        self,
        W_num=3,
        W_max=None,
        sigma_num=3,
        sigma_max=None,
        noise_frac=0.05,
        Gs=None,
        M=None,
        N=None,
        bias_mix=None,
        bias_order=None,
    ):
        if not self.valid_target:
            raise AssertionError("Must have valid target data before generating error.")
        # end if
        # This will set self.Gs -> self.M, self.N
        self._regenerate_Gs(N=N, M=M, Gs=Gs)

        W_max = W_max if W_max is not None else self.W_max
        sigma_max = sigma_max if sigma_max is not None else W_max * noise_frac
        if bias_mix is not None:
            self.target_bias_mix = bias_mix
        # end if
        if bias_order is not None:
            self.target_bias_order = bias_order
        # end if

        if W_max <= 0.0:
            raise ValueError('Must provide W_max > 0')
        # end if
        if sigma_max <= 0.0:
            raise ValueError('Must provide sigma_max > 0')
        # end if

        # Initial W and sigma grids
        Ws = linspace(0.0, W_max, W_num)
        sigmas = linspace(0.0, sigma_max, sigma_num)
        # Start from adding the first row: sigma=0 -> plain bias
        E0_row = [self._compute_target_error(W, sigmas[0]) for W in Ws]
        self.E_mat = array([E0_row])
        self.W_mat = array([Ws])
        self.S_mat = array([W_num * [sigmas[0]]])
        self.T_mat = self._generate_T_mat()
        # Then, append the noisy rows
        for sigma in sigmas[1:]:
            self.insert_sigma_data(sigma)
        # end for
    # end def

    # Compute fitting bias and error using consistent parameters that are stored in Gs and
    # target_fit
    def _compute_target_error(self, W, sigma):
        offsets = self.figure_out_adjusted_offsets(W=W, M=self.M)
        values = self.evaluate_target(offsets)
        errors = self.M * [sigma]
        grid = LineSearchGrid(offsets=offsets, values=values, errors=errors)
        error = self.compute_error(
            grid,
            Gs=self.Gs,
            fit_func=self.fit_func,
            sgn=self.sgn,
            fraction=self.target_fit.fraction,
            bias_order=self.target_bias_order,
            bias_mix=self.target_bias_mix
        )
        return error
    # end def

    def _generate_T_mat(self):
        return self.W_mat >= self.S_mat
    # end def

    # Return true if a different resampling than current is requested
    def _check_resampling_changed(
        self,
        M=None,
        N=None,
        Gs=None,
        fit_func=None,
        bias_order=None,
        bias_mix=None,
        **kwargs
    ):
        result = Gs is not None
        result |= M is not None and M != self.M
        result |= N is not None and N != self.N
        result |= fit_func is not None and fit_func != self.fit_func
        result |= bias_mix is not None and bias_mix != self.target_bias_mix
        result |= bias_order is not None and bias_order != self.target_bias_order
        return result
    # end def

    # Reset or regenerate Gs if the user changes M or N or provides new Gs
    def _regenerate_Gs(self, M=None, N=None, Gs=None):
        if Gs is not None:
            # If Gs provided, try to insert them and ignore other arguments, return False
            self.Gs = Gs
        else:
            M = M if M is not None else self.M
            N = N if N is not None else self.N
            # If M or N changes, regenerate
            if not isinstance(M, int) or M <= 2:
                raise ValueError('Must provide M > 2')
            # end if
            if not isinstance(N, int) or N <= 1:
                raise ValueError('Must provide N > 1')
            # end if
            if M != self.M and N != self.N:
                self.Gs = random.randn(N, M)
            # end if
        # end if
    # end def

    def insert_sigma_data(self, sigma):
        if not (self.resampled and isscalar(sigma) and sigma > 0):
            raise AssertionError('Must have resampled data and scalar sigma > 0')
        # end if
        E_row = [self._compute_target_error(W, sigma) for W in self.Ws]
        W_mat = append(self.W_mat, [self.Ws], axis=0)
        S_mat = append(self.S_mat, [len(self.Ws) * [sigma]], axis=0)
        E_mat = append(self.E_mat, [E_row], axis=0)
        idx = argsort(S_mat[:, 0])
        self.W_mat = W_mat[idx]
        self.S_mat = S_mat[idx]
        self.E_mat = E_mat[idx]
        self.T_mat = self._generate_T_mat()
    # end def

    def insert_W_data(self, W):
        if not (self.resampled and isscalar(W) and W > 0 and W < self.W_max):
            raise AssertionError('Must have resampled data and scalar 0 < W < W_max')
        # end if
        sigmas = self.S_mat[:, 0]
        E_col = [self._compute_target_error(W, sigma) for sigma in sigmas]
        W_mat = append(self.W_mat, array([len(sigmas) * [W]]).T, axis=1)
        S_mat = append(self.S_mat, array([sigmas]).T, axis=1)
        E_mat = append(self.E_mat, array([E_col]).T, axis=1)
        idx = argsort(W_mat[0])
        self.W_mat = W_mat[:, idx]
        self.S_mat = S_mat[:, idx]
        self.E_mat = E_mat[:, idx]
        self.T_mat = self._generate_T_mat()
    # end def

    def optimize(
        self,
        epsilon,
        fit_kind='pf3',
        fit_func=None,
        fit_args=None,
        fraction=None,
        **kwargs
        # generate_args: W_num, W_max, sigma_num, sigma_max, noise_frac, M, N, Gs
        # bias_mix, bias_order
        # maximize_sigma_args: W_resolution, S_resolution, max_rounds
    ):
        """Optimize W and sigma to a given target error epsilon > 0."""
        if not self.valid_target:
            raise AssertionError("Must have valid target data before optimization.")
        # end if
        # Set the fit function as requested
        self.set_fit_func(
            fit_func=fit_func,
            fit_kind=fit_kind,
            fit_args=fit_args
        )
        if fraction is not None:
            # Set fraction as requested
            self.target_fit.fraction
        # end if
        try:
            self.W_opt, self.sigma_opt = self.maximize_sigma(
                epsilon,
                **kwargs
            )
            self.fit_func_opt = fit_func
            self.epsilon = epsilon
        except AssertionError:
            print('Optimization failed!')
        # end try
    # end def

    # Optimize W and sigma based on maximizing sigma.
    def maximize_sigma(
        self,
        epsilon,
        W_resolution=0.05,  # W resolution fraction of the error surface
        S_resolution=0.05,  # S resolution fraction of the error surface
        max_rounds=10,  # maximum number of rounds
        **generate_args
        # W_num, W_max, sigma_num, sigma_max, noise_frac, Gs, M, N, bias_mix, bias_order
    ):
        if not isscalar(epsilon) or epsilon <= 0.0:
            raise ValueError("Must provide epsilon > 0.")
        # end if
        if W_resolution >= 1.0 or W_resolution <= 0.0:
            raise ValueError('W resolution must be 0.0 < W_resolution < 1.0')
        # end if
        if S_resolution >= 1.0 or S_resolution <= 0.0:
            raise ValueError('S resolution must be 0.0 < S_resolution < 1.0')
        # end if
        if max_rounds <= 0:
            raise ValueError('Must provide max_rounds > 0')
        # end if
        # Make sure error surface is generated
        if self.E_mat is None or self._check_resampling_changed(**generate_args):
            self.generate_error_surface(**generate_args)
        # end if

        # Find W and sigma that maximize sigma
        for round in range(max_rounds):
            xi, yi = self._argmax_y(self.E_mat, self.T_mat, epsilon)
            if self._treat_errors(xi, yi, epsilon, W_resolution, S_resolution):
                break
            # end if
        # end while
        return self.W_mat[yi, xi], self.S_mat[yi, xi]
    # end def

    # Treat various accuracy errors by adding resolution inside boundaries
    # When new points are added, the opeations return False; if none are added
    # the operations return True and the loop breaks.
    def _treat_errors(self, xi, yi, epsilon, X_resolution, Y_resolution):
        if isnan(xi) or isnan(yi):
            msg = 'Could not find the target error. '
            msg += f'The smallest epsilon is {self.E_mat.min()} and {epsilon} was requested.'
            raise AssertionError(msg)
        # end if
        result = True
        if xi == 0:
            result &= self._treat_x_underflow(X_resolution)
        elif xi == self.E_mat.shape[1] - 1:
            result &= self._treat_x_overflow(xi, X_resolution)
        else:
            result &= self._treat_x_res(xi, X_resolution)
        # end if

        if yi == 0:
            result &= self._treat_y_underflow(Y_resolution)
        elif yi == self.E_mat.shape[0] - 1:
            result &= self._treat_y_overflow(yi, Y_resolution)
        else:
            result &= self._treat_y_res(yi, Y_resolution)
        # end if
        return result
    # end def

    # Fix x underflow by adding a new W value between the first and second
    def _treat_x_underflow(self, X_resolution):
        W_max = self.W_mat[0, -1]
        # The first W is (almost) zero
        W_this = self.W_mat[0, 0]
        W_right = (self.W_mat[0, 0] + self.W_mat[0, 1]) / 2
        W_diff = (W_right - W_this) / W_max
        if W_diff > X_resolution:
            self.insert_W_data(W_right)
            return False
        else:
            # Else, print a recommendation
            msg = f'Persistent W-underflow in tls{self.d}:'
            msg += f' To improve performance, sample low W near {W_this}'
            print(msg)
            return True
        # end if
    # end def

    def _treat_x_overflow(self, xi, X_resolution):
        W_max = self.W_mat[0, -1]
        W_this = self.W_mat[0, xi]
        W_left = self.W_mat[0, xi - 1]
        W_diff = (W_this - W_left) / W_max
        if W_diff > X_resolution:
            # Add new W value to the left
            W_new = (W_this + W_left) / 2
            self.insert_W_data(W_new)
            return False
        else:
            # Else, print a recommendation
            msg = f'Persistent W-overflow in tls{self.d}:'
            msg += f' To improve performance, sample W > {W_this}'
            print(msg)
            return True
        # end if
    # end def

    def _treat_x_res(self, xi, X_resolution):
        W_max = self.W_mat[0, -1]
        W_this = self.W_mat[0, xi]
        W_left = self.W_mat[0, xi - 1]
        W_right = self.W_mat[0, xi + 1]
        res = True
        if (W_this - W_left) / W_max / 2 > X_resolution:
            # Add new W value to the left
            self.insert_W_data((W_this + W_left) / 2)
            res &= False
        # end if
        if (W_right - W_this) / W_max / 2 > X_resolution:
            # Add new W value to the right
            self.insert_W_data((W_this + W_right) / 2)
            res &= False
        # end if
        return res
    # end def

    # Fix x underflow by adding a new sigma value between the first and second
    def _treat_y_underflow(self, Y_resolution):
        S_max = self.S_mat[-1, 0]
        S_this = self.S_mat[1, 0]  # should be zero
        S_up = self.S_mat[1, 0] / 2
        S_diff = (S_up - S_this) / S_max
        if S_diff > Y_resolution:
            self.insert_sigma_data(S_up)
            return False
        else:
            # Else, print a recommendation
            msg = f'Persistent Sigma underflow in tls{self.d}:'
            msg += f' To improve performance, check Y_resolution: {Y_resolution}'
            print(msg)
            return True
        # end if
    # end def

    # Fix x underflow by adding a new sigma value twice as high until W_max
    def _treat_y_overflow(self, S_this, Y_resolution):
        S_this = self.S_mat[-1, 0]
        S_up = max(2 * S_this, self.W_max)
        S_diff = (S_up - S_this) / S_up
        if S_diff > Y_resolution:
            self.insert_sigma_data(S_up)
            return False
        else:
            msg = f'Could not add sigma higher than {S_up}.'
            msg += f' To improve performance, sample W > {self.W_max}.'
            print(msg)
            return True
        # end if
    # end def

    def _treat_y_res(self, yi, Y_resolution):
        S_max = self.S_mat[-1, 0]
        S_this = self.S_mat[yi, 0]
        S_down = self.S_mat[yi - 1, 0]
        S_up = self.S_mat[yi + 1, 0]
        res = True
        if (S_up - S_this) / S_max / 2 > Y_resolution:
            self.insert_sigma_data((S_this + S_up) / 2)
            res &= False
        # end if
        if (S_this - S_down) / S_max / 2 > Y_resolution:
            self.insert_sigma_data((S_this + S_down) / 2)
            res &= False
        # end if
        return res
    # end def

    # Return the W and maximum sigma to meet epsilon and maximization errors
    def _maximize_y(self, epsilon):
        xi, yi = self._argmax_y(self.E_mat, self.T_mat, epsilon)
        E0 = self.E_mat[yi, xi]
        x0 = self.W_mat[yi, xi]
        y0 = self.S_mat[yi, xi]
        return x0, y0, E0, (xi, yi)
    # end def

    def _argmax_y(self, E, T, epsilon):
        """Return indices to the highest point in E matrix that is lower than epsilon"""
        xi, yi = nan, nan
        W = self.W_mat[0]
        for i in range(len(E), 0, -1):  # from high to low
            err = where((E[i - 1] < epsilon) & (T[i - 1]))
            if len(err[0]) > 0:
                yi = i - 1
                # xi = err[0][argmax(E[i - 1][err[0]])]
                # take the middle
                xi = err[0][argmin(
                    abs(W[err[0]] - (W[err[0][0]] + W[err[0][-1]]) / 2))]
                break
            # end if
        # end for
        return xi, yi
    # end def

    def statistical_cost(self):
        """Return statistical cost based on sigma and M"""
        if not self.optimized:
            raise AssertionError('Cannot compute statistical cost before optimization')
        # end if
        return self.M * self.sigma_opt**-2
    # end def

    def plot_error_surface(
        self,
        ax=None
    ):
        if not self.optimized:
            print('Must optimize before plotting error surface')
            return
        # end if
        from matplotlib import pyplot as plt
        if ax is None:
            f, ax = plt.subplots(1, 1)
        # end if
        T = self.T_mat
        X = self.W_mat
        Y = self.S_mat
        Z = self.E_mat
        Z[~T] = nan
        ax.contourf(X, Y, Z)
        ax.contour(X, Y, Z, [self.epsilon], colors='k')
        ax.plot(X.flatten(), Y.flatten(), 'k.', alpha=0.3)
        ax.plot(self.W_opt, self.sigma_opt, 'ko')
    # end def

# end class
