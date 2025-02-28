#!/usr/bin/env python3
'''TargetParallelLinesearch class for assessment of parallel mixed errors

This is the surrogate model used to inform and optimize a parallel line-search.
'''

from numpy import array, isscalar, mean, linspace, argmin, where, isnan

from stalk.util import get_fraction_error
from stalk.ls import TargetLineSearch
from stalk.pls import ParallelLineSearch

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


class TargetParallelLineSearch(ParallelLineSearch):

    ls_type = TargetLineSearch
    _ls_list: list[TargetLineSearch] = []
    epsilon_p = None
    error_p = None
    error_d = None
    _temperature = None
    window_frac = None

    # Return a list of all line-searches
    @property
    def ls_list(self):
        return self._ls_list
    # end def

    @property
    def epsilon_d(self):
        return [tls.epsilon for tls in self.ls_list]
    # end def

    @property
    def optimized(self):
        return len(self) > 0 and all([tls.optimized for tls in self.ls_list])
    # end def

    @property
    def x_targets(self):
        return all([tls.target_fit.x0 for tls in self.ls_list])
    # end def

    @x_targets.setter
    def x_targets(self, targets):
        for x0, ls in zip(targets, self.ls_list):
            ls.target_fit.x0 = x0
        # end for
    # end def

    @property
    def temperature(self):
        return self._temperature
    # end def

    @temperature.setter
    def temperature(self, temperature):
        if temperature is None:
            self._temperature = None
        elif isscalar(temperature) and temperature > 0:
            self._temperature = temperature
        else:
            raise ValueError("Temperature must be positive.")
        # end if
    # end def

    @property
    def statistical_cost(self):
        return sum([ls.statistical_cost() for ls in self.ls_list])
    # end def

    @property
    def M(self):
        return [ls.M for ls in self.ls_list]
    # end def

    def __init__(
        self,
        structure=None,
        hessian=None,
        targets=None,
        **pls_args
        # windows=None, window_frac=0.25, noises=None, add_sigma=False, no_eval=False
        # pes=None, pes_func=None, pes_args={}, loader=None, load_func=None, load_args={}
        # M=7, fit_kind='pf3', fit_func=None, fit_args={}, N=200, Gs=None, fraction=0.025
    ):
        ParallelLineSearch.__init__(
            self,
            structure=structure,
            hessian=hessian,
            **pls_args
        )
        if targets is not None:
            self.x_targets = targets
        # end if
    # end def

    def optimize(
        self,
        reoptimize=True,
        windows=None,
        noises=None,
        epsilon_p=None,
        epsilon_d=None,
        temperature=None,
        **ls_args
        # noise_frac=0.1
        # M=7, fit_kind='pf3', fit_func=None, fit_args={}, N=200, Gs=None, fraction=0.025
    ):
        if self.evaluated:
            raise AssertionError('Cannot optimize before data has been evaluated.')
        # end if
        if self.optimized and not reoptimize:
            raise AssertionError('Already optimized, use reoptimize = True to reoptimize.')
        # end if
        if windows is not None and noises is not None:
            # No optimization necessary if the windows, noises are readily provided
            self.windows = windows
            self.noises = noises
        elif temperature is not None:
            self.optimize_temperature(temperature, **ls_args)
        elif epsilon_d is not None:
            self.optimize_epsilon_d(epsilon_d, **ls_args)
        elif epsilon_p is not None:
            self.optimize_epsilon_p(epsilon_p, **ls_args)
        else:
            raise AssertionError('Optimizer constraint not identified')
        # end if
        # Finalize and store the result
        self._finalize_optimization(windows, noises, **ls_args)
    # end def

    # Finalize optimization by computing error estimates
    def _finalize_optimization(self):
        errors = self._resample_errors()
        self.error_d, self.error_p = errors
    # end def

    def _errors_windows_noises(self, windows, noises, Gs=None, fit_kind=None, M=None, **kwargs):
        Gs_d = Gs if Gs is not None else self.D * [None]
        return self._resample_errors(windows, noises, Gs=Gs_d, M=M, fit_kind=fit_kind, **kwargs)
    # end def

    def optimize_epsilon_d(
        self,
        epsilon_d,
        **kwargs
        # Gs=None, N=500, M=7, fraction=0.025, fit_kind=None, fit_func=None,
        # fit_args={}, bias_mix=0.0, bias_order=1, noise_frac=0.05,
        # W_resolution=0.05, S_resolution=0.05, max_rounds=10
        # W_num=3, W_max=None, sigma_num=3, sigma_max=None
    ):
        for epsilon, ls in zip(epsilon_d, self.ls_list):
            ls.optimize(epsilon, **kwargs)
        # end for
        self.epsilon_p = None  # will be updated later if relevant
    # end def

    def optimize_temperature(
        self,
        temperature,
        **kwargs
        # Gs=None, N=500, M=7, fraction=0.025, fit_kind=None, fit_func=None,
        # fit_args={}, bias_mix=0.0, bias_order=1, noise_frac=0.05,
        # W_resolution=0.05, S_resolution=0.05, max_rounds=10
        # W_num=3, W_max=None, sigma_num=3, sigma_max=None
    ):
        self.temperature = temperature
        epsilon_d = self._get_thermal_epsilon_d(temperature)
        self.optimize_epsilon_d(epsilon_d, **kwargs)
    # end def

    def optimize_epsilon_p(
        self,
        epsilon_p,
        starting_mix=0.5,
        thermal=False,
        **kwargs,
        # Gs=None, N=500, M=7, fraction=0.025, fit_kind=None, fit_func=None,
        # fit_args={}, bias_mix=0.0, bias_order=1, noise_frac=0.05,
        # W_resolution=0.05, S_resolution=0.05, max_rounds=10
        # W_num=3, W_max=None, sigma_num=3, sigma_max=None
    ):
        epsilon_p = array(epsilon_p, dtype=float)
        if thermal:
            epsilon_d_opt = self._optimize_epsilon_p_thermal(
                epsilon_p,
                **kwargs
            )
        else:
            epsilon_d_opt = self._optimize_epsilon_p_ls(
                epsilon_p,
                starting_mix=starting_mix
            )
        # end if
        self.optimize_epsilon_d(epsilon_d_opt, **kwargs)
        self.epsilon_p = epsilon_p
    # end def

    # TODO: check for the first step
    def _optimize_epsilon_p_thermal(self, epsilon_p, T0=0.00001, dT=0.000005, verbose=False, **kwargs):
        # init
        T = T0
        error_p = array([-1, -1])
        # First loop: increase T until the errors are no longer capped
        while all(error_p[where(~isnan(error_p))] < 0.0):
            try:
                epsilon_d = self._get_thermal_epsilon_d(T)
                error_p = self._resample_errors_p_of_d(
                    epsilon_d, target=epsilon_p, verbose=verbose, **kwargs)
                error_frac = (error_p + epsilon_p) / epsilon_p
                if verbose:
                    print('T = {} highest error {} %'.format(T, error_frac * 100))
                # end if
            except AssertionError:
                if verbose:
                    print('T = {} skipped'.format(T))
                # end if
            # end try
            T *= 1.5
        # end while
        # Second loop: decrease T until the errors are capped
        while not all(error_p[where(~isnan(error_p))] < 0.0):
            T *= 0.95
            epsilon_d = self._get_thermal_epsilon_d(T)
            error_p = self._resample_errors_p_of_d(
                epsilon_d, target=epsilon_p, verbose=verbose, **kwargs)
            error_frac = (error_p + epsilon_p) / epsilon_p
            if verbose:
                print('T = {} highest error {} %'.format(T, error_frac * 100))
            # end if
        # end while
        return self._get_thermal_epsilon_d(T)
    # end def

    def _optimize_epsilon_p_ls(
        self,
        epsilon_p,
        thr=None,
        it_max=10,
        starting_mix=0.5,
        **kwargs
    ):
        thr = thr if thr is not None else mean(epsilon_p) / 10
        U = self.hessian.get_directions()
        epsilon_d0 = starting_mix * abs(U @ epsilon_p) + (1 - starting_mix) * U.T @ U @ epsilon_p

        def cost(derror_p):
            return sum(derror_p**2)**0.5
        # end def

        epsilon_d_opt = array(epsilon_d0)
        for it in range(it_max):
            coeff = 0.5**(it + 1)
            epsilon_d_old = epsilon_d_opt.copy()
            # sequential line-search from d0...dD
            for d in range(len(epsilon_d_opt)):
                epsilon_d = epsilon_d_opt.copy()
                epsilons = linspace(epsilon_d[d] * (1 - coeff), (1 + coeff) * epsilon_d[d], 6)
                costs = []
                # TODO: change to actual line-search
                for s in epsilons:
                    epsilon_d[d] = s
                    derror_p = self._resample_errors_p_of_d(
                        epsilon_d, target=epsilon_p, **kwargs)
                    costs.append(cost(derror_p))
                # end for
                epsilon_d_opt[d] = epsilons[argmin(costs)]
            # end for
            derror_p = self._resample_errors_p_of_d(
                epsilon_d_opt, target=epsilon_p, **kwargs)
            cost_it = cost(derror_p)
            if cost_it < thr or sum(abs(epsilon_d_old - epsilon_d_opt)) < thr / 10:
                break
            # end if
        # end for
        # scale down
        for c in range(100):
            if any(derror_p > 0.0):
                epsilon_d_opt = [e * 0.999 for e in epsilon_d_opt]
                derror_p = self._resample_errors_p_of_d(
                    epsilon_d_opt, target=epsilon_p, **kwargs)
            else:
                break
            # end if
        # end for
        return epsilon_d_opt
    # end def

    @property
    def Gs(self):
        return [ls.Gs for ls in self.ls_list]
    # end def

    def validate(self, N=500, thr=1.1):
        """Validate optimization by independent random resampling
        """
        if not self.optimized:
            raise AssertionError('Must be optimized first')
        # end if
        ref_error_p, ref_error_d = self._resample_errors(
            self.windows, self.noises, Gs=None, N=N)
        valid = True
        for p, ref, corr in zip(range(len(ref_error_p)), ref_error_p, self.error_p):
            ratio = ref / corr
            valid_this = ratio < thr
            valid = valid and valid_this
        # end for
        for d, ref, corr in zip(range(len(ref_error_d)), ref_error_d, self.error_d):
            ratio = ref / corr
            valid_this = ratio < thr
            valid = valid and valid_this
        # end for
        return valid
    # end def

    def _get_thermal_epsilon_d(self, temperature):
        return [(temperature / abs(Lambda))**0.5 for Lambda in self.Lambdas]
    # end def

    def _get_thermal_epsilon(self, temperature):
        return [(temperature / abs(Lambda))**0.5 for Lambda in self.hessian.hessian.diagonal]
    # end def

    def compute_bias_p(self, **kwargs):
        return self.compute_bias(**kwargs)[1]
    # end def

    def compute_bias_d(self, **kwargs):
        return self.compute_bias(**kwargs)[0]
    # end def

    def compute_bias(self, windows=None, **kwargs):
        windows = windows if windows is not None else self.windows
        return self._compute_bias(windows, **kwargs)
    # end def

    def _compute_bias(self, windows, **kwargs):
        bias_d = []
        for W, tls in zip(windows, self.ls_list):
            assert W <= tls.W_max, 'window is larger than W_max'
            grid = tls._figure_out_grid(W=W)[0]
            bias_d.append(tls.compute_bias(grid, **kwargs)[0])
        # end for
        bias_d = array(bias_d)
        bias_p = self._calculate_params_next(
            self.get_params(), self.get_directions(), bias_d) - self.get_params()
        return bias_d, bias_p
    # end def

    # based on windows, noises
    def _resample_errorbars(self):
        # provide correlated sampling
        fraction = self.ls_list[0].fraction
        biases_d, biases_p = self._compute_bias()  # biases per direction, parameter
        x0s_d, x0s_p = [], []  # list of distributions of minima per direction, parameter
        # list of statistical errorbars per direction, parameter
        errorbar_d, errorbar_p = [], []
        for tls, W, noise, bias_d, Gs in zip(self.ls_list, self.windows, self.noises, biases_d):
            x0s = tls.fit_func_opt.get_x0_distribution(Gs=tls.Gs)
            x0s_d.append(x0s)
            errorbar_d.append(get_fraction_error(x0s - bias_d, fraction)[1])
        # end for
        # parameter errorbars
        for x0 in array(x0s_d).T:
            x0s_p.append(self._calculate_params_next(
                -biases_p,
                self.hessian.get_directions(),
                x0)
            )
        # end for
        errorbar_p = [get_fraction_error(x0s, fraction)[1] for x0s in array(x0s_p).T]
        return array(errorbar_d), array(errorbar_p)
    # end def

    def _resample_errors(self):
        bias_d, bias_p = self._compute_bias()
        errorbar_d, errorbar_p = self._resample_errorbars()
        error_d = abs(bias_d) + errorbar_d
        error_p = abs(bias_p) + errorbar_p
        return error_d, error_p
    # end def

    def _resample_errors_p_of_d(self, epsilon_d, target, **kwargs):
        windows, noises = self._windows_noises_of_epsilon_d(
            epsilon_d, **kwargs)
        return self._resample_errors(windows, noises, **kwargs)[1] - target
    # end def

    def _windows_noises_of_epsilon_d(
        self,
        epsilon_d,
        **kwargs,
    ):
        windows, noises = [], []
        for epsilon, ls, in zip(epsilon_d, self.ls_list):
            W_opt, sigma_opt = ls.maximize_sigma(
                epsilon, **kwargs)  # no altering the error
            windows.append(W_opt)
            noises.append(sigma_opt)
        # end for
        return windows, noises
    # end def

    def plot_error_surfaces(self, **kwargs):
        for ls in self.ls_list:
            ls.plot_error_surface(**kwargs)
        # end for
    # end def

# end class
