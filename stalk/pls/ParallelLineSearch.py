#!/usr/bin/env python3
'''ParallelLineSearch class for simultaneous linesearches along conjugate directions'''

from numpy import ndarray, array
from textwrap import indent

from stalk.util import get_fraction_error
from stalk.params import ParameterSet
from stalk.params import ParameterHessian
from stalk.ls import LineSearch
from .PesSampler import PesSampler

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# Class for a bundle of parallel line-searches
class ParallelLineSearch(PesSampler):

    ls_type = LineSearch
    _ls_list: list[LineSearch] = []  # list of line-search objects
    _hessian = None  # hessian object
    _structure = None  # eqm structure
    _structure_next = None  # next structure

    def __init__(
        self,
        # PLS arguments
        path='pls',
        hessian=None,
        structure=None,
        windows=None,
        window_frac=0.25,
        noises=None,
        add_sigma=False,
        no_eval=False,
        # PesSampler args
        pes=None,
        pes_func=None,
        pes_args={},
        load=None,  # eliminate loading arg
        # LineSearch args
        **ls_args
        # M=7, fit_kind='pf3', fit_func=None, fit_args={}, N=200, Gs=None, fraction=0.025
    ):
        if load is not None and self.pes is not None:
            # Proxies of successful loading from disk
            return
        # end if
        PesSampler.__init__(
            self,
            path=path,
            pes=pes,
            pes_func=pes_func,
            pes_args=pes_args
        )
        if structure is not None:
            self.structure = structure
        # end if
        if hessian is not None:
            self.hessian = hessian
        # end if
        if self.setup:
            self.initialize(
                windows,
                noises,
                window_frac,
                **ls_args
            )
            if self.shifted and not no_eval:
                # Successful evaluation leads to estimation of next structure
                self.evaluate(add_sigma=add_sigma)
            # end if
        # end if
    # end def

    def initialize(
        self,
        windows=None,
        noises=None,
        window_frac=None,
        **ls_args
        # M=7, fit_kind='pf3', fit_func=None, fit_args={}, N=200, Gs=None, fraction=0.025
    ):
        if windows is None:
            windows = abs(self.Lambdas)**0.5 * window_frac
        # end if
        if noises is None:
            noises = self.D * [0.0]
        # end if
        self._reset_ls_list(windows, noises, **ls_args)
    # end def

    def _reset_ls_list(
        self,
        windows,
        noises,
        **ls_args,
        # M=7, fit_kind='pf3', fit_func=None, fit_args={}, N=200, Gs=None, fraction=0.025
    ):
        ls_list = []
        for d, window, noise in zip(range(self.D), windows, noises):
            ls = self.ls_type(
                structure=self.structure,
                hessian=self.hessian,
                d=d,
                sigma=noise,
                W=window,
                **ls_args
            )
            ls_list.append(ls)
        # end for
        self._ls_list = ls_list
        # Reset next structure if re-initialized
        self._structure_next = None
    # end def

    # Return True if the parallel line-search has starting structure and Hessian
    @property
    def setup(self):
        return self.structure is not None and self.hessian is not None
    # end def

    # Return True if the line-search structures have been shifted
    @property
    def shifted(self):
        return len(self) > 0 and all([ls.shifted for ls in self.enabled_ls])
    # end def

    # Return a list of all line-searches
    @property
    def ls_list(self):
        return self._ls_list
    # end def

    @property
    def D(self):
        if self.hessian is None:
            return 0
        else:
            return len(self.hessian)
        # end if
    # ed def

    # Return a list of enabled line-searches
    @property
    def enabled_ls(self):
        return [ls for ls in self.ls_list if ls.enabled]
    # end def

    @property
    def evaluated(self):
        return len(self) > 0 and all([ls.evaluated for ls in self.ls_list if ls.enabled])
    # end def

    @property
    def hessian(self):
        return self._hessian
    # end def

    @hessian.setter
    def hessian(self, hessian):
        if isinstance(hessian, ndarray):
            hessian = ParameterHessian(hessian=hessian)
        elif not isinstance(hessian, ParameterHessian):
            raise ValueError('Hessian matrix is not supported')
        # end if
        if self._hessian is not None:
            pass  # TODO: check for constistency
        # end if
        self._hessian = hessian
        if self.structure is None:
            self.structure = hessian.structure
        # end if
        # TODO: propagate Hessian information to ls_list?
    # end def

    @property
    def structure(self):
        return self._structure
    # end def

    @structure.setter
    def structure(self, structure):
        if not isinstance(structure, ParameterSet):
            raise TypeError("Structure must be inherited from ParameterSet clas")
        # end if
        self._structure = structure.copy(label='eqm')
        # Upon change, reset line-searches according to old windows/noises, if present
        if self.shifted:
            windows = self.windows
            noises = self.noises
            self._reset_ls_list(windows, noises)
        # end if
    # end def

    @property
    def structure_next(self):
        return self._structure_next
    # end def

    @property
    def Lambdas(self):
        if self.hessian is None:
            return array([])
        else:
            return array(self.hessian.get_lambda())
        # end if
    # end def

    @property
    def windows(self):
        result = []
        for ls in self.ls_list:
            if isinstance(ls, LineSearch):
                window = ls.W_max
            else:
                window = None
            # end if
            result.append(window)
        # end if
        return result
    # end def

    @property
    def noises(self):
        return [ls.sigma for ls in self.ls_list]
    # end def

    def evaluate(self, add_sigma=False):
        if not self.shifted:
            raise AssertionError("Must have shifted structures first!")
        # end if
        structures, sigmas = self._collect_enabled()
        self._evaluate_energies(structures, sigmas, add_sigma=add_sigma)
        # Set the eqm energy
        for ls in self.ls_list:
            eqm = ls.find_point(0.0)
            if eqm is not None:
                self.structure.value = eqm.value
                self.structure.error = eqm.error
                break
            # end if
        # end for
        self._solve_ls()
        # Calculate next params
        params_next, params_next_err = self.calculate_next_params()  # **kwargs
        self._structure_next = self.structure.copy(
            params=params_next,
            params_err=params_next_err
        )
    # end def

    def _collect_enabled(self):
        structures = []
        sigmas = []
        for ls in self.enabled_ls:
            structures += ls.grid
            sigmas += len(ls) * [ls.sigma]
        # end for
        return structures, sigmas
    # end def

    def _solve_ls(self):
        for ls in self.enabled_ls:
            ls._search_and_store()
        # end for
    # end def

    @property
    def noisy(self):
        return any([ls.noisy for ls in self.enabled_ls])
    # end def

    @property
    def params(self):
        if self.structure is not None:
            return self.structure.params
        # end if
    # end def

    @property
    def params_err(self):
        if self.structure is not None:
            return self.structure.params_err
        # end if
    # end def

    def calculate_next_params(
        self,
        N=200,
        Gs=None,
        fraction=0.025
    ):
        # deterministic
        params = self.params
        shifts = self.shifts
        params_next = self._calculate_params_next(params, shifts)
        # stochastic
        if self.noisy:
            x0s = []
            for ls in self.ls_list:
                x0s.append(ls.settings.fit_func.get_x0_distribution(ls, N=N, Gs=Gs))
            # end if
            x0s = array(x0s).T
            dparams = []
            for shifts_this in x0s:
                dparams.append(
                    self._calculate_params_next(
                        params,
                        shifts_this
                    ) - params_next
                )
            # end for
            dparams = array(dparams).T
            params_next_err = array(
                [get_fraction_error(p, fraction=fraction)[1] for p in dparams]
            )
        else:
            params_next_err = array(self.D * [0.0])
        # end if
        return params_next, params_next_err
    # end def

    def ls(self, i) -> LineSearch:
        if i < 0 or i >= len(self.ls_list):
            raise ValueError("Must choose line-search between 0 and " + str(len(self.ls_list)))
        # end if
        return self.ls_list[i]
    # end def

    def _calculate_params_next(self, params, shifts):
        directions = self.hessian.get_directions()
        return params + shifts @ directions
    # end def

    @property
    def shifts(self):
        shifts = []
        for ls in self.ls_list:
            if ls.enabled:
                shift = ls.x0
            else:
                shift = 0.0
            # end if
            shifts.append(shift)
        # end for
        return array(shifts)
    # end def

    def copy(
        self,
        path,
        structure=None,
        hessian=None,
        windows=None,
        noises=None,
        pes=None
    ):
        structure = structure if structure is not None else self.structure
        hessian = hessian if hessian is not None else self.hessian
        windows = windows if windows is not None else self.windows
        noises = noises if noises is not None else self.noises
        pes = pes if pes is not None else self.pes
        copy_pls = ParallelLineSearch(
            path=path,
            structure=structure,
            hessian=hessian,
            windows=windows,
            noises=noises,
            no_eval=True,
            pes=pes
        )
        for ls, ls_new in zip(self.ls_list, copy_pls.ls_list):
            ls_new._settings = ls._settings
        # end for
        return copy_pls
    # end def

    def propagate(
        self,
        path=None,
        write=True,
        overwrite=True,
        add_sigma=False,
        fname='pls.p'
    ):
        if not self.evaluated:
            self.evaluate(add_sigma=add_sigma)
        # end if
        path = path if path is not None else self.path + '_next/'
        # Write to disk
        if write:
            self.write_to_disk(fname=fname, overwrite=overwrite)
        # end if
        # check if manually providing structure
        pls_next = self.copy(
            path,
            structure=self.structure_next
        )
        return pls_next
    # end def

    def plot(
        self,
        **kwargs  # TODO: list kwargs
    ):
        for ls in self.ls_list:
            ls.plot(**kwargs)
        # end for
    # end def

    def __str__(self):
        string = self.__class__.__name__
        if self.ls_list is None:
            string += '\n  Line-searches: None'
        else:
            string += '\n  Line-searches:\n'
            string += indent('\n'.join(['#{:<2d} {}'.format(ls.d, str(ls)) for ls in self.ls_list]), '    ')
        # end if
        # TODO
        return string
    # end def

    def __len__(self):
        return len(self.ls_list)
    # end def

# end class
