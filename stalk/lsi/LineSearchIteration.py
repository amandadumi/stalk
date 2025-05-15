#!/usr/bin/env python3
'''LineSearchIteration class for treating iteration of subsequent parallel linesearches'''

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

from numpy import array, mean

from stalk.params.ParameterSet import ParameterSet
from stalk.params.PesFunction import PesFunction
from stalk.pls.TargetParallelLineSearch import TargetParallelLineSearch
from stalk.util import directorize
from stalk.pls import ParallelLineSearch
from stalk.lsi.util import plot_energy_convergence
from stalk.lsi.util import plot_bundled_convergence
from stalk.lsi.util import plot_parameter_convergence
from stalk.util.util import FF, FFS, FI, FIS, FU


class LineSearchIteration():
    _pls_list: list[ParallelLineSearch]  # list of ParallelLineSearch objects
    _path = ''  # base path
    _transient = 0

    def __init__(
        self,
        path='',
        surrogate=None,
        structure=None,
        hessian=None,
        pes=None,
        pes_func=None,
        pes_args={},
        **pls_args
    ):
        self.path = path
        self._pls_list = []
        # Try to load serialized iterations:
        self.load_pls()
        # if no iterations loaded, try to initialize
        if len(self) == 0 or not self.pls(0).evaluated:
            if not isinstance(pes, PesFunction):
                # If none are provided, raises TypeError
                pes = PesFunction(pes_func, pes_args)
            # end if
            # Try to load from surrogate ParallelLineSearch object
            if surrogate is not None:
                self.init_from_surrogate(
                    surrogate=surrogate,
                    structure=structure,
                    pes=pes,
                )
            # end if
            # When present, manually provided mappings, parameters and positions
            # override those from a surrogate
            if hessian is not None:
                self.init_from_hessian(
                    hessian,
                    structure,
                    pes=pes,
                    **pls_args
                )
            # end if
        # end if
    # end def

    @property
    def pls_list(self):
        return self._pls_list
    # end def

    @property
    def path(self):
        return self._path
    # end def

    @path.setter
    def path(self, path):
        if isinstance(path, str):
            self._path = directorize(path)
        else:
            raise TypeError("path must be a string")
        # end if
    # end def

    @property
    def structure_init(self):
        return self.pls(0).structure
    # end def

    @property
    def structure_final(self):
        params_list = []
        params_err_list = []
        values_list = []
        errors_list = []
        for pls in self.pls_list[self.transient:]:
            if pls.evaluated:
                params_list.append(pls.structure_next.params)
                params_err_list.append(pls.structure_next.params_err)
            # end if
        # end for
        for pls in self.pls_list[self.transient + 1:]:
            if pls.structure.valid:
                values_list.append(pls.structure.value)
                errors_list.append(pls.structure.error)
            # end if
        # end for
        mean_params = mean(array(params_list), axis=0)
        mean_params_err = (mean(array(params_err_list)**2, axis=0) / len(params_err_list))**0.5

        if len(values_list) > 0:
            mean_value = mean(values_list)
            mean_error = mean(array(errors_list)**2 / len(errors_list))**0.5
        else:
            mean_value = 0.0
            mean_error = 0.0
        # end if

        mean_structure = self.structure_init.copy(
            params=mean_params,
            params_err=mean_params_err,
        )
        mean_structure.value = mean_value
        mean_structure.error = mean_error
        return mean_structure
    # end def

    @property
    def transient(self):
        return self._transient
    # end def

    @transient.setter
    def transient(self, transient):
        if isinstance(transient, int) and transient >= 0:
            self._transient = min(transient, len(self) - 1)
        else:
            raise ValueError("transient must be integer >= 0")
        # end if
    # end def

    def init_from_surrogate(
        self,
        surrogate: ParallelLineSearch,
        structure=None,
        pes=None,
    ):
        if isinstance(surrogate, TargetParallelLineSearch):
            pls = surrogate.copy(
                path=self._get_pls_path(0),
                structure=structure,
                pes=pes
            )
        elif isinstance(surrogate, ParallelLineSearch):
            pls = surrogate.copy(
                path=self._get_pls_path(0),
                structure=structure,
                pes=pes
            )
        else:
            raise AssertionError('Surrogate parameter must be a ParallelLineSearch object')
        # end if
        self._pls_list = [pls]
    # end def

    def init_from_hessian(
        self,
        hessian,
        structure=None,
        pes=None,
        **pls_args
    ):
        if len(self) == 0:
            pls = ParallelLineSearch(
                path=self._get_pls_path(0),
                hessian=hessian,
                structure=structure,
                pes=pes,
                **pls_args
            )
            self.pls_list.append(pls)
        else:
            pls = self.pls(0)
            pls.hessian = hessian
            pls.structure = structure
        # end if
    # end def

    def _get_pls_path(self, i):
        return '{}pls{}/'.format(self.path, i)
    # end def

    def evaluate(
        self,
        add_sigma=False
    ):
        return self._get_current_pls().evaluate(add_sigma=add_sigma)
    # end def

    def _get_current_pls(self):
        # The list cannot be empty
        return self.pls_list[-1]
    # end def

    def pls(self, i=None):
        if i is None:
            return self._get_current_pls()
        elif i < len(self.pls_list):
            return self.pls_list[i]
        else:
            return None
        # end if
    # end def

    def load_pls(self):
        i = 0
        while i < 100:
            path = '{}pls.p'.format(self._get_pls_path(i))
            try:
                pls = ParallelLineSearch(load=path)
                if not pls.setup:
                    # Means loading failed
                    break
                # end if
                self._pls_list.append(pls)
                i += 1
            except TypeError:
                # This means load has failed
                break
            # end try
        # end while
    # end def

    def propagate(
        self,
        i=None,
        write=True,
        overwrite=True,
        fname='pls.p',
        add_sigma=False,
        interactive=False,
    ):
        # Do not propagate if 'i' points to earlier iteration
        if i is not None and i < len(self.pls_list) - 1:
            return
        # end if
        i = len(self)
        pls_next = self.pls().propagate(
            path=self._get_pls_path(i),
            write=write,
            overwrite=overwrite,
            fname=fname,
            add_sigma=add_sigma,
            interactive=interactive,
        )
        self.pls_list.append(pls_next)
    # end

    def plot_convergence(
        self,
        bundle=True,
        **kwargs
    ):
        if bundle:
            plot_bundled_convergence(self.pls_list, **kwargs)
        else:
            plot_energy_convergence(self.pls_list, **kwargs)
            plot_parameter_convergence(
                self.pls_list, **kwargs)
    # end def

    def __len__(self):
        return len(self.pls_list)
    # end def

    def __str__(self):
        string = self.__class__.__name__
        if len(self) > 0:
            fmt = '\n  ' + FI + FF + FU + self.pls().D * (FF + FU)
            fmts = '\n  ' + FIS + FFS + FFS + self.pls().D * (FFS + FFS)

            # Labels row
            plabels = ['pls', 'Energy', '']
            for param in self.pls().structure.params_list:
                plabels += [param.label, '']
            # end for
            string += fmts.format(*tuple(plabels))

            # Data rows
            for p, pls in enumerate(self.pls_list):
                string += self._params_row(p, pls.structure, fmt)
            # end for
        # end if
        # Mean params
        if len(self) > 1:
            string += f"\nMean[{self.transient + 1}:]:"
            string += self._params_row(len(self), self.structure_final, fmt)
        # end if
        return string
    # end def

    def _params_row(self, p, structure: ParameterSet, fmt: str):
        data = [structure.value, structure.error]
        data[0] = data[0] if not data[0] is None else 0.0
        data[1] = data[1] if not data[1] is None else 0.0
        for param, perr in zip(structure.params, structure.params_err):
            data.append(param)
            data.append(perr)
        # end for
        return fmt.format(p, *tuple(array(data)))
    # end def

# end class
