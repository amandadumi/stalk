#!/usr/bin/env python3
'''LineSearchIteration class for treating iteration of subsequent parallel linesearches
'''

from stalk.params.PesFunction import PesFunction
from stalk.pls.TargetParallelLineSearch import TargetParallelLineSearch
from stalk.util import directorize
from stalk.pls import ParallelLineSearch
from numpy import array
from stalk.lsi.util import plot_parameter_convergence, plot_energy_convergence, plot_bundled_convergence
from stalk.util.util import FF, FFS, FI, FIS, FU

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# Class for line-search iteration
class LineSearchIteration():

    _pls_list: list[ParallelLineSearch]  # list of ParallelLineSearch objects
    _path = ''  # base path

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
        # Try to serialized iterations:
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
        if len(self.pls_list) > 0:
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
                data = [pls.structure.value, pls.structure.error]
                data[0] = data[0] if not data[0] is None else 0.0
                data[1] = data[1] if not data[1] is None else 0.0
                for param, perr in zip(pls.structure.params, pls.structure.params_err):
                    data.append(param)
                    data.append(perr)
                # end for
                string += fmt.format(p, *tuple(array(data)))
            # end for
        # end if
        return string
    # end def

# end class
