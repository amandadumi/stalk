#!/usr/bin/env python3

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

from scipy.optimize import minimize

from stalk.params.ParameterSet import ParameterSet
from stalk.params.PesResult import PesResult
from stalk.util.FunctionCaller import FunctionCaller


class PesFunction(FunctionCaller):

    def evaluate(
        self,
        structure: ParameterSet,
        sigma=0.0,
        add_sigma=False,
        interactive=False,  # catch interactive
        **kwargs  # path, dep_jobs
    ):
        result = self._evaluate_structure(structure, sigma=sigma, **kwargs)
        # Load hook to be used in derived classes
        self._load_structure(
            structure,
            result=result,
            sigma=sigma,
            add_sigma=add_sigma
        )
    # end def

    def evaluate_all(
        self,
        structures: list[ParameterSet],
        sigmas=None,
        add_sigma=False,
        interactive=False,  # catch interactive
        **kwargs  # path, dep_jobs
    ):
        if sigmas is None:
            sigmas = len(structures) * [0.0]
        # end if
        for structure, sigma in zip(structures, sigmas):
            self.evaluate(structure, sigma=sigma, add_sigma=add_sigma, **kwargs)
        # end for
    # end def

    def _evaluate_structure(
        self,
        structure: ParameterSet,
        sigma=0.0,
        **kwargs
    ):
        eval_args = self.args.copy()
        # Override with kwargs
        eval_args.update(**kwargs)
        value, error = self.func(structure, sigma=sigma, **eval_args)
        return PesResult(value, error)
    # end def

    def _load_structure(
        self,
        structure: ParameterSet,
        result: PesResult = None,
        add_sigma=False,
        sigma=0.0,
    ):
        if result is None:
            return
        # end if
        # Loading is trivial because the result is readily provided
        if add_sigma:
            result.add_sigma(sigma)
        # end if
        structure.value = result.value
        structure.error = result.error
    # end def

    def relax(
        self,
        structure: ParameterSet,
        **kwargs
    ):
        # Relax numerically using a wrapper around SciPy minimize
        def relax_aux(p):
            s = structure.copy(params=p)
            self.evaluate(s)
            return s.value
        # end def
        p0 = structure.params
        res = minimize(relax_aux, p0, **kwargs)
        structure.set_params(res.x)
    # end def

# end class
