from numpy import array, isscalar
from scipy.optimize import minimize
from copy import deepcopy

from stalk.params.LineSearchPoint import LineSearchPoint
from stalk.params.PesFunction import PesFunction
from stalk.params.Parameter import Parameter

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


class ParameterSet(LineSearchPoint):
    """Base class for representing a set of parameters to optimize"""
    _param_list = []  # list of Parameter objects
    label = None  # label for identification

    def __init__(
        self,
        params=None,  # List of scalars or Parameter objects
        params_err=None,
        units=None,
        value=None,
        error=0.0,
        label=None,
        labels=None
    ):
        self.label = label
        if params is not None:
            self.init_params(
                params,
                errors=params_err,
                units=units,
                labels=labels
            )
        # end if
        if value is not None:
            self.value = value
            self.error = error
        # end if
    # end def

    def init_params(self, params, errors=None, units=None, labels=None):
        if errors is None:
            errors = len(params) * [errors]
        else:
            assert len(errors) == len(params)
        # end if
        if units is None or isinstance(units, str):
            units = len(params) * [units]
        else:
            assert len(units) == len(params)
        # end if
        if labels is None:
            labels = len(params) * [labels]
        else:
            assert len(labels) == len(labels)
        # end if
        p_list = []
        for p, (param, error, unit, label) in enumerate(zip(params, errors, units, labels)):
            if isinstance(param, Parameter):
                parameter = param
            elif isscalar(param):
                lab = label if label is not None else 'p{}'.format(p)
                parameter = Parameter(param, error, unit=unit, label=lab)
            else:
                raise ValueError('Parameter is unsupported type: ' + str(param))
            # end if
            p_list.append(parameter)
        # end for
        self._param_list = p_list
    # end def

    def set_params(self, params, errors=None):
        # If params have not been initiated yet, do it now without extra info
        if len(self) > 0:
            assert len(params) == len(self)
        else:
            self.init_params(params, errors)
        # end if
        if errors is None:
            errors = len(params) * [0.0]
        # end if
        for sparam, param, error in zip(self.params_list, params, errors):
            assert isinstance(sparam, Parameter)
            sparam.value = param
            sparam.error = error
        # end for
        self.reset_value()
    # end def

    @property
    def params_list(self):
        return [p for p in self._param_list if isinstance(p, Parameter)]
    # end def

    @property
    def params(self):
        if len(self) > 0:
            return array([p.value for p in self.params_list])
        # end if
    # end def

    @property
    def params_err(self):
        if len(self) > 0:
            return array([p.error for p in self.params_list])
        # end if
    # end def

    def __len__(self):
        return len(self.params_list)
    # end def

    def shift_params(self, shifts):
        if len(shifts) != len(self):
            raise ValueError('Shifts has wrong dimensions!')
        # end if
        for param, shift in zip(self.params_list, shifts):
            assert isinstance(param, Parameter)
            param.shift(shift)
        # end for
        self.reset_value()
    # end def

    def copy(
        self,
        params=None,
        params_err=None,
        label=None,
        offset=None
    ):
        paramset = deepcopy(self)
        if offset is not None:
            paramset.offset = offset
        # end if
        if params is not None:
            paramset.set_params(params, params_err)
        # end if
        if label is not None:
            paramset.label = label
        # end if
        return paramset
    # end def

    def check_consistency(self):
        return True
    # end def

    def relax(
        self,
        pes,
        **kwargs
    ):
        assert isinstance(pes, PesFunction), "Must provide PES as a PesFunction instance."

        # Relax numerically using a wrapper around SciPy minimize
        def relax_aux(p):
            return pes.evaluate(ParameterSet(p)).get_value()
        # end def
        res = minimize(relax_aux, self.params, **kwargs)
        self.set_params(res.x)
    # end def

# end class
