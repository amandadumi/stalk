from numpy import array, isscalar
from scipy.optimize import minimize
from copy import deepcopy

from .Parameter import Parameter
from .PesFunction import PesFunction


class ParameterSet():
    """Base class for representing a set of parameters to optimize"""
    _param_list = []  # list of Parameter objects
    value = None  # energy value
    error = 0.0  # errorbar
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
                params_err=params_err,
                units=units,
                labels=labels
            )
        # end if
        if value is not None:
            self.set_value(value, error)
        # end if
    # end def

    def init_params(self, params, params_err=None, units=None, labels=None):
        if params_err is None:
            params_err = len(params) * [params_err]
        else:
            assert len(params_err) == len(params)
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
        for p, (param, param_err, unit, label) in enumerate(zip(params, params_err, units, labels)):
            if isinstance(param, Parameter):
                parameter = param
            elif isscalar(param):
                lab = label if label is not None else 'p{}'.format(p)
                parameter = Parameter(param, param_err, unit=unit, label=lab)    
            else:
                raise ValueError('Parameter is unsupported type: ' + str(param))
            # end if
            p_list.append(parameter)
        # end for
        self._param_list = p_list
    # end def

    def set_params(self, params, params_err=None):
        # If params have not been initiated yet, do it now without extra info
        if self.num_params > 0:
            assert len(params) == self.num_params
        else:
            self.init_params(params, params_err)
        # end if
        if params_err is None:
            params_err = len(params) * [0.0]
        # end if
        for sparam, param, param_err in zip(self.params_list, params, params_err):
            assert isinstance(sparam, Parameter)
            sparam.value = param
            sparam.error = param_err
        # end for
        self.unset_value()
    # end def

    def set_value(self, value, error=0.0):
        assert self.params is not None, 'Cannot assign value to abstract structure, set params first'
        self.value = value
        self.error = error
    # end def

    def unset_value(self):
        self.value = None
        self.error = None
    # end def

    @property
    def params_list(self):
        return [p for p in self._param_list]
    # end def

    @property
    def params(self):
        if self.num_params > 0:
            return array([p.value for p in self.params_list])
        # end if
    # end def

    @property
    def params_err(self):
        if self.num_params > 0:
            return array([p.param_err for p in self.params_list])
        # end if
    # end def

    @property
    def num_params(self):
        return len(self.params_list)
    # end def

    def shift_params(self, shifts):
        if len(shifts) != self.num_params:
            raise ValueError('Shifts has wrong dimensions!')
        # end if
        for param, shift in zip(self.params_list, shifts):
            assert isinstance(param, Parameter)
            param.shift(shift)
        # end for
        self.unset_value()
    # end def

    def copy(
        self,
        params=None,
        params_err=None,
        label=None
    ):
        paramset = deepcopy(self)
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
