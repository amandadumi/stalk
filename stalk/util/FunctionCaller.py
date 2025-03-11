#!/usr/bin/env python3

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


class FunctionCaller():
    _func = None
    _args = {}

    @property
    def func(self):
        return self._func
    # end def

    @func.setter
    def func(self, func):
        if callable(func):
            self._func = func
        else:
            raise TypeError("The function must be callable!")
        # end if
    # end

    @property
    def args(self):
        return self._args
    # end def

    @args.setter
    def args(self, args):
        if isinstance(args, dict):
            self._args = args
        elif args is None:
            self._args = {}
        else:
            raise TypeError("The argument list must be a dictionary")
        # end if
    # end

    def __init__(self, func, args={}):
        if isinstance(func, FunctionCaller):
            self.func = func.func
            self.args = func.args
        else:
            self.func = func
            self.args = args
        # end if
    # end def

# end class
