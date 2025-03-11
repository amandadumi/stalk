#!/usr/bin/env python3

from numpy import loadtxt

from stalk.io.PesLoader import PesLoader
from stalk.params.PesResult import PesResult


class FilesLoader(PesLoader):

    def __init__(self, func, args={}):
        self._func = None
        self.args = args
    # end def

    def _load(self, path, suffix='energy.dat', **kwargs):
        value, error = loadtxt('{}/{}'.format(path, suffix))
        return PesResult(value, error)
    # end def

# end class
