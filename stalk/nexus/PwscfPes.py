#!/usr/bin/env python3

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

from numpy import nan

from nexus import PwscfAnalyzer

from stalk.params.PesResult import PesResult
from stalk.util.util import PL
from stalk.io.PesLoader import PesLoader


class PwscfPes(PesLoader):

    def __init__(self, args={}):
        self._func = None
        self.args = args
    # end def

    def _load(self, path, suffix='scf.in', **kwargs):
        ai = PwscfAnalyzer(PL.format(path, suffix), **kwargs)
        ai.analyze()
        if not hasattr(ai, "E") or ai.E == 0.0:
            # Analysis has failed
            E = nan
        else:
            E = ai.E
        # end if
        Err = 0.0
        return PesResult(E, Err)
    # end def

# end class
