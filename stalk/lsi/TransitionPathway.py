#!/usr/bin/env python3
'''TransitionStateSearch class for finding transition pathways.'''

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

from stalk import LineSearchIteration


class TransitionPathway():
    _lsi_list: list[LineSearchIteration]  # list of LineSearchIteration objects
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
        pass
    # end def
    
# end class
