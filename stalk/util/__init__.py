#!/usr/bin/env python3
"""Surrogate Hessian accelerated parallel line-search: utilities"""

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

from stalk.util.EffectiveVariance import EffectiveVariance
from stalk.util.util import bipolyfit
from stalk.util.util import bipolyval
from stalk.util.util import bipolynomials
from stalk.util.util import directorize
from stalk.util.util import get_fraction_error
from stalk.util.util import get_min_params
from stalk.util.util import match_to_tol
from stalk.util.util import Bohr
from stalk.util.util import Hartree
from stalk.util.util import Ry

__all__ = [
    'EffectiveVariance',
    'bipolyfit',
    'bipolynomials',
    'bipolyval',
    'directorize',
    'get_fraction_error',
    'get_min_params',
    'match_to_tol',
    'Bohr',
    'Hartree',
    'Ry',
]
