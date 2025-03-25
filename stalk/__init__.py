"""Surrogate Hessian accelerated parallel line-search."""

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"
__version__ = "0.2.0a1"

from . import util
from . import params
from . import ls
from . import pls
from . import lsi
from . import io
from .params import ParameterSet, ParameterStructure, ParameterHessian
from .lsi import LineSearchIteration
from .ls import LineSearch, TargetLineSearch
from .pls import ParallelLineSearch, TargetParallelLineSearch
try:
    from . import nexus
    from .nexus import NexusStructure
    from .nexus import NexusPes
    nexus_enabled = True
except ModuleNotFoundError:
    # Nexus not found
    nexus_enabled = False
    pass
# end try


__all__ = [
    'util',
    'params',
    'ls',
    'pls',
    'lsi',
    'io',
    'ParameterSet',
    'ParameterHessian',
    'ParameterStructure',
    'LineSearchIteration',
    'LineSearch',
    'TargetLineSearch',
    'ParallelLineSearch',
    'TargetParallelLineSearch',
    'nexus',
    'NexusPes',
    'NexusStructure',
]
