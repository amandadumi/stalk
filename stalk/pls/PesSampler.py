#!/usr/bin/env python3
'''Generic base class for sampling a PES in iterative batches
'''

from dill import dumps, loads
from os import makedirs, path

from stalk.params import PesFunction
from stalk.params.ParameterSet import ParameterSet
from stalk.util.util import directorize

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# A class for managing sampling of the PES
class PesSampler():
    _path = None
    pes = None

    @property
    def path(self):
        return self._path
    # end def

    @path.setter
    def path(self, path):
        if isinstance(path, str):
            self._path = path
        else:
            self._path = None
        # end if
    # end def

    # Try to load the instance from file before ordinary init
    def __new__(cls, load=None, *args, **kwargs):
        if load is None:
            return super().__new__(cls)
        else:
            # Try to load a pickle file from disk.
            try:
                with open(load, mode='rb') as f:
                    data = loads(f.read(), ignore=False)
                # end with
                if isinstance(data, cls):
                    return data
                else:
                    raise TypeError("The loaded file is not the same kind!")
                # end if
            except FileNotFoundError:
                return super().__new__(cls)
            # end try
        # end if
    # end def

    def __init__(
        self,
        path=None,
        pes=None,
        pes_func=None,
        pes_args={},
        load=None,  # eliminate loading arg
    ):
        if load is not None and self.pes is not None:
            # Proxies of successful loading from disk
            return
        # end if
        self.pes = PesFunction(pes, pes_func, pes_args)
        self.path = path
    # end def

    def write_to_disk(self, fname='data.p', overwrite=False):
        fpath = directorize(self.path) + fname
        if path.exists(fpath) and not overwrite:
            print(f'File {fpath} exists. To overwrite, run with overwrite = True')
            return
        # end if
        makedirs(self.path, exist_ok=True)
        with open(fpath, mode='wb') as f:
            f.write(dumps(self, byref=True))
        # end with
    # end def

    def _evaluate_energies(
        self,
        structures: list[ParameterSet],
        sigmas: list[float],
        add_sigma=False,
    ):
        for structure, sigma in zip(structures, sigmas):
            if isinstance(structure, ParameterSet):
                # Set value, error
                res = self.pes.evaluate(structure, sigma=sigma)
                if add_sigma:
                    res.add_sigma(sigma)
                # end if
                structure.value = res.get_value()
                structure.error = res.get_error()
            # end if
        # end for
    # end def

# end class
