from stalk.params.PesResult import PesResult


class PesLoader():
    _args = {}

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
            raise TypeError("The PES arguments must be a dictionary")
        # end if
    # end

    def __init__(self, loader=None, func=None, args={}):
        if isinstance(loader, PesLoader):
            self.__load__ = loader.__load__
            self.args = loader.args
        elif callable(loader):
            # Allow construction with: PesLoader(loader_func, loader_args)
            self.__load__ = loader
            self.args = func
        else:
            if callable(func):
                # Try to override __load__ method in the hopes that it conforms
                self.__load__ = func
            # end if
            self.args = args
        # end if
    # end def

    def load(self, path, sigma=0.0, **kwargs):
        '''The PES loader must accept a "path" to input file and return PesResult.
        '''
        args = self.args.copy()
        args.update(kwargs)
        res = self.__load__(path=path, **args)
        if not isinstance(res, PesResult):
            raise AssertionError('The __load__ method must return a PesResult instance.')
        # end if
        # If a non-zero, artificial errorbar is requested, add it to result
        res.add_sigma(sigma)
        return res
    # end def

    def __load__(self, path=None, *args, **kwargs):
        raise NotImplementedError("Implement __load__ function in inherited class.")
    # end def

# end class
