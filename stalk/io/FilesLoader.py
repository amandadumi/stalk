from ..params.PesResult import PesResult


class FilesLoader():
    '''A wrapper class for file-based PES loader.'''
    func = None
    args = None

    def __init__(self, func, args={}):
        self.func = func
        self.args = args
    # end def

    def load(self, path, **kwargs):
        '''The files PES loader must accept a "path" to input file and return a value/error pair.'''
        value, error = self.func(path=path, **self.args, **kwargs)
        return PesResult(value, error)
    # end def

# end class
