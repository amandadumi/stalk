from .PesResult import PesResult


class PesFunction():
    func = None
    args = None

    def __init__(self, func, args={}):
        '''A PES function is constructed from the job-generating function and arguments.'''
        self.func = func
        self.args = args
    # end def

    def evaluate(self, structure, **kwargs):
        value, error = self.func(structure, **self.args, **kwargs)
        return PesResult(value, error)
    # end def

# end class
