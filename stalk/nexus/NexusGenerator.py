from structure import Structure


class NexusGenerator():
    '''A wrapper class for generating Nexus functions to produce and represent a PES.'''
    func = None
    args = {}

    def __init__(self, func, args={}):
        '''A Nexus PES function is constructed from the job-generating function and arguments.'''
        if not callable(func):
            raise TypeError("The PES function must be callable.")
        # end if
        self.func = func
        self.args = args
    # end def

    def generate(self, structure, path, **kwargs):
        '''Return a list of Nexus jobs provided "structure" and "path" arguments.'''
        if not isinstance(structure, Structure):
            raise TypeError('Must use Structure class with NexusGenerator')
        # end if
        args = self.args.copy()
        args.update(kwargs)
        jobs = self.func(structure, path, **args)
        return jobs
    # end def

# end class
