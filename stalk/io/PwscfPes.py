from stalk.params.PesResult import PesResult
from .PesLoader import PesLoader


class PwscfPes(PesLoader):

    def __load__(self, path, suffix='scf.in'):
        from nexus import PwscfAnalyzer
        ai = PwscfAnalyzer('{}/{}'.format(path, suffix))
        ai.analyze()
        E = ai.E
        Err = 0.0
        return PesResult(E, Err)
    # end def

# end class
