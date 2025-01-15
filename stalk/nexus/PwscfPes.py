
from nexus import PwscfAnalyzer

from stalk.params.PesResult import PesResult
from ..io.PesLoader import PesLoader


class PwscfPes(PesLoader):

    def __load__(self, path, suffix='scf.in', **kwargs):
        ai = PwscfAnalyzer('{}/{}'.format(path, suffix), **kwargs)
        ai.analyze()
        E = ai.E
        Err = 0.0
        return PesResult(E, Err)
    # end def

# end class
