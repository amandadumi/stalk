
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


class PwscfEnthalpy(PesLoader):

    def __load__(self, path, suffix='scf.in', **kwargs):
        from nexus import PwscfAnalyzer
        ai = PwscfAnalyzer('{}/{}'.format(path, suffix), **kwargs)
        ai.analyze()
        E = ai.E
        # pressure and volume are marked as obsolete in nexus pwscf analyzer so need to test. 
        P = ai.pressure
        V = ai.volume
        H = E + (P*V)
        Err = 0.0
        return PesResult(E, Err)
    # end def