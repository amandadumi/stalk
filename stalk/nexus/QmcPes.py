
from nexus import QmcpackAnalyzer

from stalk.params.PesResult import PesResult
from ..io.PesLoader import PesLoader


class QmcPes(PesLoader):

    def __init__(self, args={}):
        self._func = None
        self.args = args
    # end def

    def _load(self, path, qmc_idx=1, suffix='dmc/dmc.in.xml', **kwargs):
        ai = QmcpackAnalyzer('{}/{}'.format(path, suffix), **kwargs)
        ai.analyze()
        LE = ai.qmc[qmc_idx].scalars.LocalEnergy
        E = LE.mean
        Err = LE.error
        return PesResult(E, Err)
    # end def

# end class
