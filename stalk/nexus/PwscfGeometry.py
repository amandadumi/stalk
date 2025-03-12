from nexus import PwscfAnalyzer
from stalk.util.util import PL

from ..io.GeometryLoader import GeometryLoader
from stalk.params.GeometryResult import GeometryResult


class PwscfGeometry(GeometryLoader):

    def __init__(self, args={}):
        self._func = None
        self.args = args
    # end def

    def _load(self, path, suffix='relax.in', c_pos=1.0, **kwargs):
        ai = PwscfAnalyzer(PL.format(path, suffix), **kwargs)
        ai.analyze()
        pos = ai.structures[len(ai.structures) - 1].positions * c_pos
        try:
            axes = ai.structures[len(ai.structures) - 1].axes
        except AttributeError:
            # In case axes is not present in the relaxation
            axes = None
        # end try
        return GeometryResult(pos, axes)
    # end def

# end class
