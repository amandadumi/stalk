from .LineSearch import LineSearch


class LineSearchDummy(LineSearch):

    def __init__(
        self,
        d=0,
        **kwargs,
    ):
        self.d = d
    # end def

    def set_results(self, **kwargs):
        return True
    # end def

    def evaluate_pes(self, **kwargs):
        pass
    # end def

# end class
