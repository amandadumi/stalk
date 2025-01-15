
from structure import Structure
from nexus import run_project

from stalk.io.GeometryLoader import GeometryLoader
from stalk.params.ParameterStructure import ParameterStructure
from stalk.util.util import directorize

from .NexusGenerator import NexusGenerator


class NexusStructure(ParameterStructure):

    def relax(
        self,
        pes_func=None,
        pes_args={},
        path='relax',
        loader=None,
        loader_args={},
        **kwargs,
    ):
        # Generate jobs
        relax = NexusGenerator(pes_func, pes_args)
        # Make a copy structure for job generation
        relax_jobs = relax.generate(self.get_nexus_structure(), directorize(path))
        # Run project
        run_project(relax_jobs)

        # Load results
        if isinstance(loader, GeometryLoader):
            # returns pos, axes
            pos, axes = loader.load(path, **loader_args).get_result()
            self.set_position(pos, axes)
        # end if
    # end def

    def get_nexus_structure(
        self,
        kshift=(0, 0, 0),
        kgrid=(1, 1, 1),
        units='A',
        **kwargs
    ):
        kwargs.update({
            'elem': self.elem,
            'pos': self.pos,
            'units': units,
        })
        if self.axes is not None:
            kwargs.update({
                'axes': self.axes,
                'kshift': kshift,
                'kgrid': kgrid,
            })
        # end if
        return Structure(**kwargs)
    # end def

# end class
