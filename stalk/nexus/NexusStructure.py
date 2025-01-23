
from stalk.util import EffectiveVariance
from structure import Structure
from nexus import run_project

from stalk.io.GeometryLoader import GeometryLoader
from stalk.params.ParameterStructure import ParameterStructure
from stalk.util.util import directorize

from .NexusGenerator import NexusGenerator


class NexusStructure(ParameterStructure):

    def relax(
        self,
        pes=None,
        pes_func=None,
        pes_args={},
        path='relax',
        loader=None,
        loader_args={},
        **kwargs,
    ):
        if not isinstance(pes, NexusGenerator):
            # Generate jobs
            pes = NexusGenerator(pes_func, pes_args)
        # end if
        # Make a copy structure for job generation
        relax_jobs = pes.generate(self.get_nexus_structure(), directorize(path))
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
        **kwargs
    ):
        kwargs.update({
            'elem': self.elem,
            'pos': self.pos,
            'units': self.units,
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

    def get_var_eff(
        self,
        pes=None,
        pes_func=None,
        pes_args={},
        loader=None,
        loader_args={},
        samples=10,
        path='path'
    ):
        if not isinstance(pes, NexusGenerator):
            pes = NexusGenerator(pes_func, pes_args)
        # end if
        jobs = pes.generate(self.get_nexus_structure(), path, sigma=None, samples=samples)
        run_project(jobs)

        Err = loader.load(path, **loader_args).get_error()
        var_eff = EffectiveVariance(samples, Err)
        return var_eff
    # end def

# end class
