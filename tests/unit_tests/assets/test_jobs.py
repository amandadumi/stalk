#!/usr/bin/env python3

import os
import numpy as np
from pathlib import Path
from nexus import obj, job, settings, generate_physical_system, input_template
from simulation import GenericSimulation, SimulationAnalyzer, Simulation
from stalk.io.PesLoader import PesLoader
from stalk.params.PesResult import PesResult

# Add nexus tester to path
app_path = os.path.dirname(__file__) + "/../../nxs_tester.py"

# The hard-coded energy file name
efilename = 'test_energy.dat'

# First, the user must set up Nexus according to their computing environment.
testjob = obj(app_command=app_path + ' nxs_test.in', cores=1, ppn=1, serial=True)
nx_settings = obj(
    sleep=1,
    pseudo_dir='./',  # not used in testing
    runs='',  # not used in testing
    results='',  # not used in testing
    status_only=0,
    generate_only=0,
    command_line=False,  # disable for testing
    machine='ws4',
)


# Tailor Nexus analyzer for generic testing
class TestAnalyzer(SimulationAnalyzer):
    value = None
    error = 0.0

    def __init__(
        self,
        arg0
    ):
        if isinstance(arg0, Simulation):
            self.path = arg0.path
        else:
            self.path = arg0
        # end if
    # end def

    def analyze(self):
        e_file = Path(self.path + "/" + efilename)
        if e_file.exists():
            res = np.loadtxt(e_file)
            if len(res) > 1:
                self.value = res[0]
                self.error = res[1]
            else:
                self.value = res[0]
            # end if
        else:
            raise AssertionError('The job has not run yet!')
        # end if
    # end def
# end class


# Tailor Nexus analyzer for generic testing
class TestLoader(PesLoader):

    def __load__(self, path, **kwargs):
        ai = TestAnalyzer(path, **kwargs)
        ai.analyze()
        return PesResult(ai.value, ai.error)
    # end def

# end class


def init_nexus():
    if len(settings) == 0:
        settings(**nx_settings)
    # end if
# end def


def nxs_generic_pes(
    structure,
    path,
    system_args={},
    pes_variable='dummy',
    sigma=None,
    **kwargs
):
    init_nexus()
    system = generate_physical_system(structure=structure, **system_args)
    # TODO: this could be done more elegantly using Nexus templates
    input = "nxs_test.struct.xyz\n" + pes_variable + '\n'
    job_input = input_template(input)
    sim = GenericSimulation(
        system=system,
        job=job(**testjob),
        path=path,
        input=job_input,
        identifier='nxs_test'
    )
    return [sim]
# end def
