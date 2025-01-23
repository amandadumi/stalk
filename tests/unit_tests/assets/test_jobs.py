#!/usr/bin/env python3

import os
import numpy as np
from pathlib import Path
from nexus import obj, job, settings, generate_physical_system, input_template
from simulation import GenericSimulation, SimulationAnalyzer

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


class TestAnalyzer(SimulationAnalyzer):
    value = None
    error = 0.0

    def __init__(
        self,
        sim
    ):
        self.path = sim.path
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
    # end def
# end class


def init_nexus():
    if len(settings) == 0:
        settings(**nx_settings)
    # end if
# end def


def nxs_generic_pes(structure, path, sigma=None, system_args={}, pes_variable='dummy', **kwargs):
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
        identifier='nxs_test',
        analyzer_type=TestAnalyzer  # This is not needed here
    )
    return [sim]
# end def
