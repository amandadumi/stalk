#!/usr/bin/env python3

import sys
import numpy as np

from stalk.io import XyzGeometry
from stalk.params.ParameterStructure import ParameterStructure
from unit_tests.assets.test_jobs import efilename
from unit_tests.assets.h2o import pes_H2O, pos_H2O, elem_H2O


def evaluate_pes(pos, pes_variable):
    res1, res2 = 0.0, 0.0
    if pes_variable == 'h2o':
        res1, res2 = pes_H2O(pos)
    if pes_variable == 'relax_h2o':
        res1 = pos_H2O
        res2 = elem_H2O
    else:  # default: dummy
        pass
    # end if
    return res1, res2
# end def


if __name__ == '__main__':
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as fhandle:
            # The first line points to structure file
            struct_name = fhandle.readline().replace("\n", "")
            pos = XyzGeometry({'suffix': struct_name}).load('.').get_pos()
            # The second line points to one the hardcoded pes functions
            pes_variable = fhandle.readline().replace("\n", "")
            # Evaluate
            res1, res2 = evaluate_pes(pos, pes_variable)
        # end with
        if np.isscalar(res1):
            # Write energy and errorbar to disk
            np.savetxt(efilename, np.array([[res1, res2]]))
        else:
            # Write pos+elem to the disk
            structure = ParameterStructure(pos=res1, elem=res2)
            writer = XyzGeometry({'suffix': 'relax.xyz'})
            writer.write(structure, '.')
        # end if
    # end if
# end if
