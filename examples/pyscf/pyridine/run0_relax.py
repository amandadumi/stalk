#!/usr/bin/env python3

from numpy import array, pi, sin, cos

from stalk import ParameterStructure
from stalk.io import XyzGeometry

from params import forward, backward, relax_pyscf


# Let us initiate a ParameterStructure object that implements the parametric mappings
r_CH = 1.08
pi_3 = pi / 3
params_init = array([
    1.335 * sin(pi_3),  # x_C04
    1.335 * cos(pi_3),  # y_C04
    1.335 * sin(pi_3),  # x_C13
    1.395 + 1.335 * cos(pi_3),  # y_C13
    1.395 + 1.335 * cos(pi_3) + 1.390 * cos(pi_3),  # y_C2
    (1.335 + r_CH) * sin(pi_3),  # x_H04
    (1.335 + r_CH) * cos(pi_3),  # y_H04
    (1.335 + r_CH) * sin(pi_3),  # x_H13
    1.395 + (1.335 + r_CH) * cos(pi_3),  # y_H13
    r_CH + 1.395 + 1.335 * cos(pi_3) + 1.390 * cos(pi_3),  # y_H2
])
elem = ['N'] + 5 * ['C'] + 5 * ['H']
structure_init = ParameterStructure(
    forward=forward,
    backward=backward,
    params=params_init,
    elem=elem,
    units='A'
)

outfile = 'relax.xyz'
try:
    geom = XyzGeometry({'suffix': outfile}).load('./')
except FileNotFoundError:
    new_params = relax_pyscf(structure_init, outfile)
    geom = XyzGeometry({'suffix': outfile}).load('./')
# end try
new_params = structure_init.map_forward(geom.get_pos())
structure_relax = structure_init.copy(params=new_params)

print(structure_init)
print(structure_relax)
