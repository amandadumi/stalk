#!/usr/bin/env python3

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

from numpy import linalg, pi, arccos, array, dot, sin, cos, linspace
from scipy.optimize import minimize

from stalk.params.ParameterSet import ParameterSet


def distance(r0, r1):
    '''Return Euclidean distance between two positions'''
    r = linalg.norm(r0 - r1)
    return r
# end def


def bond_angle(r0, rc, r1, units='ang'):
    '''Return dihedral angle between 3 bodies'''
    v0 = r0 - rc
    v1 = r1 - rc
    ang = angle(v0, v1, units=units)
    return ang
# end def


def angle(v0, v1, units='ang'):
    cosang = dot(v0, v1) / linalg.norm(v0) / linalg.norm(v1)
    ang = arccos(cosang) * 180 / pi if units == 'ang' else arccos(cosang)
    return ang
# end def


def mean_distances(pairs):
    '''Return average distance over (presumably) identical position pairs'''
    rs = []
    for pair in pairs:
        rs.append(distance(pair[0], pair[1]))
    # end for
    return array(rs).mean()
# end def


def mean_param(params, tol=1e-6):
    avg = array([params]).mean()
    if not all(params - avg < tol):
        print("Warning! Some of symmetric parameters stand out:")
        print(params)
    # end if
    return avg
# end def


def invert_pos(pos0, params, forward=None, tol=1.0e-7, method='BFGS'):
    assert forward is not None, 'Must provide forward mapping'

    def dparams_sum(pos1):
        return sum((params - forward(pos1))**2)
    # end def
    pos1 = minimize(dparams_sum, pos0, tol=tol, method=method).x
    return pos1
# end def


def rotate_2d(arr_2d, ang, units='ang'):
    if units == 'ang':
        ang *= pi / 180
    # end if
    R = array([
        [cos(ang), -sin(ang)],
        [sin(ang), cos(ang)]
    ])
    return R @ arr_2d
# end def


def interpolate_params(structure_a: ParameterSet, structure_b: ParameterSet, num_int):
    scales = linspace(0.0, 1.0, num_int + 2)
    dparams = structure_b.params - structure_a.params
    traj = []
    for scale in scales:
        new_params = structure_a.params + scale * dparams
        structure = structure_a.copy(params=new_params)
        traj.append(structure)
    # end for
    return traj
# end def
