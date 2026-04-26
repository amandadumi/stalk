#!/usr/bin/env python

from stalk import TransitionPathway

from params import pes
from run1_neb import traj_neb

basedir = 'tpw'

tpw = TransitionPathway(
    path=basedir,
    images=traj_neb
)
tpw.calculate_hessians(pes=pes)
