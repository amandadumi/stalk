#!/usr/bin/env python

from params import pes
from run2_tpw_hessian import tpw

tpw.generate_surrogates(
    pes=pes,
    fit_kind='pf3',
    M=15,
    window_frac=0.25
)
tpw.optimize_surrogates(
    M=7,
    fit_kind='pf3',
    temperature=0.0002,
)
