#!/usr/bin/env $python_exe

from numpy import savetxt
$pyscfimport

$system

$calculation

savetxt('energy.dat', [[e_scf, 0.0]])