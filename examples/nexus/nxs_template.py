#!/usr/bin/env python3

# First, the user must set up Nexus according to their computing environment.
from nexus import obj, settings

# Configure Nexus on a local machine
cores = 8
nx_settings = obj(
    sleep=3,
    pseudo_dir='../pseudos',
    runs='',
    results='',
    status_only=0,
    generate_only=0,
    machine=f'ws{cores}',
)

# Make sure to init only once
if len(settings) == 0:
    settings(**nx_settings)
# end if

# Configure
presub = ''
qeapp = 'pw.x'
p2qapp = 'pw2qmcpack.x'
qmcapp = 'qmcpack'
pwscfjob = obj(app=qeapp, cores=cores, ppn=cores, presub=presub)
pyscfjob = obj(app='python3', serial=True)
p2qjob = obj(app=p2qapp, cores=1, ppn=1, presub=presub)
optjob = obj(app=qmcapp, cores=cores, ppn=cores, presub=presub)
dmcjob = obj(app=qmcapp, cores=cores, ppn=cores, presub=presub)
