Installation
============

Requirements
------------

First, install the following requirements

- NumPy (2.0.1)
- SciPy (1.14.1)
- Matplotlib (3.9.2)
- dill (0.3.9)

::

    pip install numpy scipy matploblit dill

Optional
~~~~~~~~

To manage atomic structures and simulation jobs, install Nexus in a preferred location:

::

    git clone https://github.com/QMCPACK/qmcpack.git qmcpack
    export PYTHONPATH=$PYTHONPATH:"$PWD/qmcpack/nexus/lib" 


See
`Nexus documentation <https://nexus-workflows.readthedocs.io/en/latest/installation.html>`_
for more details on additional requirements and recommendations.

In addition, install any software that are going to be interfaced with ``stalk``, such as:

- QMCPACK
- PySCF
- Quantum ESPRESSO

Installation
------------

**Note!** *PyPI installation is on coming up, once the ongoing code revision is done.*

To install ``stalk``, clone the git repository and point PYTHONPATH to the repository root:

::

    git clone https://github.com/QMCPACK/stalk.git stalk
    export PYTHONPATH=$PYTHONPATH:"$PWD/stalk" 
