## v0.2.0 - 15 Apr 2025

- Comprehensive code refactorization, as overviewed in the following:
- PyPI support
- Isolation and streamlining of Nexus-related functionalities
    - Nexus support is strictly in stalk.nexus module and only available with Nexus
    - The code also works without nexus unless Nexus features are explicitly requested
- Added wrapper classes for core functionalities, to enable checks and enforce API:
    - e.g. PesFunction, NexusGenerator, FittinfFunction, GeometryLoader etc
- Functionality updates in line-search and optimizer, including:
    - Automatic ordering of grid, and omission of grid points
    - Target bias bracketing for improved optimizer performance
- Streamlining of API and script usage, e.g.
    - Object pickling/unpickling
    - Nexus job control; no more need to specify mode of operation
    - Surrogate optimization
- Increased use of @property for better property control in classes and to avoid redundancy
of numerical properties
- More diverse class inheritance chains and type hinting
- Increased and more straightforward unit test coverage
- Revised printing and plotting features and their inheritance
- Additional examples and PySCF support
- Documentation updates

## v0.1 - 11 Nov 2024 and earlier

- A development version used in original or modified variants in selected research projects.