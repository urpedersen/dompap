"""
dompap
======

Simulations of point-like particles in any dimension with any pair potential.
The package uses NumPy and Numba for efficient calculations and SymPy to implement any pair potentials.
The user is not expected to be familiar with these packages but only basic Python syntax.

Repository: <https://github.com/urpedersen/dompap>
"""
from .Simulation import Simulation
from .to_lammps_dump import to_lammps_dump
from .main import main
from . import tools
