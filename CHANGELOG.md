# Change log for `dompap`
## Version 0.0.1
First release of the `dompap` package. The first version can simulations of point-like particles in any dimension with any pair potential.
The package uses NumPy and Numba for efficient calculations and SymPy to implement any pair potentials. 
The user is not expected to be familiar with these packages but only basic Python syntax. Other functionalities are:
* Constant NVT (Langevin) or NVE simulations where forces are calculated 
  - using the neighbor list (multiple cores),
  - using a double loop (multiple cores), or
  - using a double loop (single core).
* Single component systems such as
  - Harmonic repulsive particles in any dimension
  - The Lennard-Jones fluid
  - The WCA fluid
* Multi component systems such as
  - The Wahnström Lennard-Jones mixture
* Autotuner to find optimal parameters for efficient simulations.
* Dump to LAMMPS data file (e.g. visualization in Ovito)
* Compute radial distribution function on the fly
* Save and load simulation state to the disk

## Version 0.0.2
Bugfix:
* `sympy` was missing from  requirements

## Version 0.0.3
Bugfix:
* `scipy` was missing from  requirements
