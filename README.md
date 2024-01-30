# The dompap simulation package

The `dompap` package focuses on simulations of point-like particles in any dimension with any pair potential.
The package uses NumPy and Numba for efficient calculations and SymPy to implement any pair potentials. 
The user is not expected to be familiar with these packages but only basic Python syntax.

## Installation

### Use the Python package index (PyPI)
The package can be installed from the Python package index (PyPI) using pip.
```bash
pip install dompap
```

### Download source from GitHub
Clone the repository from github at https://github.com/urpedersen/dompap.git 
(into a working directory of your choice), and add the package to your python path.
```bash
# Clone repository into some directory (replace [dir])
cd [dir]
git clone https://github.com/urpedersen/dompap.git

# Add to python path
export PYTHONPATH=$PYTHONPATH:[dir]/dompap
```


## Usage example
Below is an example of a 3D system of harmonic repulsive particles with the pair potential
$$v(r) = (1 - r)^2$$
for $r<1$ and zero otherwise. The initial positions are set to a face-centered cubic (fcc) lattice,
with five unit cells in each direction. 
The simulation is run for 100 steps (constant $NVT$ with Langevin thermostat), and the potential energy is printed every 10 steps.

```python
from dompap import Simulation

# Initialize simulation object
sim = Simulation()

# Setup simulation
fcc_unit_cell = ([0.0, 0.0, 0.0], 
                 [0.5, 0.5, 0.0], 
                 [0.5, 0.0, 0.5], 
                 [0.0, 0.5, 0.5])
sim.set_positions(unit_cell_coordinates=fcc_unit_cell,
                  cells=(5, 5, 5), 
                  lattice_constants=(1.0, 1.0, 1.0))
sim.set_density(density=1.0)
sim.set_masses(masses=1.0)
sim.set_random_velocities(temperature=1.0)
sim.set_pair_potential(pair_potential_str='(1-r)**2', 
                       r_cut=1.0,
                       force_method='neighbor list', 
                       energy_method='neighbor list')
sim.set_pair_potential_parameters(sigma=1.0, epsilon=1.0)
sim.set_neighbor_list(skin=0.7, 
                      max_number_of_neighbors=128, 
                      method_str='double loop')
sim.set_integrator(time_step=0.01, 
                   target_temperature=1.0, 
                   temperature_damping_time=0.1)

# Run simulation
steps = 100
for step in range(steps):
    sim.step()
    if step % 10 == 0:
        print(f'Energy after {step} steps: {sim.get_potential_energy()}')
```
This simulation produces the output
```
Energy after 0 steps: 0.0
Energy after 10 steps: 3.1573922419447613
Energy after 20 steps: 16.330136084973663
Energy after 30 steps: 31.47341041787513
Energy after 40 steps: 43.913179017390576
Energy after 50 steps: 52.04197939534787
Energy after 60 steps: 57.968309542867964
Energy after 70 steps: 61.752276744879524
Energy after 80 steps: 67.33278804505039
Energy after 90 steps: 72.00507120397305
```
See [examples](https://github.com/urpedersen/dompap/tree/master/examples) for more examples of the capabilities of the `dompap` package.