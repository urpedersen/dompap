""" Evaluate thermodynamic properties """

import numpy as np
import pandas as pd

from dompap import Simulation

# Setup Lennard-Jones simulation
sim = Simulation()
fcc_unit_cell = np.array([
    (0.0, 0.0, 0.0),
    (0.5, 0.5, 0.0),
    (0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5)
], dtype=np.float64)
sim.set_positions(unit_cell_coordinates=fcc_unit_cell, cells=(4, 4, 4))
sim.set_density(density=0.7)
sim.set_masses(masses=1.0)
sim.set_random_velocities(temperature=0.2)
sim.set_pair_potential(pair_potential_str='4*(r**-12-r**-6)', r_cut=2.5)
sim.set_pair_potential_parameters(sigma=1.0, epsilon=1.0)
sim.set_neighbor_list(skin=0.1, max_number_of_neighbors=128)
sim.set_integrator(time_step=0.004, target_temperature=0.2, temperature_damping_time=10.0)

# Set parameters
steps = 1000  # Number of steps to run
number_of_evaluations = 20  # Number of evaluations
stride = steps // number_of_evaluations  # Stride between evaluations
print(f'Evaluate thermodynamic properties every {stride} steps')

# Run simulation
thermodynamic_data = []
for step in range(steps):
    if step % stride == 0:
        thermodynamic_data.append([sim.get_time(), sim.get_potential_energy(), sim.get_temperature(), sim.get_virial()])
    sim.step()


# Convert data to pandas DataFrame
columns = ['time', 'potential_energy', 'temperature', 'virial']
df = pd.DataFrame(data=thermodynamic_data, columns=columns)

# Calculate pressure from virial and kinetic temperature
density = sim.get_density()
print(f'Density: {density}')
volume = sim.get_volume()
df['pressure'] = df['temperature'] * density + df['virial'] / volume

print(df)

