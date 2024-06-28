import numpy as np
import matplotlib.pyplot as plt

from dompap import Simulation

# Setup simulation
sim = Simulation()
fcc_unit_cell = (
    (0.0, 0.0, 0.0),
    (0.5, 0.5, 0.0),
    (0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5)
)
temperature = 0.8
sim.set_positions(unit_cell_coordinates=fcc_unit_cell, cells=(5, 5, 5))
specific_volume = 1.0277
density = 1.0 / specific_volume
sim.set_density(density=density)
sim.set_masses(masses=1.0)
sim.set_random_velocities(temperature=temperature * 2)
sim.set_pair_potential(pair_potential_str='4*(r**-12-r**-6)', r_cut=2.5)
sim.set_pair_potential_parameters(sigma=1.0, epsilon=1.0)
sim.set_neighbor_list(skin=0.6, max_number_of_neighbors=128)
sim.set_integrator(time_step=0.004, target_temperature=temperature, temperature_damping_time=2.0)

# Equilibrate
steps_for_equilibration = 1000
sim.run(steps_for_equilibration)

# Run simulation
times = []
T_confs = []
T_kins = []
N = sim.get_number_of_particles()
steps = 1000
num_points = 100
for step in range(steps):
    if step % (steps // num_points) == 0:
        print(f'  Step {step}:')
        times.append(sim.get_time())
        T_conf = sim.get_configurational_temperature()
        T_confs.append(T_conf)
        T_kin = sim.get_temperature()
        T_kins.append(T_kin)
        print(f'{T_conf = }, {T_kin = }')
    sim.step()

# Print summary statistics
print(f'Average configurational temperature: T_c = {np.mean(T_confs)}')
print(f'Average kinetic temperature: T_k = {np.mean(T_kins)}')
print(f'T_k/T_c ratio: {np.mean(T_kins) / np.mean(T_confs)}')

# Print standard deviations
print(f'Standard deviation of configurational temperature: T_c = {np.std(T_confs)}')
print(f'Standard deviation of kinetic temperature: T_k = {np.std(T_kins)}')

# Plot configurational temperature and kinetic temperature
plt.figure()
plt.plot(times, T_confs, label='Configurational temperature')
plt.plot(times, T_kins, label='Kinetic temperature')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.xlabel('Time')
plt.show()
