import numpy as np
import matplotlib.pyplot as plt

from dompap import Simulation
from dompap.tools import autotune


# Setup Lennard-Jones simulation
sim = Simulation()
fcc_unit_cell = np.array([
    (0.0, 0.0, 0.0),
    (0.5, 0.5, 0.0),
    (0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5)
], dtype=np.float64)
temperature = 0.7
sim.set_positions(unit_cell_coordinates=fcc_unit_cell, cells=(5, 5, 5))
sim.set_density(density=1/1.0452)
sim.set_masses(masses=1.0)
sim.set_random_velocities(temperature=temperature*2)
sim.set_pair_potential(pair_potential_str='4*(r**-12-r**-6)', r_cut=2.5)
sim.set_pair_potential_parameters(sigma=1.0, epsilon=1.0)
sim.set_neighbor_list(skin=0.6, max_number_of_neighbors=128)
sim.set_integrator(time_step=0.004, target_temperature=temperature, temperature_damping_time=0.5)

# Equilibrate
sim.run(200)

# Run simulation
times = []
E_pots = []
T_confs = []
T_kins = []
pressures = []
N = sim.get_number_of_particles()
steps = 1000
for step in range(steps):
    if step % (steps//100) == 0:
        times.append(sim.get_time())
        pressure = sim.get_pressure()
        print(f'Pressure: {pressure}')
        pressures.append(pressure)
        print(f'  Step {step}:')
        E_pot = sim.get_potential_energy()
        E_pots.append(E_pot)
        print(f'Energy per particle         : {E_pot/N}')
        T_conf = sim.get_configurational_temperature()
        T_confs.append(T_conf)
        print(f'Configurational temperature : {T_conf}')
        T_kin = sim.get_temperature()
        T_kins.append(T_kin)
        print(f'Kinetic temperature         : {sim.get_temperature()}')

        #F = sim.get_forces()
        #print(f'Average of force square: {np.sum(F**2)/N}')
        #lap = sim.get_laplacian()
        #print(f'Average of particle laplacians: {np.sum(lap)/N}')
    sim.step()

# Print summary statistics
print(f'Average potential energy per particle: U = {np.mean(E_pots)/N}')
# from litterature −5.156
u_litterature = -5.156
print(f'{u_litterature = }')
print(f'Pressure = {np.mean(pressures)}')
p_litterature = 0.928
print(f'{p_litterature = }')
print(f'Average configurational temperature: T_c = {np.mean(T_confs)}')
print(f'Average kinetic temperature: T_k = {np.mean(T_kins)}')
print(f'T_k/T_c ratio: {np.mean(T_kins)/np.mean(T_confs)}')

# Print standard deviations
print(f'Standard deviation of potential energy per particle: U = {np.std(E_pots)/N}')
print(f'Standard deviation of configurational temperature: T_c = {np.std(T_confs)}')
print(f'Standard deviation of kinetic temperature: T_k = {np.std(T_kins)}')


# Plot configurational temperature and kinetic temperature
plt.figure()
plt.plot(times, T_confs, label='Configurational temperature')
plt.plot(times, np.array(T_confs)*2.0, label='Configurational temperature x2')
plt.plot(times, T_kins, label='Kinetic temperature')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.xlabel('Time')
plt.show()

# Plot positions
plt.figure()
plt.plot(sim.positions[:, 0], sim.positions[:, 1], 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
