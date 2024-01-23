""" Lennard-Jones simulation. """
import numpy as np
import matplotlib.pyplot as plt

from dompap import Simulation

# Initialize simulation object
sim = Simulation()

# Setup simulation
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
sim.set_integrator(time_step=0.004, target_temperature=0.2, temperature_damping_time=np.inf)

# Plot pair potential
r = np.linspace(0.90, 3.0, 256)
plt.figure()
plt.title('The Lennard-Jones pair potential')
plt.plot(r, sim.pair_potential(r))
plt.xlabel(r'Pair distance $r$')
plt.ylabel(r'Pair potential $v(r)$')
plt.xlim(0, 3)
plt.ylim(-1.2, 1.2)
plt.show()

# Print some information about the simulation
print('NVE simulation of Lennard-Jones particles in FCC lattice')
print(f'Number of particles: {sim.get_number_of_particles()}')
print(f'Box: {sim.box_vectors}')
print(f'Density: {sim.get_density():0.4}')
print(f'Initial temperature: {sim.get_temperature():0.4}')

# Equilibrate
print('Equilibrating...')
steps = 100
for step in range(steps):
    sim.step()
    if step % 10 == 0:
        print(f'Step {step}: E = {sim.get_potential_energy():0.1f}, T= {sim.get_temperature():0.3f}')

# Production run
print('Running production...')
r_bins = np.linspace(0.1, 3.0, 100)
r, rdf = sim.get_radial_distribution_function(r_bins=r_bins)
frames = 0
steps = 400
time = 0.0
for step in range(steps):
    E_kin = sim.get_kinetic_energy()
    sim.step()
    E_kin = (sim.get_kinetic_energy() + E_kin) / 2
    time += sim.time_step
    if step % 20 == 0:
        E_pot = sim.get_potential_energy()
        print(
            f'Step={step} t={time:0.4f} E={E_pot:0.1f} E_kin={E_kin:0.1f} E_tot={E_kin + E_pot:0.1f} T={sim.get_temperature():0.3f}')
        frames += 1
        r, this_rdf = sim.get_radial_distribution_function(r_bins=r_bins)
        rdf += this_rdf
rdf /= frames + 1

# Plot radial distribution function
plt.figure()
plt.plot(r, rdf, '-')
plt.xlabel(r'Pair distance $r$')
plt.ylabel(r'Radial distribution function $g(r)$')
plt.show()
