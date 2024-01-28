""" Two rods with harmonic repulsive potential in 1D.
    This example shows how to simulate two rods with harmonic repulsive potential in 1D.
    The pair distance between the rods is monitored during the equilibration run, and
    the pair distance histogram is plotted in a production run.

    Output example:

    ```
    sim.number_of_particles()=2
    sim.get_density()=0.8
    sim.pair_potential_str='(1-r)**2'
    Initial positions: sim.positions[0]=array([0.]) sim.positions[1]=array([1.25])
    Box length: L=2.5
    Total number of steps: 100010
    2024-01-28 11:54:26.524871            0 |....:....|....:....|....:....|....:....|....:....|
    2024-01-28 11:54:28.275120        5,000 |....:....|....:....|....:....|....:....|....:....|
    2024-01-28 11:54:29.977057       10,000 |  done
    ```
"""

import numpy as np
import matplotlib.pyplot as plt

from dompap import Simulation, autotune, progress_bar

# Create a simulation with two rods
temperature = 0.01
sim = Simulation()
sim.set_positions(unit_cell_coordinates=((0.0,),), cells=(2,), lattice_constants=(1.0,))
sim.set_density(density=0.80)
sim.set_masses(masses=1)
sim.set_random_velocities(temperature=temperature)
sim.set_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0,
                       force_method='double loop single core', energy_method='double loop single core')
sim.set_pair_potential_parameters(sigma=1.0, epsilon=1.0)
sim.set_integrator(time_step=0.01, target_temperature=temperature, temperature_damping_time=1.0)

# Print information about the simulation
print(f'{sim.number_of_particles()=}')
print(f'{sim.get_density()=}')
print(f'{sim.pair_potential_str=}')
print(f'Initial positions: {sim.positions[0]=} {sim.positions[1]=}')

# sim = autotune(sim, steps=100, verbose=True, plot=True)  # Autotune parameters

L = float(sim.box_vectors[0])
print(f'Box length: {L=}')

# Equilibration run
pair_distances = []
number_of_steps = 8000
for _ in range(number_of_steps):
    sim.step()
    dr = float(sim.positions[1] - sim.positions[0])
    if dr > L / 2:
        dr -= L
    elif dr < -L / 2:
        dr += L
    dr = abs(dr)
    pair_distances.append(dr)

times = sim.time_step * np.arange(number_of_steps)

plt.figure(figsize=(5, 3))
plt.title(f'Equilibration run, T={temperature}')
plt.plot(times, pair_distances, '+')
plt.plot([0, times[-1]], [1, 1], '--', color='black')
plt.xlabel(r'Time, $t$')
plt.ylabel(r'Pair distance, $r=|r_1-r_0|$')
plt.xlim(0, times[-1] + sim.time_step)
plt.ylim(0.7, L / 2)
plt.show()

# Production run
pair_distances = []
number_of_steps = 10_000 + 1
inner_steps = 10
stride = 100
print(f'Total number of steps: {number_of_steps * inner_steps}')
for step in range(number_of_steps):
    if step % stride == 0:
        progress_bar(step, number_of_steps, stride)
    sim.run(inner_steps)
    dr = float(sim.positions[1] - sim.positions[0])
    if dr > L / 2:
        dr -= L
    elif dr < -L / 2:
        dr += L
    dr = abs(dr)
    pair_distances.append(dr)
progress_bar(number_of_steps, number_of_steps, stride, finalize=True)

bins = np.linspace(0, L / 2, 200)
bins_center = bins[:-1] + 0.5 * (bins[1] - bins[0])
histogram = np.histogram(pair_distances, bins=bins, density=True)

# Plot pair distance histogram
plt.figure(figsize=(5, 3))
plt.title(f'Production run, T={temperature}')
plt.plot(bins_center, histogram[0], '+')
plt.plot([1, 1], [0, 1.1 * max(histogram[0])], '--', color='black')
plt.xlabel(r'Pair distance, $r=|r_1-r_0|$')
plt.ylabel(r'Probability density')
plt.xlim(0.7, L / 2)
plt.ylim(0, 1.1 * max(histogram[0]))
plt.show()
