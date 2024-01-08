from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dompap import Simulation, autotune


def benchmark(nx=5, verbose=False, plot=False):
    sim = Simulation()
    sim.set_positions(unit_cell_coordinates=([0.0, 0.0, 0.0],), cells=(nx, nx, nx), lattice_constants=(1.0, 1.0, 1.0))
    sim.set_masses(masses=1.0)
    sim.set_random_velocities(temperature=1.0)
    sim.set_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
    sim.set_pair_potential_parameters(sigma=1.0, epsilon=1.0)
    sim.set_neighbor_list(skin=1.0, max_number_of_neighbors=512)
    sim.set_integrator(time_step=0.01, target_temperature=1.0, temperature_damping_time=1.0)
    sim.set_density(density=1.0)
    sim = autotune(sim, verbose=verbose, plot=plot)

    # Equilibrate
    steps_eq = 10
    sim.run(steps_eq)

    max_run_time = 5.0  # seconds
    steps_per_round = 500

    times = []
    start = perf_counter()
    while perf_counter() - start < max_run_time:
        tic = perf_counter()
        sim.run(steps_per_round)
        toc = perf_counter()
        times.append(toc - tic)

    number_of_particles = sim.get_number_of_particles()
    seconds_per_step = sum(times) / len(times)
    err_seconds_per_step = np.std(times) / np.sqrt(len(times))

    if verbose:
        print(f'{number_of_particles=}, {len(times)=}, {seconds_per_step=:.3f} +/- {err_seconds_per_step=:.3f} seconds/step')

    return number_of_particles, seconds_per_step, err_seconds_per_step

sizes, times, times_err = [], [], []

for nx in 5, 6, 7, 8, 9, 10:
    number_of_particles, seconds_per_step, err_seconds_per_step = benchmark(nx, verbose=True)
    sizes.append(number_of_particles)
    times.append(seconds_per_step)
    times_err.append(err_seconds_per_step)

plt.figure(figsize=(4, 4))
plt.errorbar(sizes, times, yerr=times_err, fmt='o')
plt.xlabel('Number of particles')
plt.ylabel('Seconds per step')
plt.xscale('log')
plt.yscale('log')
plt.savefig('system_sizes.png', dpi=300)
plt.show()

# Save data to cvs file
df = pd.DataFrame(data={'size': sizes, 'time': times, 'time_err': times_err})
df.to_csv('system_sizes.csv', index=False)