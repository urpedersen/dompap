from time import perf_counter

import numpy as np

from dompap import Simulation, autotune

tic = perf_counter()

sim = Simulation()
sim.set_positions(unit_cell_coordinates=([0.0, 0.0, 0.0],), cells=(10, 10, 10), lattice_constants=(1.0, 1.0, 1.0))
sim.set_masses(masses=1.0)
sim.set_random_velocities(temperature=1.0)
sim.set_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
# sim.set_pair_potential('Harmonic repulsive')  # Test hardcoded potential
# sim.force_method_str = 'vectorized'  # Test how force method affects performance
sim.set_pair_potential_parameters(sigma=1.0, epsilon=1.0)
sim.set_neighbor_list(skin=1.0, max_number_of_neighbors=512, method_str='cell list')
sim.set_integrator(time_step=0.01, target_temperature=1.0, temperature_damping_time=1.0)
sim.set_density(density=1.0)

toc = perf_counter()
print(f'Time to setup: {toc - tic:.2f} seconds')

# Print information about simulation
print(f"""{sim.pair_potential_str=}, 
{sim.pair_potential_r_cut=}
{sim.get_number_of_particles()=} 
{sim.get_volume()=} 
{sim.get_density()=}
{sim.get_temperature()=}
{sim.time_step=}
{sim.force_method_str=}
{sim.neighbor_list_skin=}
{sim.max_number_of_neighbors=}
{sim.neighbor_list_method_str=}""")
print(f'Position of particle 0 is {sim.positions[0]=}')


for _ in range(4):
    tic = perf_counter()
    sim.update_neighbor_list(check=False)
    toc = perf_counter()
    print(f'Time to update neighbor list: {(toc - tic) * 1000:.3f} milliseconds')

for _ in range(4):
    tic = perf_counter()
    sim.update_neighbor_list()
    toc = perf_counter()
    print(f'Time to update neighbor list (check): {(toc - tic) * 1000:.3f} milliseconds')

sim.positions[0] = 0.9, 0.9, 0.9  # Move particle 0
print(f'Move particle 0 to {sim.positions[0]=}')
for _ in range(4):
    tic = perf_counter()
    sim.update_neighbor_list()
    toc = perf_counter()
    print(f'Time to update neighbor list (check): {(toc - tic) * 1000:.3f} milliseconds')

sim.positions[0] = 0.0, 0.0, 0.0  # Move particle 0
print(f'Move particle 0 to {sim.positions[0]=}')
# Test time to compute forces
for _ in range(4):
    tic = perf_counter()
    F = sim.get_forces()
    toc = perf_counter()
    print(f'Time to get forces: {(toc - tic) * 1000:.3f} milliseconds')

sim.positions[0] = 0.9, 0.9, 0.9  # Move particle 0
print(f'Move particle 0 to {sim.positions[0]=}')
# Test time to compute forces
for _ in range(4):
    tic = perf_counter()
    F = sim.get_forces()
    toc = perf_counter()
    print(f'Time to get forces: {(toc - tic) * 1000:.3f} milliseconds')

sim.positions[0] = 0.0, 0.0, 0.0  # Move particle 0
print(f'Move particle 0 to {sim.positions[0]=}')
print('\nRunning simulation, printing time in milliseconds for each step:')
times = []
for step in range(16*8):
    if step % 8 == 0:
        print()
    tic = perf_counter()
    sim.step()
    toc = perf_counter()
    times.append(toc - tic)
    print(f'{(toc - tic) * 1000:8.3f}', end=' ')

average_time = np.mean(times)
print(f'\nAverage time per step: {average_time * 1000:.3f} milliseconds')

number_of_neighbors = np.sum(sim.neighbor_list != -1, axis=1)
print(f'\n{max(number_of_neighbors)=}  {min(number_of_neighbors)=}  {np.mean(number_of_neighbors)=}')

