from time import perf_counter

from dompap import Simulation

tic = perf_counter()

sim = Simulation()
sim.set_positions(unit_cell_coordinates=([0.0, 0.0, 0.0],), cells=(5, 5, 5), lattice_constants=(1.0, 1.0, 1.0))
sim.set_masses(masses=1.0)
sim.set_random_velocities(temperature=1.0)
sim.set_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
sim.set_pair_potential_parameters(sigma=1.0, epsilon=1.0)
sim.set_neighbor_list(skin=1.0, max_number_of_neighbors=512)
sim.set_integrator(time_step=0.01, target_temperature=1.0, temperature_damping_time=1.0)
sim.set_density(density=1.0)

toc = perf_counter()
print(f'Time to setup: {toc - tic:.2f} seconds')

tic = perf_counter()
sim.step()
toc = perf_counter()
print(f'Time to make first step: {toc - tic:.2f} seconds')

tic = perf_counter()
sim.step()
toc = perf_counter()
print(f'Time to make second step: {(toc - tic) * 1000:.3f} milliseconds')

tic = perf_counter()
# Equilibrate
steps_eq = 100
sim.run(steps_eq)
toc = perf_counter()
print(f'Time to equilibrate: {toc - tic:.2f} seconds')

run_time = 1.0  # seconds
steps_per_round = 1000


def run():
    rounds = 0
    tic = perf_counter()
    while perf_counter() - tic < run_time:
        sim.run(steps_per_round)
        rounds += 1
    toc = perf_counter()
    steps_per_second = rounds * steps_per_round / (toc - tic)
    return steps_per_second


times = []
for _ in range(4):
    times.append(run())
sps = sum(times) / len(times)
sps_std = (sum((sps - t) ** 2 for t in times) / len(times)) ** 0.5
print(f'Average steps per second: {sps:.0f} ± {sps_std:.0f} steps/second')
