""" 3D harmonic repulsive system. """

import matplotlib.pyplot as plt
import pandas as pd

from dompap import Simulation, autotune, progress_bar

def plot_sim():
    global sim
    plt.figure(figsize=(6, 6))
    plt.title(f'Time: {sim.get_time():0.3f}, Temperature: {sim.get_temperature():0.3f}')
    plt.plot(sim.positions[:, 0], sim.positions[:, 1], 'o')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

fcc_unit_cell = (
    (0.0, 0.0, 0.0),
    (0.5, 0.5, 0.0),
    (0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5)
)

sim = Simulation()
sim.set_positions(unit_cell_coordinates=fcc_unit_cell, cells=(5, 5, 5))
sim.set_density(density=0.8)
sim.set_masses(masses=1.0)
sim.set_random_velocities(temperature=1.0)
sim.set_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
sim.set_pair_potential_parameters(sigma=1.0, epsilon=1.0)
sim.set_neighbor_list(skin=0.6, max_number_of_neighbors=128, method_str='double loop')
sim.set_integrator(time_step=0.01, target_temperature=1.0, temperature_damping_time=1.0)

print(f'{sim.number_of_particles()=}')
print(f'{sim.get_density()=}')


plot_sim()

# Melt crystal
steps_melt = 200
for step in range(steps_melt):
    sim.step()
    if step % 20 == 0:
        print(f'Melt {step=}: {sim.get_potential_energy()=}')

plot_sim()

# Equilibrate
sim.set_integrator(target_temperature=0.002)
print(sim.temperature_target)
steps_equilibrate = 10_000
for step in range(steps_equilibrate):
    sim.step()
    if step % 500 == 0:
        print(f'Equilibrate {step}/{steps_equilibrate}={step/steps_equilibrate*100:0.1f}%: '
              f'E={sim.get_potential_energy()} T={sim.get_temperature()} P={sim.get_pressure()}')

plot_sim()

print('Autotuning...')
sim = autotune(sim, steps=2_000, verbose=True, plot=True, smallest_skin=0.3, step_skin=0.1, test_double_loop=False)
print(f'{sim.neighbor_list_skin=}')

# Production run
steps = 40_000  # Number of steps to run
stride = 40  # Stride between evaluations
thermodynamic_data = []
for step in range(steps):
    if step % stride == 0:
        progress_bar(step, steps, stride)
        thermodynamic_data.append([sim.get_time(), sim.get_potential_energy(), sim.get_temperature(), sim.get_virial()])
    sim.step()
progress_bar(steps, steps, stride, finalize=True)
columns = ['time', 'potential_energy', 'temperature', 'virial']
df = pd.DataFrame(data=thermodynamic_data, columns=columns)
df['pressure'] = df['temperature'] * sim.get_density() + df['virial'] / sim.get_volume()

print(df.describe())

# Save final state to disk
sim.to_disk(particle_data='harmonic_repulsive_3d.csv', meta_data='harmonic_repulsive_3d.toml')
