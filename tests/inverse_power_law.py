""" Inverse power law pair potential """

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dompap import Simulation
from dompap import progress_bar
import dompap


def run_simulation(sim, target_temperature=0.8, verbose=False):
    sim.set_integrator(time_step=0.004, target_temperature=target_temperature, temperature_damping_time=2.0)

    # Set simulation parameters
    steps = 400  # Number of steps to run
    number_of_evaluations = 100  # Number of evaluations
    stride = steps // number_of_evaluations  # Stride between evaluations

    # Run simulation
    thermodynamic_data = []
    for step in range(steps):
        if step % stride == 0:
            if verbose:
                progress_bar(step, steps, stride)
            N = sim.get_number_of_particles()
            thermodynamic_data.append(
                [sim.get_time(),
                 sim.get_potential_energy() / N,
                 sim.get_temperature(),
                 sim.get_kinetic_energy() / N,
                 sim.get_virial() / N,
                 sim.get_pressure()])
        if sim.get_virial() < -0.1:
            r = sim.get_positions()
            W = sim.get_virial()
            print(f'{W=}')
            F = sim.get_forces()
            ...
        sim.step()

    if verbose:
        progress_bar(steps, steps, stride, finalize=True)

    # Convert data to pandas DataFrame
    columns = ['time', 'potential_energy', 'temperature', 'kinetic_energy', 'virial', 'pressure']
    df = pd.DataFrame(data=thermodynamic_data, columns=columns)

    # Compute derivatives
    density = sim.get_density()
    volume = sim.get_volume()
    # number_of_particles = sim.get_number_of_particles()
    df['pressure_recalc'] = df['temperature'] * density + df['virial'] / volume

    return sim, df


def test_inverse_power_law(verbose=False, plot=False):
    # Setup Lennard-Jones simulation
    sim = Simulation()
    fcc_unit_cell = np.array([
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5)
    ], dtype=np.float64)
    sim.set_positions(unit_cell_coordinates=fcc_unit_cell, cells=(6, 6, 6))
    density = 0.7
    if verbose:
        print(f'Density: {density}')
    sim.set_density(density=density)
    sim.set_masses(masses=1.0)
    sim.set_particle_types(types=0)
    sim.set_random_velocities(temperature=1.0)
    sim.set_pair_potential(pair_potential_str='r**-12', r_cut=2.0)
    sim.set_pair_potential_parameters(sigma=1.0, epsilon=1.0)
    sim.set_neighbor_list(skin=1.2, max_number_of_neighbors=512)
    sim.set_integrator(time_step=0.004, target_temperature=1.0, temperature_damping_time=2.0)
    # sim.force_method_str = 'double loop'
    # sim.energy_method_str = 'double loop'
    # sim = dompap.autotune(sim, verbose=verbose)

    if plot:
        # Plot pair potential
        r = np.linspace(0.5, 2.5, 1000)
        plt.figure(figsize=(4, 6))
        plt.subplot(2, 1, 1)
        plt.plot(r, sim.pair_potential(r))
        plt.xlabel('r')
        plt.ylabel('Pair potential, v(r)')
        plt.ylim(0, 2)
        plt.subplot(2, 1, 2)
        # Plot pair force
        plt.plot(r, sim.pair_force(r))
        plt.xlabel('r')
        plt.ylabel('Pair force, f(r)')
        plt.ylim(-1, 1.2)
        plt.show()

    if verbose:
        print(f'Equilibrate')
    sim, _ = run_simulation(sim, verbose=verbose)  # Equilibrate
    if verbose:
        print(f'Production run')
    sim, df = run_simulation(sim, verbose=verbose)  # Production run

    print(f'{df["potential_energy"].mean()=}')
    print(f'{df["kinetic_energy"].mean()=}')
    print(f'{df["virial"].mean()=}')
    print(f'{df["temperature"].mean()=}')
    print(f'{df["pressure"].mean()=}')
    print(f'{sim.get_volume()=}')
    print(f'{sim.get_number_of_particles()=}')

    # Assert correlation between virial and potential energy
    x = df['potential_energy']
    y = df['virial']
    slope = np.cov(x, y)[0, 1] / np.var(x)
    intersection = np.mean(y) - slope * np.mean(x)
    corr_coef = np.corrcoef(x, y)[0, 1]
    expected_slope = 4
    expected_intersection = 0
    expected_corr_coef = 1
    tolerance = 0.1
    assert expected_slope - tolerance < slope < expected_slope + tolerance, f'{slope=}'
    assert expected_intersection - tolerance < intersection < expected_intersection + tolerance, f'{intersection=}'
    assert expected_corr_coef - tolerance < corr_coef < expected_corr_coef + tolerance, f'{corr_coef=}'

    if plot:
        # Plot positions
        plt.figure(figsize=(4, 4))
        plt.plot(sim.positions[:, 0], sim.positions[:, 1], '.')
        plt.axis('equal')
        plt.show()

        # Virial versus potential energy
        plt.figure(figsize=(4, 4))
        plt.plot(x, y, '.')
        plt.plot([min(x), max(x)],
                 [slope * min(x) + intersection, slope * max(x) + intersection],
                 '-')  # y = ax + b
        plt.title(f'{slope=:.3f} {corr_coef=:.3f}')
        plt.xlabel('Potential energy')
        plt.ylabel('Virial')
        plt.show()

        # Thermodynamic data
        plt.figure(figsize=(6, 10))
        plt.subplot(4, 1, 1)
        plt.plot(df['time'], df['potential_energy'])
        plt.ylabel('Potential energy')
        plt.subplot(4, 1, 2)
        plt.plot(df['time'], df['temperature'])
        plt.ylabel('Temperature')
        plt.subplot(4, 1, 3)
        plt.plot(df['time'], df['pressure'])
        plt.ylabel('Pressure')
        plt.subplot(4, 1, 4)
        plt.plot(df['time'], df['virial'])
        plt.ylabel('Virial')
        plt.xlabel('Time')
        plt.subplots_adjust(hspace=0)
        plt.show()


if __name__ == '__main__':
    test_inverse_power_law(verbose=True, plot=True)
