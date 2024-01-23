""" Compare to values in [J. Chem. Phys. 139, 104102 (2013); https://doi.org/10.1063/1.4818747].
Simulation of the Lennard-Jones crystal truncated at 2.5 sigma.

Values from Table I in the paper, crystal at coexistence state-point:

Temperature: 0.800
Pressure: 2.185

[Crystal]
Specific volume: 1.0277
Energy per particle: −4.953

[Liquid]
Specific volume: 1.1360
Energy per particle: −4.075

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dompap import Simulation, progress_bar, autotune


def run_simulation(sim, target_temperature=0.8, verbose=False):
    sim.set_integrator(time_step=0.004, target_temperature=target_temperature, temperature_damping_time=0.1)

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


def test_lennard_jones_crystal(verbose=False, plot=False):
    # Setup Lennard-Jones simulation
    sim = Simulation()
    fcc_unit_cell = np.array([
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5)
    ], dtype=np.float64)
    sim.set_positions(unit_cell_coordinates=fcc_unit_cell, cells=(6, 6, 6))
    specific_volume = 1.0277
    density = 1.0 / specific_volume
    if verbose:
        print(f'Density: {density}')
    sim.set_density(density=density)
    sim.set_masses(masses=1.0)
    sim.set_particle_types(types=0)
    sim.set_random_velocities(temperature=0.8 * 2)
    sim.set_pair_potential(pair_potential_str='4*(r**-12-r**-6)', r_cut=2.5)
    sim.set_pair_potential_parameters(sigma=1.0, epsilon=1.0)
    sim.set_neighbor_list(skin=1.2, max_number_of_neighbors=512)
    sim.set_integrator(time_step=0.004, target_temperature=0.8, temperature_damping_time=2.0)
    # sim.force_method_str = 'double loop'
    # sim.energy_method_str = 'double loop'
    # sim = autotune(sim, verbose=verbose)

    if verbose:
        print(f'Equilibrate crystal')
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

    average_kinetic_energy = df['kinetic_energy'].mean()
    expected_kinetic_energy = 0.8 * 3 / 2
    tolerance = 0.1 * expected_kinetic_energy
    print(f'{average_kinetic_energy=} {expected_kinetic_energy=}')
    assert expected_kinetic_energy - tolerance < average_kinetic_energy < expected_kinetic_energy + tolerance

    average_pressure = df['pressure'].mean()
    expected_pressure = 2.185
    tolerance = 0.1 * expected_pressure
    print(f'{average_pressure=} {expected_pressure=}')
    assert expected_pressure - tolerance < average_pressure < expected_pressure + tolerance

    average_potential_energy = df['potential_energy'].mean()
    expected_potential_energy = -4.953 - 0.8 * 3 / 2
    tolerance = 0.5
    print(f'{average_potential_energy=} {expected_potential_energy=}')
    assert expected_potential_energy - tolerance < average_potential_energy < expected_potential_energy + tolerance

    average_energy = df['potential_energy'].mean() + df['kinetic_energy'].mean()
    expected_energy = -4.953
    tolerance = 0.5
    print(f'{average_energy=} {expected_energy=}')
    assert expected_energy - tolerance < average_energy < expected_energy + tolerance

    if plot:
        # Plot positions
        plt.figure(figsize=(4, 4))
        plt.plot(sim.positions[:, 0], sim.positions[:, 1], '.')
        plt.axis('equal')
        plt.show()

        # Virial versus potential energy
        plt.figure(figsize=(4, 4))
        x = df['potential_energy']
        y = df['virial']
        plt.plot(x, y, '.')
        # Linear fit
        slope = np.cov(x, y)[0, 1] / np.var(x)
        intersection = np.mean(y) - slope * np.mean(x)
        corr_coef = np.corrcoef(x, y)[0, 1]
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
    test_lennard_jones_crystal(verbose=True, plot=True)
