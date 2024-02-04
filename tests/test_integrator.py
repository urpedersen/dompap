import numpy as np

from dompap.integrator import make_one_step, make_one_step_leap_frog


def test_energy_conservation(verbose=False):
    """ Make a simulation of four particles with gravitational forces """
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
                         dtype=np.float64)
    velocities = np.array([[-0.3, -0.1, 0.1], [0.2, 0.1, 0], [-0.2, -0.2, 0.1], [0.3, 0.2, -0.1]],
                          dtype=np.float64)
    masses = np.array([[1.0], [2.0], [1.5], [0.5]], dtype=np.float64)
    betas = np.zeros_like(velocities)
    time_step = np.float64(0.001)
    temperature_target = np.float64(1.0)
    temperature_damping_time = np.inf

    def compute_gravitational_forces(pos, mass):
        num_particles = pos.shape[0]
        out = np.zeros_like(pos)
        for i in range(num_particles):
            for j in range(num_particles):
                if i != j:
                    r_vec = pos[j] - pos[i]
                    r_mag = np.linalg.norm(r_vec)
                    force_mag = mass[i] * mass[j] / r_mag ** 2
                    out[i] += force_mag * r_vec / r_mag
        return out

    def calculate_potential_energy(pos, mass):
        out = 0.0
        num_particles = pos.shape[0]
        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                r_vec = pos[j] - pos[i]
                r_mag = np.linalg.norm(r_vec)
                out -= mass[i] * mass[j] / r_mag
        return out

    def calculate_kinetic_energy(vel, mass):
        out = 0.0
        num_particles = vel.shape[0]
        for i in range(num_particles):
            out += 0.5 * mass[i] * np.linalg.norm(vel[i]) ** 2
        return out

    # Test Leap-Frog
    if verbose:
        print('Testing Leap-Frog')
    data = []
    for _ in range(100):
        kinetic_energy = calculate_kinetic_energy(velocities, masses)
        forces = compute_gravitational_forces(positions, masses)
        positions, velocities = make_one_step_leap_frog(positions, velocities, forces, masses, time_step)
        potential_energy = calculate_potential_energy(positions, masses)
        kinetic_energy = 0.5 * (kinetic_energy + calculate_kinetic_energy(velocities, masses))
        total_energy = potential_energy + kinetic_energy
        data.append([float(potential_energy), float(kinetic_energy), float(total_energy)])
        if verbose:
            print(data[-1])
    data = np.array(data, dtype=float)
    if verbose:
        print(f'Std of Total Energy fluctuation: {np.std(data, axis=0)[2]}')
    assert np.std(data, axis=0)[2] < 1e-2, f'Std of Total Energy fluctuation: {np.std(data, axis=0)[2]}'

    # Test G-JF thermostat
    data = []
    if verbose:
        print('Testing G-JF thermostat')
    for _ in range(100):
        forces = compute_gravitational_forces(positions, masses)
        positions, velocities, _ = make_one_step(positions, velocities, betas, forces, masses,
                                                 time_step, temperature_target, temperature_damping_time)
        potential_energy = calculate_potential_energy(positions, masses)
        kinetic_energy = calculate_kinetic_energy(velocities, masses)
        total_energy = potential_energy + kinetic_energy
        data.append([float(potential_energy), float(kinetic_energy), float(total_energy)])
        if verbose:
            print(data[-1])
    data = np.array(data, dtype=float)
    if verbose:
        print(f'Std of Total Energy fluctuation: {np.std(data, axis=0)[2]}')
    assert np.std(data, axis=0)[2] < 1e-2, f'Std of Total Energy fluctuation: {np.std(data, axis=0)[2]}'


def test_make_step():
    positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    velocities = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float64)
    betas = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float64)
    forces = np.array([[-1, 0, 0], [1, 0, 0]], dtype=np.float64)
    masses = np.array([[1], [1]], dtype=np.float64)
    time_step = np.float64(0.001)
    temperature_target = np.float64(1.0)
    temperature_damping_time = np.float64(1.0)
    new_state = make_one_step(positions, velocities, betas, forces, masses, time_step, temperature_target,
                              temperature_damping_time)
    positions_new, velocities_new, betas_new = new_state
    # Assert that the first particle has moved
    assert positions_new[0, 0] != 0.0

