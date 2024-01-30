from functools import lru_cache
from time import perf_counter

import matplotlib.pyplot as plt
import numba
import numpy
import numpy as np
import sympy as sp

from dompap.positions import get_displacement_vector


@numba.njit
def harmonic_repulsion_pot(r):
    if r < 1:
        return 0.5 * (1 - r) ** 2
    else:
        return 0.0


@numba.njit
def harmonic_repulsion_force(r):
    if r < 1:
        return 1 - r
    else:
        return 0.0


hardcoded_pair_potentials = {
    # Name: (pair_potential_str, energy, force, r_cut)
    'Harmonic repulsive': ('0.5*(1-r)**2',
                           harmonic_repulsion_pot,
                           harmonic_repulsion_force,
                           1.0)
}


def make_pair_potential(pair_potential_str='(((1-r)**2)**(1/2))**(7/2)', r_cut=1.0):
    """ Make pair potential function, and its derivative """
    r = sp.symbols('r')
    # Define pair potential and force
    pair_potential = sp.sympify(pair_potential_str)
    pair_force = sp.diff(-pair_potential, r)
    # Lambdify to numpy functions
    pair_potential = numba.njit(sp.lambdify(r, pair_potential, 'numpy'))
    pair_force = numba.njit(sp.lambdify(r, pair_force, 'numpy'))
    # Shift and truncate at r_cut
    pair_potential_shift = numba.njit(lambda r: pair_potential(r) - pair_potential(r_cut))
    # Let function return zero if r > r_cut
    pair_potential_cut = numba.njit(lambda r: np.where(r > r_cut, 0, pair_potential_shift(r)))
    pair_force_cut = numba.njit(lambda r: np.where(r > r_cut, 0, pair_force(r)))
    return pair_potential_cut, pair_force_cut


def test_make_pair_potential():
    # Test harmonic repulsive (1-r)**2 for r < 1
    pair_potential, pair_force = make_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
    assert pair_potential(0.5) == 0.25
    assert pair_force(0.5) == 1.0


def test_hardcoded_potentials(verbose=False, plot=False):
    """  Loop over all hardcoded potentials, plot them, and test that they are equal to make_pair_potential(...) """
    for name, (pair_potential_str, pair_potential, pair_force, r_cut) in hardcoded_pair_potentials.items():
        if verbose:
            print(f'Testing {name}...')
        # Test that hardcoded potential is equal to make_pair_potential(...)
        pair_potential_test, pair_force_test = make_pair_potential(pair_potential_str, r_cut)

        r = np.linspace(0, r_cut * 1.2, 1000)

        # Plot potential
        if plot:
            plt.figure(figsize=(4, 6))
            plt.title(name)
            plt.subplot(2, 1, 1)
            for x in r:
                plt.plot(x, pair_potential(x), 'bo', fillstyle='none')
                plt.plot(x, pair_potential_test(x), 'rx')
            plt.xlabel('r')
            plt.ylim(-1, 1.5)
            plt.ylabel('Pair potential')
            # Plot force
            plt.subplot(2, 1, 2)
            for x in r:
                plt.plot(x, pair_force(x), 'bo', fillstyle='none')
                plt.plot(x, pair_force_test(x), 'rx')
            plt.xlabel('r')
            plt.ylim(-1, 1.5)
            plt.ylabel('Pair force')
            plt.show()


@numba.njit
def _get_total_energy(positions: np.ndarray,
                      box_vectors: np.ndarray,
                      pair_potential: callable,
                      neighbor_list: np.ndarray,
                      sigma_func: callable,
                      epsilon_func: callable) -> float:
    """ Get total energy of the system """
    dimension_of_space = positions.shape[1]
    energy: float = 0.0
    for n, position in enumerate(positions):
        for m in neighbor_list[n]:
            if m == -1:
                break
            other_position = positions[m]
            # Get distance between particles (hard coded)
            displacement = position - other_position
            # Apply periodic boundary conditions
            for dim in range(dimension_of_space):
                if displacement[dim] > box_vectors[dim] / 2:
                    displacement[dim] -= box_vectors[dim]
                elif displacement[dim] < -box_vectors[dim] / 2:
                    displacement[dim] += box_vectors[dim]
            distance = np.sqrt(np.sum(displacement ** 2))
            sigma = sigma_func(n, m)
            epsilon = epsilon_func(n, m)
            energy = energy + epsilon * pair_potential(distance / sigma).astype(np.float64)
    return energy / 2


def test_get_total_energy():
    positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    box_vectors = np.array([3, 3, 3], dtype=np.float64)
    pair_potential, pair_force = make_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
    neighbor_list = np.array([[1], [0]], dtype=np.int32)
    sigma_func = numba.njit(lambda n, m: np.float64(2))
    epsilon_func = numba.njit(lambda n, m: np.float64(4))
    energy = _get_total_energy(positions, box_vectors, pair_potential, neighbor_list, sigma_func, epsilon_func)
    assert energy == 1.0, f'{energy=}'


@numba.njit(parallel=True)
def _get_total_energy_double_loop(positions: np.ndarray,
                                  box_vectors: np.ndarray,
                                  pair_potential: callable,
                                  sigma_func: callable,
                                  epsilon_func: callable) -> float:
    """ Get total energy of the system using double loop """
    dimension_of_space = positions.shape[1]
    energy: float = 0.0
    number_of_particles = positions.shape[0]
    for n in numba.prange(number_of_particles - 1):
        for m in range(n + 1, number_of_particles):
            displacement = positions[n] - positions[m]
            # Periodic boundary conditions
            for dim in range(dimension_of_space):
                if displacement[dim] > box_vectors[dim] / 2:
                    displacement[dim] -= box_vectors[dim]
                elif displacement[dim] < -box_vectors[dim] / 2:
                    displacement[dim] += box_vectors[dim]
            distance = np.sqrt(np.sum(displacement ** 2))
            sigma = sigma_func(n, m)
            epsilon = epsilon_func(n, m)
            energy = energy + epsilon * pair_potential(distance / sigma).astype(np.float64)
    return energy


def test_get_total_energy_double_loop():
    positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    box_vectors = np.array([3, 3, 3], dtype=np.float64)
    pair_potential, pair_force = make_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
    sigma_func = numba.njit(lambda n, m: np.float64(2))
    epsilon_func = numba.njit(lambda n, m: np.float64(4))
    energy = _get_total_energy_double_loop(positions, box_vectors, pair_potential, sigma_func, epsilon_func)
    assert energy == 1.0, f'{energy=}'


@numba.njit
def _get_total_energy_double_loop_single_core(positions: np.ndarray,
                                  box_vectors: np.ndarray,
                                  pair_potential: callable,
                                  sigma_func: callable,
                                  epsilon_func: callable) -> float:
    """ Get total energy of the system using double loop """
    dimension_of_space = positions.shape[1]
    energy: float = 0.0
    number_of_particles = positions.shape[0]
    for n in numba.prange(number_of_particles - 1):
        for m in range(n + 1, number_of_particles):
            displacement = positions[n] - positions[m]
            # Periodic boundary conditions
            for dim in range(dimension_of_space):
                if displacement[dim] > box_vectors[dim] / 2:
                    displacement[dim] -= box_vectors[dim]
                elif displacement[dim] < -box_vectors[dim] / 2:
                    displacement[dim] += box_vectors[dim]
            distance = np.sqrt(np.sum(displacement ** 2))
            sigma = sigma_func(n, m)
            epsilon = epsilon_func(n, m)
            energy = energy + epsilon * pair_potential(distance / sigma).astype(np.float64)
    return energy


def test_get_total_energy_double_loop_single_core():
    positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    box_vectors = np.array([3, 3, 3], dtype=np.float64)
    pair_potential, pair_force = make_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
    sigma_func = numba.njit(lambda n, m: np.float64(2))
    epsilon_func = numba.njit(lambda n, m: np.float64(4))
    energy = _get_total_energy_double_loop(positions, box_vectors, pair_potential, sigma_func, epsilon_func)
    assert energy == 1.0, f'{energy=}'

@numba.njit(parallel=True)
def _get_forces(positions: np.ndarray,
                box_vectors: np.ndarray,
                pair_force: callable,
                neighbor_list: np.ndarray,
                sigma_func: callable,
                epsilon_func: callable) -> np.ndarray:
    """ Get forces on all particles """
    forces = np.zeros(shape=positions.shape, dtype=np.float64)
    number_of_particles = positions.shape[0]
    for n in numba.prange(number_of_particles):
        positions_n = positions[n]
        for m in neighbor_list[n]:
            if m == -1:
                break
            position_m = positions[m]
            displacement = get_displacement_vector(positions_n, position_m, box_vectors)
            distance = np.sum(displacement ** 2) ** 0.5
            sigma = sigma_func(n, m)
            epsilon = epsilon_func(n, m)
            scalar_force = epsilon * pair_force(distance / sigma)
            unit_vector = displacement / distance
            forces[n] = forces[n] + scalar_force * unit_vector
    return forces


def test_get_forces():
    positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    box_vectors = np.array([3, 3, 3], dtype=np.float64)
    pair_potential, pair_force = make_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
    neighbor_list = np.array([[1, -1, -1], [0, -1, -1]], dtype=np.int32)
    sigma_func = numba.njit(lambda n, m: np.float64(2))
    epsilon_func = numba.njit(lambda n, m: np.float64(4))
    forces = _get_forces(positions, box_vectors, pair_force, neighbor_list, sigma_func, epsilon_func)
    assert np.allclose(forces, np.array([[-4, 0, 0], [4, 0, 0]], dtype=np.float64))


@numba.njit
def _get_forces_double_loop_single_core(positions: np.ndarray,
                                        box_vectors: np.ndarray,
                                        pair_force: callable,
                                        sigma_func: callable,
                                        epsilon_func: callable) -> np.ndarray:
    """ Get forces on all particles using double loop """
    forces = np.zeros(shape=positions.shape, dtype=np.float64)
    number_of_particles = positions.shape[0]
    for n in range(number_of_particles - 1):
        for m in range(n + 1, number_of_particles):
            displacement = get_displacement_vector(positions[n], positions[m], box_vectors)
            distance = np.sum(displacement ** 2) ** 0.5
            sigma = sigma_func(n, m)
            epsilon = epsilon_func(n, m)
            scalar_force = epsilon * pair_force(distance / sigma)
            unit_vector = displacement / distance
            forces[n] = forces[n] + scalar_force * unit_vector
            forces[m] = forces[m] - scalar_force * unit_vector
    return forces


def test_get_forces_double_loop_single_core():
    positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    box_vectors = np.array([3, 3, 3], dtype=np.float64)
    pair_potential, pair_force = make_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
    sigma_func = numba.njit(lambda n, m: np.float64(2))
    epsilon_func = numba.njit(lambda n, m: np.float64(4))
    forces = _get_forces_double_loop_single_core(positions, box_vectors, pair_force, sigma_func, epsilon_func)
    assert np.allclose(forces, np.array([[-4, 0, 0], [4, 0, 0]], dtype=np.float64))


@numba.njit(parallel=True)
def _get_forces_double_loop(positions: np.ndarray,
                            box_vectors: np.ndarray,
                            pair_force: callable,
                            sigma_func: callable,
                            epsilon_func: callable
                            ) -> np.ndarray:
    """ Get forces on all particles using double loop """
    forces = np.zeros(shape=positions.shape, dtype=np.float64)
    number_of_particles = positions.shape[0]
    for n in numba.prange(number_of_particles):
        this_force = np.zeros(shape=positions.shape[1], dtype=np.float64)
        for m in range(number_of_particles):
            if m == n:
                continue
            displacement = positions[n] - positions[m]
            # Periodic boundary conditions
            dimension_of_space = positions.shape[1]
            distance: np.float64 = np.float64(0.0)
            for d in range(dimension_of_space):
                if displacement[d] > box_vectors[d] / 2:
                    displacement[d] -= box_vectors[d]
                elif displacement[d] < -box_vectors[d] / 2:
                    displacement[d] += box_vectors[d]
                distance += displacement[d] ** 2
            distance = distance ** 0.5
            sigma = sigma_func(n, m)
            epsilon = epsilon_func(n, m)
            scalar_force = epsilon * pair_force(distance / sigma)
            unit_vector = displacement / distance
            this_force = this_force + scalar_force * unit_vector
        forces[n] = this_force
    return forces


def test_get_forces_double_loop():
    positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    box_vectors = np.array([3, 3, 3], dtype=np.float64)
    pair_potential, pair_force = make_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
    sigma_func = numba.njit(lambda n, m: np.float64(2))
    epsilon_func = numba.njit(lambda n, m: np.float64(4))
    forces = _get_forces_double_loop(positions, box_vectors, pair_force, sigma_func, epsilon_func)
    assert np.allclose(forces, np.array([[-4, 0, 0], [4, 0, 0]], dtype=np.float64))


@numba.njit(parallel=True)
def _get_virial_double_loop(positions: np.ndarray,
                            box_vectors: np.ndarray,
                            pair_force: callable,
                            sigma_func: callable,
                            epsilon_func: callable) -> np.float64:
    """ Get (pair ) virial of the system using a double loop """
    num_particles = positions.shape[0]
    dimension_of_space = positions.shape[1]
    virial: np.float64 = np.float64(0.0)
    for n in numba.prange(num_particles - 1):
        for m in range(n + 1, num_particles):
            displacement: np.float64 = positions[n] - positions[m]
            # Periodic boundary conditions
            for d in range(dimension_of_space):
                if displacement[d] > box_vectors[d] / 2:
                    displacement[d] -= box_vectors[d]
                elif displacement[d] < -box_vectors[d] / 2:
                    displacement[d] += box_vectors[d]
            distance = np.sum(displacement ** 2) ** 0.5
            sigma = sigma_func(n, m)
            epsilon = epsilon_func(n, m)
            scalar_force = epsilon * pair_force(distance / sigma)
            virial += scalar_force * distance
    return numpy.float64(virial / dimension_of_space)

@lru_cache
def from_func_to_array(func: callable, num_particles: int, dtype=np.float64) -> np.ndarray:
    """ Convert a function to an array by evaluating it at all combinations of indices """
    # Initialize an empty array of the desired shape
    array = np.empty((num_particles, num_particles), dtype=dtype)

    # Iterate over all combinations of indices
    for i in range(num_particles):
        for j in range(num_particles):
            array[i, j] = func(i, j)

    return array
def test_from_func_to_array():
    def func(n, m):
        if n == m:
            return 1.0
        else:
            return 2.0
    num_particles = 3
    array = from_func_to_array(func, num_particles)
    expected = np.array([[1., 2., 2.], [2., 1., 2.], [2., 2., 1.]])
    assert np.allclose(array, expected)

def test_if_cache_works(verbose=False):
    """ Test that cache works """
    def func(n, m):
        if n == m:
            return 1.0
        else:
            return 2.0
    num_particles = 4
    tic = perf_counter()
    array1 = from_func_to_array(func, num_particles)
    toc = perf_counter()
    time_first_call = toc - tic
    tic = perf_counter()
    array2 = from_func_to_array(func, num_particles)
    toc = perf_counter()
    time_second_call = toc - tic
    if verbose:
        print(f'\n    Time first call: {time_first_call*1e6} microseconds\n')
        print(f'\n    Time second call: {time_second_call*1e6} microseconds\n')
    assert time_second_call < time_first_call
    assert np.allclose(array1, array2)

def _get_forces_vectorized(positions, box_vectors, pair_force, sigma_func, epsilon_func):
    """ Get forces on all particles using vectorized operations.
    This implementation uses only NumPy, no Numba. """
    displacement_vectors = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    for dim in range(positions.shape[1]):
        displacement_vectors[:, :, dim] = np.where(displacement_vectors[:, :, dim] > box_vectors[dim] / 2,
                                                   displacement_vectors[:, :, dim] - box_vectors[dim],
                                                   displacement_vectors[:, :, dim])
        displacement_vectors[:, :, dim] = np.where(displacement_vectors[:, :, dim] < -box_vectors[dim] / 2,
                                                   displacement_vectors[:, :, dim] + box_vectors[dim],
                                                   displacement_vectors[:, :, dim])
    distances = np.sum(displacement_vectors ** 2, axis=2) ** 0.5
    np.fill_diagonal(distances, np.inf)  # Set diagonal to infinity to avoid division by zero
    sigmas = from_func_to_array(sigma_func, positions.shape[0])
    epsilons = from_func_to_array(epsilon_func, positions.shape[0])
    scalar_forces = epsilons * pair_force(distances / sigmas)
    unit_vectors = displacement_vectors / distances[:, :, np.newaxis]
    forces = np.sum(scalar_forces[:, :, np.newaxis] * unit_vectors, axis=1)
    return forces

def test_get_forces_vectorized():
    positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    box_vectors = np.array([3, 3, 3], dtype=np.float64)
    pair_potential, pair_force = make_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
    sigma_func = numba.njit(lambda n, m: np.float64(2))
    epsilon_func = numba.njit(lambda n, m: np.float64(4))
    forces = _get_forces_vectorized(positions, box_vectors, pair_force, sigma_func, epsilon_func)
    assert np.allclose(forces, np.array([[-4, 0, 0], [4, 0, 0]], dtype=np.float64))

def test_sigma_and_epsilon_functions():
    """ Test that sigma_func and epsilon_func can be more complex functions """
    # Include four particles
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
    box_vectors = np.array([3, 3, 3], dtype=np.float64)
    pair_potential, pair_force = make_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)

    # Make everybody neighbors
    neighbor_list = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=np.int32)

    @numba.njit
    def sigma_func(n, m):
        if n == m:
            return np.float64(1)
        else:
            return np.float64(2)

    @numba.njit
    def epsilon_func(n, m):
        if n == m:
            return np.float64(1)
        else:
            return np.float64(4)

    F_expected = _get_forces(positions, box_vectors, pair_force, neighbor_list, sigma_func, epsilon_func)

    list_of_other_funcs = [
        _get_forces_double_loop_single_core,
        _get_forces_double_loop,
        _get_forces_vectorized
    ]
    for other_func in list_of_other_funcs:
        F_other = other_func(positions, box_vectors, pair_force, sigma_func, epsilon_func)
        assert np.allclose(F_expected, F_other)

