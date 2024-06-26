from time import perf_counter

import numba
import numpy as np
from matplotlib import pyplot as plt

from dompap.potential import (
    make_pair_potential,
    hardcoded_pair_potentials,
    _get_total_energy,
    _get_total_energy_double_loop,
    _get_forces,
    _get_forces_double_loop_single_core,
    _get_forces_double_loop,
    from_func_to_array,
    _get_forces_vectorized
)


def test_make_pair_potential():
    # Test harmonic repulsive (1-r)**2 for r < 1
    pair_potential, pair_force, pair_curvature = make_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
    assert pair_potential(0.5) == 0.25
    assert pair_force(0.5) == 1.0
    assert pair_curvature(0.5) == 2.0

    # Test Lenard-Jones 4*r**-12-4*r**-6  (infinite cutoff)
    pair_potential, pair_force, pair_curvature = make_pair_potential(pair_potential_str='4*(r**-12-r**-6)',
                                                                     r_cut=np.inf)
    r_min = 2 ** (1 / 6)
    # Assert properties at minimum
    assert np.isclose(pair_potential(r_min), -1.0, atol=1e-6)
    assert np.isclose(pair_potential(1.0), 0.0, atol=1e-6)
    # Force: 48*r**(-13)-24*r**(-7)
    assert np.isclose(pair_force(r_min), 0.0, atol=1e-6)
    assert np.isclose(pair_force(1.0), 48 - 24, atol=1e-6)
    # curvature: 624*r**(-14)-168*r**(-8)
    assert np.isclose(pair_curvature(1.), 624. - 168., atol=1e-6)
    assert np.isclose(pair_curvature(r_min), 57.14643787, atol=1e-6)


def test_hardcoded_potentials(verbose=False, plot=False):
    """  Loop over all hardcoded potentials, plot them, and test that they are equal to make_pair_potential(...) """
    for name, (
    pair_potential_str, pair_potential, pair_force, pair_gradient, r_cut) in hardcoded_pair_potentials.items():
        if verbose:
            print(f'Testing {name}...')
        # Test that hardcoded potential is equal to make_pair_potential(...)
        pair_potential_test, pair_force_test, pair_curvature_test = make_pair_potential(pair_potential_str, r_cut)

        r = np.linspace(0, r_cut * 1.2, 1000)

        # Plot potential
        if plot:
            plt.figure(figsize=(4, 6))
            plt.title(name)
            plt.subplot(3, 1, 1)
            for x in r:
                plt.plot(x, pair_potential(x), 'bo', fillstyle='none')
                plt.plot(x, pair_potential_test(x), 'rx')
            plt.xlabel('r')
            plt.ylim(-1, 1.5)
            plt.ylabel('Pair potential')
            # Plot force
            plt.subplot(3, 1, 2)
            for x in r:
                plt.plot(x, pair_force(x), 'bo', fillstyle='none')
                plt.plot(x, pair_force_test(x), 'rx')
            plt.xlabel('r')
            plt.ylim(-1, 1.5)
            plt.ylabel('Pair force')
            # Plot gradient
            plt.subplot(3, 1, 3)
            for x in r:
                plt.plot(x, pair_gradient(x), 'bo', fillstyle='none')
                plt.plot(x, pair_curvature_test(x), 'rx')
            plt.xlabel('r')
            plt.show()


def test_get_total_energy():
    positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    box_vectors = np.array([3, 3, 3], dtype=np.float64)
    pair_potential, pair_force, pair_curvature = make_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
    neighbor_list = np.array([[1], [0]], dtype=np.int32)
    sigma_func = numba.njit(lambda n, m: np.float64(2))
    epsilon_func = numba.njit(lambda n, m: np.float64(4))
    energy = _get_total_energy(positions, box_vectors, pair_potential, neighbor_list, sigma_func, epsilon_func)
    assert energy == 1.0, f'{energy=}'


def test_get_total_energy_double_loop():
    positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    box_vectors = np.array([3, 3, 3], dtype=np.float64)
    pair_potential, pair_force, pair_curvature = make_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
    sigma_func = numba.njit(lambda n, m: np.float64(2))
    epsilon_func = numba.njit(lambda n, m: np.float64(4))
    energy = _get_total_energy_double_loop(positions, box_vectors, pair_potential, sigma_func, epsilon_func)
    assert energy == 1.0, f'{energy=}'


def test_get_total_energy_double_loop_single_core():
    positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    box_vectors = np.array([3, 3, 3], dtype=np.float64)
    pair_potential, pair_force, pair_curvature = make_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
    sigma_func = numba.njit(lambda n, m: np.float64(2))
    epsilon_func = numba.njit(lambda n, m: np.float64(4))
    energy = _get_total_energy_double_loop(positions, box_vectors, pair_potential, sigma_func, epsilon_func)
    assert energy == 1.0, f'{energy=}'


def test_get_forces():
    positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    box_vectors = np.array([3, 3, 3], dtype=np.float64)
    pair_potential, pair_force, pair_curvature = make_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
    neighbor_list = np.array([[1, -1, -1], [0, -1, -1]], dtype=np.int32)
    sigma_func = numba.njit(lambda n, m: np.float64(2))
    epsilon_func = numba.njit(lambda n, m: np.float64(4))
    forces = _get_forces(positions, box_vectors, pair_force, neighbor_list, sigma_func, epsilon_func)
    assert np.allclose(forces, np.array([[-4, 0, 0], [4, 0, 0]], dtype=np.float64))


def test_get_forces_double_loop_single_core():
    positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    box_vectors = np.array([3, 3, 3], dtype=np.float64)
    pair_potential, pair_force, pair_curvature = make_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
    sigma_func = numba.njit(lambda n, m: np.float64(2))
    epsilon_func = numba.njit(lambda n, m: np.float64(4))
    forces = _get_forces_double_loop_single_core(positions, box_vectors, pair_force, sigma_func, epsilon_func)
    assert np.allclose(forces, np.array([[-4, 0, 0], [4, 0, 0]], dtype=np.float64))


def test_get_forces_double_loop():
    positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    box_vectors = np.array([3, 3, 3], dtype=np.float64)
    pair_potential, pair_force, pair_curvature = make_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
    sigma_func = numba.njit(lambda n, m: np.float64(2))
    epsilon_func = numba.njit(lambda n, m: np.float64(4))
    forces = _get_forces_double_loop(positions, box_vectors, pair_force, sigma_func, epsilon_func)
    assert np.allclose(forces, np.array([[-4, 0, 0], [4, 0, 0]], dtype=np.float64))


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
        print(f'\n    Time first call: {time_first_call * 1e6} microseconds\n')
        print(f'\n    Time second call: {time_second_call * 1e6} microseconds\n')
    assert time_second_call < time_first_call
    assert np.allclose(array1, array2)


def test_get_forces_vectorized():
    positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    box_vectors = np.array([3, 3, 3], dtype=np.float64)
    pair_potential, pair_force, pair_curvature = make_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
    sigma_func = numba.njit(lambda n, m: np.float64(2))
    epsilon_func = numba.njit(lambda n, m: np.float64(4))
    forces = _get_forces_vectorized(positions, box_vectors, pair_force, sigma_func, epsilon_func)
    assert np.allclose(forces, np.array([[-4, 0, 0], [4, 0, 0]], dtype=np.float64))


def test_sigma_and_epsilon_functions():
    """ Test that sigma_func and epsilon_func can be more complex functions """
    # Include four particles
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
    box_vectors = np.array([3, 3, 3], dtype=np.float64)
    pair_potential, pair_force, pair_curvature = make_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)

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
