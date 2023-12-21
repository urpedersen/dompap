import matplotlib.pyplot as plt
import numba
import numpy as np
import sympy as sp

from .positions import get_distance, get_displacement_vector


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
    return "Test passed: make_pair_potential() is as expected."


def _get_total_energy(positions: np.ndarray,
                      box_vectors: np.ndarray,
                      pair_potential: callable,
                      neighbor_list: np.ndarray,
                      sigmas: np.ndarray,
                      epsilons: np.ndarray) -> float:
    """ Get total energy of the system """
    energy: float = 0.0
    for n, position in enumerate(positions):
        for m in neighbor_list[n]:
            if m == -1:
                break
            other_position = positions[m]
            distance = get_distance(position, other_position, box_vectors)
            sigma = (sigmas[n] + sigmas[m]) / 2
            epsilon = np.sqrt(epsilons[n] * epsilons[m])
            energy = energy + epsilon * np.float64(pair_potential(distance / sigma))
    return energy / 2


def test_get_total_energy():
    positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    box_vectors = np.array([3, 3, 3], dtype=np.float64)
    pair_potential, pair_force = make_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
    neighbor_list = np.array([[1], [0]], dtype=np.int32)
    sigmas = np.array([[2], [2]], dtype=np.float64)
    epsilons = np.array([[4], [4]], dtype=np.float64)
    energy = _get_total_energy(positions, box_vectors, pair_potential, neighbor_list, sigmas, epsilons)
    assert energy == 1.0


@numba.njit
def _get_forces(positions: np.ndarray,
                box_vectors: np.ndarray,
                pair_force: callable,
                neighbor_list: np.ndarray,
                sigmas: np.ndarray,
                epsilons: np.ndarray) -> np.ndarray:
    """ Get forces on all particles """
    forces = np.zeros(shape=positions.shape, dtype=np.float64)
    number_of_particles = positions.shape[0]
    for n in range(number_of_particles):
        positions_n = positions[n]
        for m in neighbor_list[n]:
            if m == -1:
                break
            position_m = positions[m]
            distance = get_distance(positions_n, position_m, box_vectors)
            sigma = (sigmas[n] + sigmas[m]) / 2
            epsilon = np.sqrt(epsilons[n] * epsilons[m])
            scalar_force = epsilon * pair_force(distance / sigma).astype(np.float64)
            displacement = get_displacement_vector(positions_n, position_m, box_vectors)
            unit_vector = displacement / distance
            forces[n] = forces[n] + scalar_force * unit_vector
    return forces


def test_get_forces():
    positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    box_vectors = np.array([3, 3, 3], dtype=np.float64)
    pair_potential, pair_force = make_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
    neighbor_list = np.array([[1, -1, -1], [0, -1, -1]], dtype=np.int32)
    sigmas = np.array([[2], [2]], dtype=np.float64)
    epsilons = np.array([[4], [4]], dtype=np.float64)
    forces = _get_forces(positions, box_vectors, pair_force, neighbor_list, sigmas, epsilons)
    assert np.allclose(forces, np.array([[-4, 0, 0], [4, 0, 0]], dtype=np.float64))
