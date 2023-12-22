import numba
import numpy as np

from .positions import get_distance, get_displacement_vector


@numba.njit
def get_neighbor_list(positions, box_vectors, cutoff_distance, max_number_of_neighbors) -> np.ndarray:
    """ Get neighbour list using a N^2 loop (backend) """
    number_of_particles: int = positions.shape[0]

    neighbor_list = np.zeros(shape=(number_of_particles, max_number_of_neighbors), dtype=np.int32) - 1  # -1 is empty
    for n, position in enumerate(positions):
        current_idx = 0
        for m, other_position in enumerate(positions):
            if n == m:
                continue
            distance = get_distance(position, other_position, box_vectors)
            if distance < cutoff_distance:
                neighbor_list[n][current_idx] = m
                current_idx += 1
    return neighbor_list


@numba.njit
def neighbor_list_is_old(positions: np.ndarray,
                         old_positions: np.ndarray,
                         box_vectors: np.ndarray,
                         skin: np.float64) -> bool:
    """ Check if the neighbor list needs to be updated, i.e, if a particle hase moved more than the skin distance. """
    number_of_particles: int = positions.shape[0]
    max_distance_squared: float = 0.0
    for n in range(number_of_particles):
        displacement = get_displacement_vector(positions[n], old_positions[n], box_vectors)
        distance_squared = np.sum(displacement ** 2)
        if distance_squared > max_distance_squared:
            max_distance_squared = distance_squared
    max_allowed_distance_squared = (0.5 * skin) ** 2
    return max_distance_squared > max_allowed_distance_squared


def test_neighbor_list_is_old():
    positions = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float64)
    old_positions = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float64)
    box_vectors = np.array([5, 5, 5], dtype=np.float64)
    skin = np.float64(0.5)
    assert neighbor_list_is_old(positions, old_positions, box_vectors, skin) == False
    old_positions[0] = [skin/2 - 0.01, 0.0, 0.0]  # Particle 0 has moved less 0.01 larger half the skin
    assert neighbor_list_is_old(positions, old_positions, box_vectors, skin) == False
    old_positions[0] = [skin/2 + 0.01, 0.0, 0.0]  # Particle 0 has moved 0.01 larger half the skin
    assert neighbor_list_is_old(positions, old_positions, box_vectors, skin) == True
    old_positions[0] = [0.0 + box_vectors[0], 0.0, 0.0]  # Particle 0 has moved one box length to its own image
    assert neighbor_list_is_old(positions, old_positions, box_vectors, skin) == False

def test_neighbor_list_is_old_skin_range():
    positions = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
    old_positions = np.array([[0.5, 0, 0], [1, 1, 1]], dtype=np.float64)  # Particle 0 has moved 0.5 in x-direction
    box_vectors = np.array([5, 5, 5], dtype=np.float64)
    # Test range of skins, where the neighbor list should be not be old
    for skin in np.linspace(0.01, 0.99, 10, dtype=np.float64):
        assert neighbor_list_is_old(positions, old_positions, box_vectors, skin) == True, f'skin={skin}'

    # Test range of skins, where the neighbor list should be old
    for skin in np.linspace(1.01, 1.5, 10, dtype=np.float64):
        assert neighbor_list_is_old(positions, old_positions, box_vectors, skin) == False, f'skin={skin}'


@numba.njit
def get_number_of_neighbors(neighbor_list):
    """ Get number of neighbours for each particle """
    return np.sum(neighbor_list != -1, axis=1)

def test_get_number_of_neighbors_all_minus_one():
    neighbor_list = np.array([[-1, -1, -1], [-1, -1, -1]], dtype=np.int32)
    assert np.all(get_number_of_neighbors(neighbor_list) == 0)  # No particles have neighbors

def test_get_number_of_neighbors():
    neighbor_list = np.array([[1, 2, -1], [1, -1, -1]], dtype=np.int32)
    assert np.allclose(get_number_of_neighbors(neighbor_list), np.array([2, 1]))