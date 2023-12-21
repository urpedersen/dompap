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
    return max_distance_squared > skin ** 2


def test_neighbor_list_is_old():
    positions = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float64)
    old_positions = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float64)
    box_vectors = np.array([1, 1, 1], dtype=np.float64)
    skin = np.float64(0.5)
    assert neighbor_list_is_old(positions, old_positions, box_vectors, skin) == False
    old_positions[0] = [0.5, 0.5, 0.5]
    assert neighbor_list_is_old(positions, old_positions, box_vectors, skin) == True
    print("Test passed: neighbor_list_is_old() is as expected.")


def get_number_of_neighbors(neighbor_list):
    """ Get number of neighbours for each particle """
    return np.sum(neighbor_list != -1, axis=1)
