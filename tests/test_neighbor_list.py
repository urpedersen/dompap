import numpy as np

from dompap.neighbor_list import (
    get_neighbor_list_double_loop,
    get_neighbor_list_cell_list_3d,
    get_neighbor_list_cell_list,
    neighbor_list_is_old,
    get_number_of_neighbors
)


def test_get_neighbor_list_3d():
    # Test 3D example
    positions = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [2, 2, 2],
        [5, 5, 0]], dtype=np.float64)
    box_vectors = np.array([6, 6, 6], dtype=np.float64)
    cutoff_distance = np.float64(1.5)
    max_number_of_neighbors = 4
    neighbor_list = get_neighbor_list_double_loop(positions, box_vectors, cutoff_distance, max_number_of_neighbors)
    assert np.all(neighbor_list == np.array([
        [1, 2, 3, 5],
        [0, 2, 3, -1],
        [0, 1, 3, -1],
        [0, 1, 2, -1],
        [-1, -1, -1, -1],
        [0, -1, -1, -1]], dtype=np.int32))


def test_get_neighbor_list_2d():
    # Test 2D example
    positions = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [2, 2],
        [5, 5]], dtype=np.float64)
    box_vectors = np.array([6, 6], dtype=np.float64)
    cutoff_distance = np.float64(1.5)
    max_number_of_neighbors = 4
    neighbor_list = get_neighbor_list_double_loop(positions, box_vectors, cutoff_distance, max_number_of_neighbors)
    assert np.all(neighbor_list == np.array([
        [1, 2, 4, -1],
        [0, 2, -1, -1],
        [0, 1, -1, -1],
        [-1, -1, -1, -1],
        [0, -1, -1, -1]], dtype=np.int32))


def test_get_neighbor_list_cell_list_3d():
    positions = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [2, 2, 2],
        [5, 5, 0]], dtype=np.float64)
    box_vectors = np.array([6, 6, 6], dtype=np.float64)
    cutoff_distance = np.float64(1.5)
    max_number_of_neighbors = 4
    neighbor_list = get_neighbor_list_cell_list_3d(positions, box_vectors, cutoff_distance, max_number_of_neighbors)
    assert np.all(neighbor_list == np.array([
        [5, 1, 2, 3],
        [0, 2, 3, -1],
        [0, 1, 3, -1],
        [0, 1, 2, -1],
        [-1, -1, -1, -1],
        [0, -1, -1, -1]], dtype=np.int32))


def test_get_neighbor_list_cell_list_2d():
    positions = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [2, 2],
        [5, 5]], dtype=np.float64)
    box_vectors = np.array([6, 6], dtype=np.float64)
    cutoff_distance = np.float64(1.5)
    max_number_of_neighbors = 4
    neighbor_list = get_neighbor_list_cell_list(positions, box_vectors, cutoff_distance, max_number_of_neighbors)
    assert np.all(neighbor_list == np.array([
        [4, 1, 2, -1],
        [0, 2, -1, -1],
        [0, 1, -1, -1],
        [-1, -1, -1, -1],
        [0, -1, -1, -1]], dtype=np.int32))


def test_neighbor_list_is_old():
    positions = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float64)
    old_positions = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float64)
    box_vectors = np.array([5, 5, 5], dtype=np.float64)
    skin = np.float64(0.5)
    assert neighbor_list_is_old(positions, old_positions, box_vectors, skin) == False
    old_positions[0] = [skin / 2 - 0.01, 0.0, 0.0]  # Particle 0 has moved less 0.01 larger half the skin
    assert neighbor_list_is_old(positions, old_positions, box_vectors, skin) == False
    old_positions[0] = [skin / 2 + 0.01, 0.0, 0.0]  # Particle 0 has moved 0.01 larger half the skin
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


def test_get_number_of_neighbors_all_minus_one():
    neighbor_list = np.array([[-1, -1, -1], [-1, -1, -1]], dtype=np.int32)
    assert np.all(get_number_of_neighbors(neighbor_list) == 0)  # No particles have neighbors


def test_get_number_of_neighbors():
    neighbor_list = np.array([[1, 2, -1], [1, -1, -1]], dtype=np.int32)
    assert np.allclose(get_number_of_neighbors(neighbor_list), np.array([2, 1]))
