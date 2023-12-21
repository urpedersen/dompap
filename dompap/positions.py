import numpy as np
import numba


@numba.njit
def get_displacement_vector(position: np.ndarray, other_position: np.ndarray, box_vectors: np.ndarray) -> np.ndarray:
    displacement = position - other_position
    # Apply periodic boundary conditions
    for dim in range(len(position)):
        if displacement[dim] > box_vectors[dim] / 2:
            displacement[dim] -= box_vectors[dim]
        elif displacement[dim] < -box_vectors[dim] / 2:
            displacement[dim] += box_vectors[dim]
    return displacement

def test_get_displacement_vector():
    position = np.array([0.0, 0.0, 0.0])
    other_position = np.array([0.5, 0.5, 0.5])
    box_vectors = np.array([2.0, 2.0, 2.0])
    displacement = get_displacement_vector(position, other_position, box_vectors)
    assert np.allclose(displacement, np.array([-0.5, -0.5, -0.5]))
    return True

@numba.njit
def get_distance(position: np.ndarray, other_position: np.ndarray, box_vectors: np.ndarray) -> float:
    displacement = get_displacement_vector(position, other_position, box_vectors)
    distances = np.zeros(shape=(len(position),), dtype=np.float64)
    for dim in range(len(position)):
        distances[dim] = displacement[dim] ** 2
    return np.sqrt(np.sum(distances))

def test_get_distance():
    position = np.array([0.0, 0.0, 0.0])
    other_position = np.array([0.5, 0.5, 0.5])
    box_vectors = np.array([2.0, 2.0, 2.0])
    distance = get_distance(position, other_position, box_vectors)
    assert distance == np.sqrt(3) / 2
    return True

def generate_positions(unit_cell_coordinates: np.ndarray,
                       cells: np.ndarray,
                       lattice_constants: np.ndarray) -> np.ndarray:
    spatial_dimension = unit_cell_coordinates.shape[1]
    number_of_particles = unit_cell_coordinates.shape[0]
    number_of_cells = np.prod(cells)
    positions = np.zeros(shape=(number_of_cells * number_of_particles, spatial_dimension), dtype=np.float64)
    for cell_index in range(number_of_cells):
        cell_coordinates = np.array(np.unravel_index(cell_index, cells), dtype=np.float64)
        for particle_index in range(number_of_particles):
            positions[cell_index * number_of_particles + particle_index] = unit_cell_coordinates[
                                                                               particle_index] + cell_coordinates
    positions *= lattice_constants
    return positions


def test_generate_positions_simple_cubic_in_space():
    unit_cell_coordinates = np.array([[0, 0, 0]], dtype=np.float64)
    cells = np.array([5, 5, 5], dtype=np.int32)
    lattice_constants = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    positions = generate_positions(unit_cell_coordinates, cells, lattice_constants)
    assert np.allclose(positions[:5],
                       np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4]], dtype=np.float64))
    assert np.allclose(positions[-5:],
                       np.array([[4, 4, 0], [4, 4, 1], [4, 4, 2], [4, 4, 3], [4, 4, 4]], dtype=np.float64))
    assert positions.shape == (125, 3)
    return True

def test_generate_positions_simple_cubic_in_plane():
    unit_cell_coordinates = np.array([[0, 0]], dtype=np.float64)
    cells = np.array([5, 5], dtype=np.int32)
    lattice_constants = np.array([1.0, 1.0], dtype=np.float64)
    positions = generate_positions(unit_cell_coordinates, cells, lattice_constants)
    assert np.allclose(positions[:5],
                       np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]], dtype=np.float64))
    assert np.allclose(positions[-5:],
                       np.array([[4, 0], [4, 1], [4, 2], [4, 3], [4, 4]], dtype=np.float64))
    assert positions.shape == (25, 2)
    return True

def test_generate_positions_body_centered_cubic():
    unit_cell_coordinates = np.array([[0, 0, 0], [0.5, 0.5, 0.5]], dtype=np.float64)
    cells = np.array([5, 5, 5], dtype=np.int32)
    lattice_constants = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    positions = generate_positions(unit_cell_coordinates, cells, lattice_constants)
    assert np.allclose(positions[:5],
                       np.array([[0, 0, 0], [0.5, 0.5, 0.5], [0, 0, 1], [0.5, 0.5, 1.5], [0, 0, 2]],
                                dtype=np.float64))
    assert np.allclose(positions[-5:],
                       np.array([[4.5, 4.5, 2.5], [4., 4., 3.], [4.5, 4.5, 3.5], [4., 4., 4.], [4.5, 4.5, 4.5]],
                                dtype=np.float64))
    assert positions.shape == (250, 3)
    return True