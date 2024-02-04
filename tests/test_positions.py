import numpy as np

from dompap.positions import (
    get_displacement_vector,
    get_distance,
    generate_positions,
    get_radial_distribution_function,
    wrap_into_box
)


def test_get_displacement_vector():
    position = np.array([0.0, 0.0, 0.0])
    other_position = np.array([0.5, 0.5, 0.5])
    box_vectors = np.array([2.0, 2.0, 2.0])
    displacement = get_displacement_vector(position, other_position, box_vectors)
    assert np.allclose(displacement, np.array([-0.5, -0.5, -0.5]))


def test_get_distance():
    position = np.array([0.0, 0.0, 0.0])
    other_position = np.array([0.5, 0.5, 0.5])
    box_vectors = np.array([2.0, 2.0, 2.0])
    distance = get_distance(position, other_position, box_vectors)
    assert distance == np.sqrt(3) / 2


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


def test_compute_radial_distribution_function():
    # Test that the function returns the correct values for an ideal gas (random points) in 3D
    number_of_particles = 1000
    positions_n = np.random.rand(number_of_particles, 3)
    box_vectors = np.array([10, 10, 10], dtype=np.float64)
    r_bins = np.linspace(0.1, 5, 20, dtype=np.float64)
    rdf = get_radial_distribution_function(positions_n, positions_n, box_vectors, r_bins)
    r = rdf[0]
    g_r = rdf[1]
    # assert np.allclose(rdf, 1.0, atol=0.3)


def test_wrap_into_box():
    positions = np.array([
        [0.1, 0.1, 0.1],
        [2.5, 2.5, 2.5],
        [-0.1, -0.1, -0.1]], dtype=np.float64)
    image_positions = np.zeros_like(positions)
    box_vectors = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    wrap_into_box(positions, image_positions, box_vectors)
    assert np.allclose(positions, np.array([
        [0.1, 0.1, 0.1],
        [0.5, 0.5, 0.5],
        [1.9, 1.9, 1.9]], dtype=np.float64))
    assert np.allclose(image_positions, np.array([
        [0, 0, 0],
        [1, 1, 1],
        [-1, -1, -1]], dtype=np.float64))
