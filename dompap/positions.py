import numpy as np
import numba
from scipy.special import gamma


@numba.njit
def get_displacement_vector(position: np.ndarray, other_position: np.ndarray,
                            box_vectors: np.ndarray) -> np.ndarray:
    """ Calculate displacement vector considering periodic boundary conditions for any number of dimensions."""
    displacement = position - other_position
    for dim in range(len(position)):
        if displacement[dim] > box_vectors[dim] / 2:
            displacement[dim] -= box_vectors[dim]
        elif displacement[dim] < -box_vectors[dim] / 2:
            displacement[dim] += box_vectors[dim]
    return displacement


@numba.njit
def get_distance(position: np.ndarray, other_position: np.ndarray, box_vectors: np.ndarray) -> float:
    displacement: np.ndarray = get_displacement_vector(position, other_position, box_vectors)
    distances: float = np.zeros(shape=(len(position),), dtype=np.float64)
    for dim in range(len(position)):
        distances[dim] = displacement[dim] ** 2
    return np.sqrt(np.sum(distances))


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


import numpy as np


def get_radial_distribution_function(positions_n: np.ndarray,
                                     positions_m: np.ndarray,
                                     box_vectors: np.ndarray,
                                     r_bins: np.ndarray) -> [np.ndarray, np.ndarray]:
    """ Compute radial distribution function between two sets of positions."""
    if not np.all(np.diff(r_bins) >= 0):
        raise ValueError("r_bins array must be sorted in ascending order.")

    diff = positions_n[:, np.newaxis, :] - positions_m[np.newaxis, :, :]
    for i in range(diff.shape[-1]):  # iterate over dimensions
        box_length = np.linalg.norm(box_vectors[i])
        diff[:, :, i] -= box_length * np.round(diff[:, :, i] / box_length)
    distances = np.linalg.norm(diff, axis=2)  # calculate distances
    histogram = np.histogram(distances, bins=r_bins)[0]
    # Normalize histogram by number of pairs
    number_of_particles_n = positions_n.shape[0]
    number_of_particles_m = positions_m.shape[0]
    number_of_pairs = number_of_particles_n * number_of_particles_m
    box_volume = np.prod(box_vectors)
    dr_bins = np.diff(r_bins)
    r_mid = r_bins[:-1] + dr_bins / 2
    # Surface area of unit n-sphere in n dimensions: $S = 2 \pi^{n/2} / \Gamma(n/2)$
    dimensions = positions_n.shape[1]
    surface_area_n_sphere = 2 * np.pi ** (dimensions / 2) / gamma(dimensions / 2)
    shell_volume = surface_area_n_sphere * r_mid ** (dimensions - 1) * dr_bins
    radial_distribution = histogram / shell_volume / number_of_pairs * box_volume
    pair_distances = r_mid
    return pair_distances, radial_distribution


def wrap_into_box(positions: np.ndarray, image_positions: np.ndarray, box_vectors: np.ndarray):
    """ Wrap positions into box """
    shift: np.ndarray = np.floor(positions / box_vectors[np.newaxis, :], dtype=np.float64)
    image_positions += shift.astype(np.int32)
    positions -= shift * box_vectors[np.newaxis, :]
