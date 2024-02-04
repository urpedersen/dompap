import numpy as np
import numba

from .positions import get_distance, get_displacement_vector


@numba.njit(parallel=True)
def get_neighbor_list_double_loop(positions, box_vectors, cutoff_distance, max_number_of_neighbors) -> np.ndarray:
    """ Get neighbour list using a N^2 loop (backend) """
    number_of_particles: int = positions.shape[0]

    neighbor_list = np.zeros(shape=(number_of_particles, max_number_of_neighbors), dtype=np.int32) - 1  # -1 is empty
    for n in numba.prange(number_of_particles):
        position = positions[n]
        current_idx = 0
        for m in range(number_of_particles):
            if n == m:
                continue
            other_position = positions[m]
            displacement = position - other_position
            # Periodic boundary conditions
            for d in range(len(displacement)):
                if displacement[d] < -box_vectors[d] / 2:
                    displacement[d] += box_vectors[d]
                elif displacement[d] > box_vectors[d] / 2:
                    displacement[d] -= box_vectors[d]
            distance = np.sqrt(np.sum(displacement ** 2))
            if distance < cutoff_distance:
                neighbor_list[n][current_idx] = m
                current_idx += 1
                # if current_idx >= max_number_of_neighbors:  May be expensive, so we don't check, user get malloc error
                #    raise ValueError(f'Number of neighbors for particle {n} '
                #                     f'exceeds max_number_of_neighbors={max_number_of_neighbors}.')
    return neighbor_list


def get_neighbor_list_cell_list(positions, box_vectors, cutoff_distance, max_number_of_neighbors) -> np.ndarray:
    """ Get neighbour list using a cell list (backend). This works for any number of spatial dimensions, but is slower,
    since it cannot be numba.jit'ed. """
    number_of_spatial_dimensions = positions.shape[1]
    max_number_of_cell_neighbors = max_number_of_neighbors * 3 ** number_of_spatial_dimensions
    # Find the number of cells in each direction
    number_of_cells = np.ceil(box_vectors / cutoff_distance).astype(np.int32)
    # Create a list of empty cells
    cells = np.zeros(shape=(*tuple(number_of_cells), max_number_of_cell_neighbors), dtype=np.int32) - 1  # -1 is empty

    # Loop particles and add them to the cells
    number_of_particles = positions.shape[0]
    for n in range(number_of_particles):
        position = positions[n]
        cell_idx = np.floor(position / cutoff_distance).astype(np.int32)
        this_cell = cells[tuple(cell_idx)]
        idx = np.sum(this_cell != -1)  # Index of the cell in the cells array
        this_cell[idx] = n

    # Loop particles and find neighbors
    neighbor_list = np.zeros(shape=(number_of_particles, max_number_of_neighbors), dtype=np.int32) - 1  # -1 is empty
    for n in range(number_of_particles):
        next_available_idx = 0
        position = positions[n]
        cell_idx = np.floor(position / cutoff_distance).astype(np.int32)
        number_of_spatial_dimensions = len(cell_idx)
        # Loop over all cells in the neighborhood
        for i in range(3 ** number_of_spatial_dimensions):
            # Get the cell index of the neighbor cell
            neighbor_cell_idx = cell_idx + np.unravel_index(i, (3,) * number_of_spatial_dimensions) - 1
            # Apply per½iodic boundary conditions to cell idx
            for d in range(number_of_spatial_dimensions):
                if neighbor_cell_idx[d] < 0:
                    neighbor_cell_idx[d] += number_of_cells[d]
                elif neighbor_cell_idx[d] >= number_of_cells[d]:
                    neighbor_cell_idx[d] -= number_of_cells[d]
            # Loop particles in cell and add them to neighbor list
            neighbor_cell = cells[tuple(neighbor_cell_idx)]
            for m in range(max_number_of_cell_neighbors):
                if neighbor_cell[m] == -1:
                    break
                other_position = positions[neighbor_cell[m]]
                distance = get_distance(position, other_position, box_vectors)
                if distance < cutoff_distance and neighbor_cell[m] != n:
                    neighbor_list[n][next_available_idx] = neighbor_cell[m]
                    next_available_idx += 1
    return neighbor_list


@numba.njit
def get_neighbor_list_cell_list_3d(positions, box_vectors, cutoff_distance, max_number_of_neighbors) -> np.ndarray:
    number_of_spatial_dimensions = positions.shape[1]
    if number_of_spatial_dimensions != 3:
        raise NotImplementedError(f'get_neighbor_list_cell_list_3d is only for 3 spatial dimensions.')
    max_number_of_cell_neighbors = np.int32(max_number_of_neighbors * 3 ** number_of_spatial_dimensions)
    # Find the number of cells in each direction
    number_of_cells = np.ceil(box_vectors / cutoff_distance).astype(np.int32)
    # Create a list of empty cells
    cells_shape = (number_of_cells[0], number_of_cells[1], number_of_cells[2], max_number_of_cell_neighbors)
    cells = np.zeros(shape=cells_shape, dtype=np.int32) - 1  # -1 is empty

    # Loop particles and add them to the cells
    number_of_particles = positions.shape[0]
    for n in range(number_of_particles):
        position = positions[n]
        cell_idx = np.floor(position / cutoff_distance).astype(np.int32)
        cell_idx_tuple = cell_idx[0], cell_idx[1], cell_idx[2]
        this_cell = cells[cell_idx_tuple]
        idx = np.sum(this_cell != -1)  # Index of the cell in the cells array
        this_cell[idx] = n

    # Loop particles and find neighbors
    neighbor_list = np.zeros(shape=(number_of_particles, max_number_of_neighbors), dtype=np.int32) - 1  # -1 is empty
    for n in range(number_of_particles):
        next_available_idx = 0
        position = positions[n]
        cell_idx = np.floor(position / cutoff_distance).astype(np.int32)
        number_of_spatial_dimensions = len(cell_idx)
        # Loop over all cells in the neighborhood
        for i in range(3 ** number_of_spatial_dimensions):
            # Get the cell index of the neighbor cell
            lst = []
            tmp_i = i
            for d in range(number_of_spatial_dimensions):
                lst.append(tmp_i % 3 - 1)
                tmp_i //= 3
            cell_shift = np.array(lst, dtype=np.int32)
            neighbor_cell_idx = cell_idx + cell_shift
            # Apply periodic boundary conditions to cell idx
            for d in range(number_of_spatial_dimensions):
                if neighbor_cell_idx[d] < 0:
                    neighbor_cell_idx[d] += number_of_cells[d]
                elif neighbor_cell_idx[d] >= number_of_cells[d]:
                    neighbor_cell_idx[d] -= number_of_cells[d]
            # Loop particles in cell and add them to neighbor list
            neighbor_cell_idx_tuple = neighbor_cell_idx[0], neighbor_cell_idx[1], neighbor_cell_idx[2]
            neighbor_cell = cells[neighbor_cell_idx_tuple]
            for m in range(max_number_of_cell_neighbors):
                if neighbor_cell[m] == -1:
                    break
                other_position = positions[neighbor_cell[m]]
                distance = get_distance(position, other_position, box_vectors)
                if distance < cutoff_distance and neighbor_cell[m] != n:
                    neighbor_list[n][next_available_idx] = neighbor_cell[m]
                    next_available_idx += 1
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


@numba.njit
def get_number_of_neighbors(neighbor_list):
    """ Get number of neighbours for each particle """
    return np.sum(neighbor_list != -1, axis=1)
