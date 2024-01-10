from dataclasses import dataclass, field

import numba
import numpy as np


DEFAULT_SPATIAL_DIMENSION = 3
DEFAULT_NUMBER_OF_PARTICLES = 125  # 5x5x5 simple cubic lattice


def default_particle_types():
    return np.zeros(shape=(DEFAULT_NUMBER_OF_PARTICLES, 1), dtype=np.int32)


def default_positions():
    """ Setup 5x5x5 simple cubic lattice """
    positions = np.zeros(shape=(DEFAULT_NUMBER_OF_PARTICLES, DEFAULT_SPATIAL_DIMENSION), dtype=np.float64)
    for i in range(5):
        for j in range(5):
            for k in range(5):
                positions[i * 25 + j * 5 + k] = [i, j, k]

    return positions


def default_box_vectors():
    return np.array([5, 5, 5], dtype=np.float64)


def default_image_positions():
    return np.zeros(shape=(DEFAULT_NUMBER_OF_PARTICLES, DEFAULT_SPATIAL_DIMENSION), dtype=np.int32)


def default_velocities():
    """ Random velocities from Normal distribution with variance 1 """
    return np.random.normal(loc=0.0, scale=1.0, size=(DEFAULT_NUMBER_OF_PARTICLES, DEFAULT_SPATIAL_DIMENSION))


def default_betas():
    return np.zeros(shape=(DEFAULT_NUMBER_OF_PARTICLES, DEFAULT_SPATIAL_DIMENSION), dtype=np.float64)


def default_masses():
    return np.ones(shape=(DEFAULT_NUMBER_OF_PARTICLES, 1), dtype=np.float64)


def default_func_n_m():
    return lambda n, m: np.float64(1.0)


def default_func_r():
    return lambda r: np.float64(0.0)


@dataclass
class Simulation:
    """ Simulation class. The default simulation is a 5x5x5 simple cubic lattice of particles with unit mass, diameter
    and epsilon. The box vectors are [5, 5, 5] and the particles are placed at [0, 0, 0], [0, 0, 1], ... [4, 4, 4].
    The default pair potential is the harmonic repulsive potential (1-r)**2, where r is the distance between two
    particles. The default pair potential is truncated at r_cut = 1.0. The default neighbor list is updated if a
    particle has moved more than the skin distance, which is 0.5 by default. The default maximum number of neighbors
    is 128. The ntegrator is a Langevin VT Leap-frog integrator. The default time step is 0.01,
    the default target temperature is 1.0, and the default temperature damping time is 0.1. Below is an example of how
    to set up a simulatio (using the default values):
    >>> from dompap import Simulation
    >>> sim = Simulation()
    >>> sim.set_positions(unit_cell_coordinates=([0.0, 0.0, 0.0],), cells=(5, 5, 5), lattice_constants=(1.0, 1.0, 1.0))
    >>> sim.set_masses(masses=1.0)
    >>> sim.set_random_velocities(temperature=1.0)
    >>> sim.set_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
    >>> sim.set_neighbor_list(skin=1.0, max_number_of_neighbors=128)
    >>> sim.set_integrator(time_step=0.01, target_temperature=1.0, temperature_damping_time=1.0)
    >>> print(sim)
    Simulation:
        positions: [[0. 0. 0.]] ... [[4. 4. 4.]]
        box_vectors: [5. 5. 5.]
        masses: [[1.]] ... [[1.]]
    """
    # System properties
    particle_types: np.ndarray = field(default_factory=default_particle_types)
    positions: np.ndarray = field(default_factory=default_positions)
    box_vectors: np.ndarray = field(default_factory=default_box_vectors)
    image_positions: np.ndarray = field(default_factory=default_image_positions)
    velocities: np.ndarray = field(default_factory=default_velocities)
    betas: np.ndarray = field(default_factory=default_betas)

    # Neighbor list properties
    neighbor_list: np.ndarray = None
    positions_neighbour_list: np.ndarray = field(default_factory=default_positions)

    # Particle properties
    masses: np.ndarray = field(default_factory=default_masses)

    # Neighbor list parameters
    energy_method_str = 'neighbor list'
    force_method_str = 'neighbor list'
    neighbor_list_method_str = 'cell list'
    pair_potential_r_cut: np.float64 = np.float64(1.0)
    neighbor_list_skin: np.float64 = np.float64(2.0)
    max_number_of_neighbors: np.float64 = np.int32(512)

    # Simulation parameters
    number_of_steps: int = 0
    time_step: np.float64 = np.float64(0.01)
    temperature_target: np.float64 = np.float64(1.0)
    temperature_damping_time: np.float64 = np.float64(0.1)

    # Pair potential
    pair_potential: callable = None
    pair_potential_str: str = None
    pair_force: callable = None
    epsilon_func: callable = field(default_factory=default_func_n_m)
    sigma_func: callable = field(default_factory=default_func_n_m)

    def __post_init__(self):
        """ Setup neighbor list """
        self.set_pair_potential()
        self.set_pair_potential_parameters(sigma=1.0, epsilon=1.0)
        self.set_neighbor_list()

    def __str__(self):
        """ Print only first there and last three particles """
        return f"""Simulation:
    positions: {self.positions[:1]} ... {self.positions[-1:]}
    box_vectors: {self.box_vectors}
    masses: {self.masses[:1]} ... {self.masses[-1:]}"""

    def copy(self):
        """ Make a deep copy of the simulation

        >>> from dompap import Simulation
        >>> sim = Simulation()
        >>> print(sim.positions[:3])
        [[0. 0. 0.]
         [0. 0. 1.]
         [0. 0. 2.]]
        >>> sim_copy = sim.copy()
        >>> sim_copy.positions[0] = [1.0, 0.0, 0.0]
        >>> print(sim.positions[:3])
        [[0. 0. 0.]
         [0. 0. 1.]
         [0. 0. 2.]]
        >>> print(sim_copy.positions[:3])
        [[1. 0. 0.]
         [0. 0. 1.]
         [0. 0. 2.]]
        """
        from copy import deepcopy
        return deepcopy(self)

    def set_positions(self, unit_cell_coordinates: tuple = ([0.0, 0.0, 0.0],), cells: tuple = (5, 5, 5),
                      lattice_constants: tuple = (1.0, 1.0, 1.0)):
        """ Set positions of particles
        >>> from dompap import Simulation
        >>> sim = Simulation()
        >>> sim.set_positions(unit_cell_coordinates=([0, 0, 0],), cells=(5, 5, 5))
        >>> print(sim.positions[:5])
        [[0. 0. 0.]
         [0. 0. 1.]
         [0. 0. 2.]
         [0. 0. 3.]
         [0. 0. 4.]]
        """
        from .positions import generate_positions
        unit_cell_coordinates = np.array(unit_cell_coordinates, dtype=np.float64)
        cells = np.array(cells, dtype=np.int32)
        lattice_constants = np.array(lattice_constants, dtype=np.float64)
        self.box_vectors = np.array(cells, dtype=np.float64) * np.array(lattice_constants, dtype=np.float64)
        self.positions = generate_positions(unit_cell_coordinates, cells, lattice_constants)
        self.image_positions = np.zeros(shape=(self.positions.shape[0], self.positions.shape[1]), dtype=np.int32)
        # Set betas and velocities if shape is not correct
        if self.betas.shape != self.positions.shape:
            self.betas = np.zeros(shape=self.positions.shape, dtype=np.float64)
        if self.velocities.shape != self.positions.shape:
            self.velocities = np.random.normal(loc=0.0, scale=1.0, size=self.positions.shape).astype(np.float64)

    def set_masses(self, masses: float = 1.0):
        """ Set masses of particles
        >>> from dompap import Simulation
        >>> sim = Simulation()
        >>> sim.set_masses(masses=1.0)
        >>> print(sim.masses[:3])
        [[1.]
         [1.]
         [1.]]
        """
        self.masses = np.ones(shape=(self.positions.shape[0], 1), dtype=np.float64) * np.float64(masses)

    def set_random_velocities(self, temperature: float = 1.0):
        """ Set velocities from Normal distribution with variance temperature / mass
        >>> from dompap import Simulation
        >>> sim = Simulation()
        >>> sim.set_random_velocities(temperature=1.0)
        """
        self.velocities = np.random.normal(loc=0.0, scale=np.sqrt(temperature / self.masses),
                                           size=self.positions.shape).astype(np.float64)

    _KNOWN_CELL_LIST_METHODS = {'cell list', 'double loop'}
    def set_neighbor_list(self, skin: float = None, max_number_of_neighbors: int = None, method_str=None):
        """ Update neighbour list
        >>> from dompap import Simulation
        >>> from dompap.neighbor_list import get_number_of_neighbors
        >>> sim = Simulation()
        >>> sim.set_neighbor_list()
        >>> max_number_of_neighbours = np.max(get_number_of_neighbors(sim.neighbor_list))
        >>> print(f'Max number of neighbours: {max_number_of_neighbours}')
        Max number of neighbours: 92
        """
        if skin is not None:
            self.neighbor_list_skin = np.float64(skin)
        if max_number_of_neighbors is not None:
            self.max_number_of_neighbors = np.int32(max_number_of_neighbors)
        if method_str is not None:
            if method_str not in self._KNOWN_CELL_LIST_METHODS:
                raise ValueError(f'Unknown neighbor list method: {method_str}. Try: {self._KNOWN_CELL_LIST_METHODS}.')
            self.neighbor_list_method_str = method_str
        # Get largest possible sigma
        number_of_particles = self.positions.shape[0]
        largest_sigma = 0.0
        for n in range(number_of_particles):
            for m in range(number_of_particles):
                largest_sigma = max(largest_sigma, self.sigma_func(n, m))

        global_truncation_distance = largest_sigma * self.pair_potential_r_cut
        positions = self.positions
        box_vectors = self.box_vectors
        cutoff_distance = global_truncation_distance + self.neighbor_list_skin
        max_number_of_neighbours = self.max_number_of_neighbors
        if self.neighbor_list_method_str == 'cell list':
            from .neighbor_list import get_neighbor_list_cell_list
            self.neighbor_list = get_neighbor_list_cell_list(positions, box_vectors, cutoff_distance,
                                                             max_number_of_neighbours)
        elif self.neighbor_list_method_str == 'double loop':
            from .neighbor_list import get_neighbor_list_double_loop
            self.neighbor_list = get_neighbor_list_double_loop(positions, box_vectors, cutoff_distance,
                                                               max_number_of_neighbours)
        else:
            raise ValueError(f'Unknown neighbor list method: {self.neighbor_list_method_str}')
        self.positions_neighbour_list = self.positions.copy()


    def update_neighbor_list(self, check=True):
        """ Update neighbour list if needed """
        from .neighbor_list import neighbor_list_is_old
        if check and not neighbor_list_is_old(self.positions, self.positions_neighbour_list, self.box_vectors,
                                              self.neighbor_list_skin):
            return
        self.set_neighbor_list()

    def set_pair_potential(self, pair_potential_str: str = '(1-r)**2', r_cut: float = 1.0):
        """ Set pair potential.py and force
        >>> from dompap import Simulation
        >>> sim = Simulation()
        >>> sim.set_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
        >>> sim.set_pair_potential_parameters(sigma=2.0, epsilon=4.0)
        >>> print(sim.pair_potential(0.5))
        0.25
        """
        from .potential import hardcoded_pair_potentials
        if pair_potential_str in hardcoded_pair_potentials:
            self.pair_potential_str = hardcoded_pair_potentials[pair_potential_str][0]
            self.pair_potential = hardcoded_pair_potentials[pair_potential_str][1]
            self.pair_force = hardcoded_pair_potentials[pair_potential_str][2]
            self.pair_potential_r_cut = hardcoded_pair_potentials[pair_potential_str][3]
            self.set_neighbor_list()
            return

        # If not hardcoded, make pair potential using SymPy
        from .potential import make_pair_potential
        self.pair_potential_str = pair_potential_str
        self.pair_potential_r_cut = np.float64(r_cut)
        self.pair_potential, self.pair_force = make_pair_potential(
            pair_potential_str=pair_potential_str,
            r_cut=r_cut
        )
        self.set_neighbor_list()

    def set_pair_potential_parameters(self, sigma: float = 1.0, epsilon: float = 1.0):
        """ Set potential parameters
        >>> from dompap import Simulation
        >>> sim = Simulation()
        >>> sim.set_pair_potential_parameters(sigma=1.0, epsilon=1.0)
        """
        self.sigma_func = numba.njit(lambda n, m: np.float64(sigma))
        self.epsilon_func = numba.njit(lambda n, m: np.float64(epsilon))
        self.set_neighbor_list()

    def scale_box(self, scale_factor: float):
        """ Scale box vectors and positions
        >>> from dompap import Simulation
        >>> sim = Simulation()
        >>> print(sim.box_vectors)
        [5. 5. 5.]
        >>> sim.scale_box(0.5)
        >>> print(sim.box_vectors)
        [2.5 2.5 2.5]
        """
        self.box_vectors = self.box_vectors * scale_factor
        self.positions = self.positions * scale_factor
        self.set_neighbor_list()

    def get_potential_energy(self):
        """ Get total energy of the system
        >>> from dompap import Simulation
        >>> sim = Simulation()
        >>> sim.scale_box(0.5)
        >>> print(sim.get_potential_energy())
        167.06442443573454
        """

        if self.energy_method_str == 'neighbor list':
            from .potential import _get_total_energy
            self.update_neighbor_list()
            energy = _get_total_energy(self.positions, self.box_vectors, self.pair_potential, self.neighbor_list,
                                       self.sigma_func, self.epsilon_func)
            return float(energy)
        elif self.energy_method_str == 'double loop':
            from .potential import _get_total_energy_double_loop
            energy = _get_total_energy_double_loop(self.positions, self.box_vectors, self.pair_potential,
                                                   self.sigma_func, self.epsilon_func)
            return float(energy)

    def get_forces(self) -> np.ndarray:
        """ Get forces on all particles
        >>> from dompap import Simulation
        >>> sim = Simulation()
        >>> sim.positions[0] = [0.5, 0.0, 0.0]  # Shift particle 0 so that the force is not zero
        >>> forces = sim.get_forces()
        >>> print(f'Force on particle 0: F_x={forces[0, 0]}, F_y={forces[0, 1]}, F_z={forces[0, 2]}')
        Force on particle 0: F_x=-1.0, F_y=0.0, F_z=0.0
        """
        if self.force_method_str == 'neighbor list':
            from .potential import _get_forces
            self.update_neighbor_list()
            forces = _get_forces(self.positions, self.box_vectors, self.pair_force, self.neighbor_list,
                             self.sigma_func, self.epsilon_func)
            return forces
        elif self.force_method_str == 'double loop':
            from .potential import _get_forces_double_loop
            forces = _get_forces_double_loop(self.positions, self.box_vectors, self.pair_force,
                                             self.sigma_func, self.epsilon_func)
            return forces
        elif self.force_method_str == 'double loop single core':
            from .potential import _get_forces_double_loop_single_core
            forces = _get_forces_double_loop_single_core(self.positions, self.box_vectors, self.pair_force,
                                                         self.sigma_func, self.epsilon_func)
            return forces
        elif self.force_method_str == 'vectorized':
            from .potential import _get_forces_vectorized
            forces = _get_forces_vectorized(self.positions, self.box_vectors, self.pair_force,
                                            self.sigma_func, self.epsilon_func)
            return forces
        else:
            raise ValueError(f'Unknown force method: {self.force_method_str}')

    def wrap_into_box(self):
        """ Wrap all particles into box
        >>> from dompap import Simulation
        >>> sim = Simulation()
        >>> print(sim.box_vectors)
        [5. 5. 5.]
        >>> print(sim.image_positions[0])
        [0 0 0]
        >>> sim.positions[0] = [6.0, 0.0, 0.0]  # Shift particle 0 outside the box
        >>> print(sim.positions[0])
        [6. 0. 0.]
        >>> sim.wrap_into_box()
        >>> print(sim.positions[0])
        [1. 0. 0.]
        >>> print(sim.image_positions[0])
        [1 0 0]
        """
        from .positions import wrap_into_box
        wrap_into_box(self.positions, self.image_positions, self.box_vectors)

    def step(self):
        """ Make one step in the simulation
        >>> from dompap import Simulation
        >>> sim = Simulation()
        >>> sim.step()
        """
        from .integrator import make_one_step
        self.update_neighbor_list()
        forces = self.get_forces()
        old_state = self.positions, self.velocities, self.betas
        parameters = self.time_step, self.temperature_target, self.temperature_damping_time
        new_state = make_one_step(*old_state, forces, self.masses, *parameters)
        self.positions, self.velocities, self.betas = new_state
        self.wrap_into_box()
        self.number_of_steps += 1

    def run(self, steps: int = 1000):
        """ Run simulation
        >>> from dompap import Simulation
        >>> sim = Simulation()
        >>> sim.run(steps=1000)
        """
        for i in range(steps):
            self.step()

    def set_integrator(self,
                       time_step: float = 0.01,
                       target_temperature: float = 1.0,
                       temperature_damping_time: float = 0.1):
        """ Set integrator parameters
        >>> from dompap import Simulation
        >>> sim = Simulation()
        >>> sim.set_integrator(time_step=0.01, target_temperature=1.0, temperature_damping_time=0.1)
        """
        self.time_step = np.float64(time_step)
        self.temperature_target = np.float64(target_temperature)
        self.temperature_damping_time = np.float64(temperature_damping_time)

    def number_of_particles(self) -> int:
        """ Get number of particles
        >>> from dompap import Simulation
        >>> sim = Simulation()
        >>> print(sim.number_of_particles())
        125
        """
        return self.positions.shape[0]

    def get_density(self) -> float:
        """ Get density of the system
        >>> from dompap import Simulation
        >>> sim = Simulation()
        >>> print(f'Density: {sim.get_density():.3f}')
        Density: 1.000
        """
        return self.number_of_particles() / np.prod(self.box_vectors)

    def set_density(self, density: float = 1.0):
        """ Set density of the system
        >>> from dompap import Simulation
        >>> sim = Simulation()
        >>> sim.set_density(density=0.9)
        >>> print(f'Density: {sim.get_density():.3f}')
        Density: 0.900
        """
        dimensions = self.box_vectors.shape[0]
        current_density = self.get_density()
        scale_factor = (current_density / density) ** (1 / dimensions)
        self.scale_box(scale_factor)

    def get_dimensions_of_space(self) -> int:
        """ Get dimensions of space
        >>> from dompap import Simulation
        >>> sim = Simulation()
        >>> print(sim.get_dimensions_of_space())
        3
        """
        return self.box_vectors.shape[0]

    def get_temperature(self):
        """ Get temperature of the system
        >>> from dompap import Simulation
        >>> sim = Simulation()
        >>> print(f'Temperature: {sim.get_temperature():.0f}')
        Temperature: 1
        """
        m = self.masses[:, 0]
        v = self.velocities
        dimensions_of_space = self.get_dimensions_of_space()
        number_of_particles = self.number_of_particles()
        v_squared = np.sum(v ** 2, axis=1)
        temperature = np.sum(m * v_squared) / (dimensions_of_space * number_of_particles)
        return temperature

    def get_radial_distribution_function(self, r_bins: np.ndarray) -> [np.ndarray, np.ndarray]:
        """ Get radial distribution function
        >>> from dompap import Simulation
        >>> sim = Simulation()
        >>> r_bins = np.linspace(0.1, 3.0, 100)
        >>> g_r = sim.get_radial_distribution_function(r_bins=r_bins)
        """
        from .positions import get_radial_distribution_function
        r, g_r = get_radial_distribution_function(self.positions, self.positions, self.box_vectors, r_bins)
        return r, g_r

    def get_number_of_particles(self):
        return self.positions.shape[0]

    def get_kinetic_energy(self):
        """ Get kinetic energy of the system
        >>> from dompap import Simulation
        >>> sim = Simulation()
        >>> sim.velocities = np.ones_like(sim.velocities)  # Set all velocities to 1
        >>> print(f'Kinetic energy: {sim.get_kinetic_energy():.1f}')
        Kinetic energy: 187.5
        """
        v = self.velocities
        m = self.masses
        return 0.5 * np.sum(m*v*v)

    def get_diameters(self) -> np.ndarray:
        """ Get diameters of particles
        >>> from dompap import Simulation
        >>> sim = Simulation()
        >>> print(sim.get_diameters()[:3])
        [[1.]
         [1.]
         [1.]]
        """
        diameters = np.ones(shape=(self.number_of_particles(), 1), dtype=np.float64)
        for n in range(self.number_of_particles()):
            diameters[n] = self.sigma_func(n, n)
        return diameters

    def get_time(self):
        """ Get time of the simulation
        >>> from dompap import Simulation
        >>> sim = Simulation()
        >>> sim.set_integrator(time_step=0.01, target_temperature=1.0, temperature_damping_time=0.1)
        >>> for _ in range(10):
        ...     sim.step()
        >>> print(sim.get_time())
        0.1
        """
        return self.number_of_steps * self.time_step

    def get_virial(self) -> float:
        from .potential import _get_virial_double_loop
        self.update_neighbor_list()
        virial = _get_virial_double_loop(self.positions, self.box_vectors, self.pair_force,
                                         self.sigma_func, self.epsilon_func)
        return virial

    def get_pressure(self) -> float:
        """ Get pressure of the system
        >>> from dompap import Simulation
        >>> sim = Simulation()
        >>> sim.set_random_velocities(temperature=0.0)
        >>> print(sim.get_pressure())
        0.0
        """
        V = self.get_volume()
        W = self.get_virial()
        P_c = W / V
        D = self.get_dimensions_of_space()
        v = self.velocities
        m = self.masses
        P_id = np.sum(v*v*m) / D / V
        return P_c + P_id

    def get_volume(self):
        """ Get volume of the system
        >>> from dompap import Simulation
        >>> sim = Simulation()
        >>> print(sim.get_volume())
        125.0
        """
        return np.prod(self.box_vectors)

    def set_particle_types(self, types):
        """ Set particle types
        >>> from dompap import Simulation
        >>> sim = Simulation()
        >>> sim.set_particle_types(types=1)
        >>> print(sim.particle_types[:3])
        [[1]
         [1]
         [1]]
        """
        self.particle_types = np.ones_like(self.masses, dtype=np.int32) * types



def test_simulation():
    sim = Simulation()
    steps = 100
    for i in range(steps):
        sim.step()
    assert sim.get_potential_energy() > 0.0


if __name__ == '__main__':
    test_simulation()
