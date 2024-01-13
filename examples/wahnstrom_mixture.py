""" Simulation of the Wahnström mixture
This is an example of how to use the package to simulate a mixture of particles with different sizes.

In the Wahnstrom mixture (https://doi.org/10.1103/PhysRevA.44.3752), there are two types of particles, A's and B's interacting with the Lennard-Jones potential.
The composition of the mixture is 50% A's and 50% B's. The model parameters are
$\sigma_{AA} = 1.0$, $\sigma_{BB} = 1.2$, $\sigma_{AB} = 1.1$,
$\epsilon_{AA} = \epsilon_{BB} = \epsilon_{AB} = 1.0$,
$m_A = 1.0$ and $m_B = 2.0$.

"""

from dompap import Simulation, autotune, to_lammps_dump

make_lammps_dump = False  # Set to True to output a LAMMPS dump file

# Set up the simulation
sim = Simulation()

CsCl_lattice = ([0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5])
sim.set_positions(unit_cell_coordinates=CsCl_lattice,
                  cells=(5, 5, 5),
                  lattice_constants=(1.0, 1.0, 1.0))
sim.set_density(density=1.0)
sim.set_pair_potential(pair_potential_str='(1-r)**2',
                       r_cut=1.0,
                       force_method='neighbor list',
                       energy_method='neighbor list')
sim.set_pair_potential_parameters(sigma=1.0, epsilon=1.0)
sim.set_neighbor_list(skin=0.6,
                      max_number_of_neighbors=128,
                      method_str='double loop')
sim.set_integrator(time_step=0.01,
                   target_temperature=1.0,
                   temperature_damping_time=0.1)

# Set up the mixture
number_of_particles = sim.get_number_of_particles()
number_of_As = number_of_particles // 2
number_of_Bs = number_of_particles // 2
if number_of_particles != number_of_As + number_of_Bs:
    raise ValueError("The sum of the A's and B's must be equal to the total number of particles.")

# Set the masses
sim.set_types(types=[0, 1] * number_of_As)
sim.set_masses(masses=[1.0, 2.0] * number_of_As)  # Set the masses
sim.set_random_velocities(temperature=1.0)


# Set the pair potential parameters
def sigma_func(n, m):
    """ Return the sigma parameter for the pair potential between particles with index n and m """
    if n == m:
        if n % 2 == 0:
            return 1.0
        else:
            return 1.2
    else:
        return 1.1


sim.set_pair_potential_parameters(sigma=sigma_func, epsilon=1.0)

# Equilibrate and autotune
sim.run(200)
sim = autotune(sim, steps=100, verbose=True, plot=False, smallest_skin=0.2, step_skin=0.05)

# Production run
steps = 1_000
stride = 50
for step in range(steps):
    if step % stride == 0:
        print(f'Energy after {step} steps: {sim.get_potential_energy()}')
        if make_lammps_dump:
            print(to_lammps_dump(sim), file=open(f'dump.lammps', 'a'))
    sim.step()

