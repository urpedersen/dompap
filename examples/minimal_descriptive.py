""" Minimal descriptive example """

from dompap import Simulation

# Initialize simulation object
sim = Simulation()

# Setup simulation
fcc_unit_cell = ([0.0, 0.0, 0.0],
                 [0.5, 0.5, 0.0],
                 [0.5, 0.0, 0.5],
                 [0.0, 0.5, 0.5])
sim.set_positions(unit_cell_coordinates=fcc_unit_cell,
                  cells=(5, 5, 5),
                  lattice_constants=(1.0, 1.0, 1.0))
sim.set_density(density=1.0)
sim.set_masses(masses=1.0)
sim.set_random_velocities(temperature=1.0)
sim.set_pair_potential(pair_potential_str='(1-r)**2',
                       r_cut=1.0,
                       force_method='neighbor list',
                       energy_method='neighbor list')
sim.set_pair_potential_parameters(sigma=1.0, epsilon=1.0)
sim.set_neighbor_list(skin=0.7,
                      max_number_of_neighbors=128,
                      method_str='double loop')
sim.set_integrator(time_step=0.01,
                   target_temperature=1.0,
                   temperature_damping_time=0.1)

# Run simulation
steps = 100
for step in range(steps):
    sim.step()
    if step % 10 == 0:
        print(f'Energy after {step} steps: {sim.get_potential_energy()}')