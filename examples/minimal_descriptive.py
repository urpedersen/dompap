from dompap import Simulation

# Initialize simulation object
sim = Simulation()

# Setup simulation
sim.set_positions(unit_cell_coordinates=([0.0, 0.0, 0.0],), cells=(5, 5, 5), lattice_constants=(1.0, 1.0, 1.0))
sim.set_masses(masses=1.0)
sim.set_random_velocities(temperature=1.0)
sim.set_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
sim.set_pair_potential_parameters(sigma=1.0, epsilon=1.0)
sim.set_neighbor_list(skin=0.5, max_number_of_neighbors=128)
sim.set_integrator(time_step=0.01, target_temperature=1.0, temperature_damping_time=0.1)

# Run simulation
steps = 100
for step in range(steps):
    sim.make_step()
    if step % 10 == 0:
        print(f'Energy after {step} steps: {sim.get_total_energy()}')