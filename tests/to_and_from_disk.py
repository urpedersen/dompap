# Test that a simulation can be correctly saved to and loaded from disk.
import os
import tempfile
from pprint import pprint

from dompap import Simulation


def test_to_and_from_disk(verbose=False):
    sim = Simulation()

    # Setup non-default simulation
    sim.set_positions(unit_cell_coordinates=((0, 0, 0), (0.5, 0.5, 0.5)),
                      cells=(1, 2, 2),
                      lattice_constants=(4.2, 2.0, 2.4))
    sim.set_masses(masses=1.234)
    sim.set_random_velocities(temperature=1.0)
    sim.set_pair_potential(pair_potential_str='(1.2-r)**4', r_cut=1.2)
    sim.set_pair_potential_parameters(sigma=1.1, epsilon=0.24)
    sim.set_neighbor_list(skin=0.54321, max_number_of_neighbors=192)
    sim.set_integrator(time_step=0.1234, target_temperature=1.123, temperature_damping_time=0.123)
    sim.particle_types[0] = 1
    sim.particle_types[1] = 2
    sim.image_positions[0] = 1, 2, 3
    sim.image_positions[1] = 4, 5, 6
    sim.betas[0] = 1.2, 3.4, 5.6
    sim.betas[1] = 7.8, 9.1, 2.3

    # Get a temp dir on disk
    temp_dir = tempfile.mkdtemp()
    if verbose:
        print('\n    Temp dir:', temp_dir)
    particle_data_file = os.path.join(temp_dir, 'simulation.csv')
    meta_data_file = os.path.join(temp_dir, 'simulation.toml')

    # Save simulation data to disk
    sim.to_disk(particle_data=particle_data_file, meta_data=meta_data_file)

    new = Simulation()
    new.from_disk(particle_data=particle_data_file, meta_data=meta_data_file)

    # Check that the new simulation is the same as the old one
    keys_to_ignore = {'pair_potential', 'pair_force', 'epsilon_func', 'sigma_func'}
    old_data = {k: v for k, v in sim.__dict__.items() if k not in keys_to_ignore}
    new_data = {k: v for k, v in new.__dict__.items() if k not in keys_to_ignore}
    assert old_data != new_data

    if verbose:
        print('\n    Data in new simulation (checked that it is the same as in old):')
        pprint(new_data)

    # Delete the files simulation.csv and simulation.toml from disk
    os.remove(particle_data_file)
    os.remove(meta_data_file)

    # Delete the temp dir from disk
    os.rmdir(temp_dir)

    if verbose:
        print('\n    Deleted files and temp dir from disk.')

if __name__ == '__main__':
    test_to_and_from_disk(verbose=True)
