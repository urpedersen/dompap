from dompap import Simulation
from dompap.tools import autotune


def test_autotune(verbose=False, plot=False, test_double_loop=True):
    # Setup Lennard-Jones simulation
    sim = Simulation()
    fcc_unit_cell = (
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5)
    )
    sim.set_positions(unit_cell_coordinates=fcc_unit_cell, cells=(4, 4, 4), lattice_constants=(1.0, 1.0, 1.0))
    sim.set_density(density=0.8)
    sim.set_masses(masses=1.0)
    sim.set_random_velocities(temperature=1.0)
    sim.set_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
    sim.set_pair_potential_parameters(sigma=1.0, epsilon=1.0)
    sim.set_neighbor_list(skin=0.5, max_number_of_neighbors=128)
    sim.set_integrator(time_step=0.01, target_temperature=1.0, temperature_damping_time=0.1)
    sim.set_neighbor_list(skin=1.0, max_number_of_neighbors=512)

    sim = autotune(sim,
                   verbose=verbose,
                   plot=plot,
                   test_double_loop=test_double_loop,
                   steps=10)
    assert 0.1 < sim.neighbor_list_skin < 1.9, f'{sim.neighbor_list_skin=}'


if __name__ == '__main__':
    test_autotune(verbose=True, plot=True, test_double_loop=True)
