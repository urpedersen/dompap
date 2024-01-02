from dompap import Simulation


def autotune(sim: Simulation, skin=0.5, max_number_of_neighbors=512, steps=100,
             verbose=False, plot=False) -> Simulation:
    from time import perf_counter

    sim_copy = sim.copy()
    sim_copy.step()  # Run one step to initialize

    skin_values = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
    times = []
    for skin in skin_values:
        sim_copy.set_neighbor_list(skin=skin, max_number_of_neighbors=max_number_of_neighbors)
        tic = perf_counter()
        sim_copy.run(steps)
        toc = perf_counter()
        times.append(toc - tic)

    # Print table with skin and time values
    if verbose:
        print(' Skin | Time  ')
        for skin, time in zip(skin_values, times):
            print(f'{skin:4.1f} | {time:6.2f}')

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(skin_values, times)
        plt.xlabel('Skin')
        plt.ylabel('Time')
        plt.show()

    # Find skin value with minimum time
    skin = skin_values[times.index(min(times))]
    sim.set_neighbor_list(skin=skin, max_number_of_neighbors=max_number_of_neighbors)
    return sim


def test_autotune(verbose=False, plot=False):
    # Setup Lennard-Jones simulation
    sim = Simulation()
    sim.set_positions(unit_cell_coordinates=([0.0, 0.0, 0.0],), cells=(8, 8, 8), lattice_constants=(1.0, 1.0, 1.0))
    sim.set_masses(masses=1.0)
    sim.set_random_velocities(temperature=1.0)
    sim.set_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)
    sim.set_pair_potential_parameters(sigma=1.0, epsilon=1.0)
    sim.set_neighbor_list(skin=0.5, max_number_of_neighbors=128)
    sim.set_integrator(time_step=0.01, target_temperature=1.0, temperature_damping_time=0.1)
    sim.set_neighbor_list(skin=1.0, max_number_of_neighbors=512)

    sim = autotune(sim, verbose=verbose, plot=plot)
    assert 0.6 < sim.neighbor_list_skin < 1.7, f'{sim.neighbor_list_skin=}'


if __name__ == '__main__':
    test_autotune(verbose=True, plot=True)
