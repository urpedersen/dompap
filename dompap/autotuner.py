from dompap import Simulation


def autotune(sim: Simulation, steps=100, test_double_loop=True,
             smallest_skin=0.1, step_skin=0.1,
             verbose=False, plot=False) -> Simulation:
    from time import perf_counter

    sim_copy = sim.copy()
    sim_copy.step()  # Run one step to initialize

    # Find the fastest method for neighbor list update method
    methods = ['double loop', 'cell list']
    results = dict()
    for method in methods:
        sim_copy.set_neighbor_list(method_str=method)
        times = []
        for _ in range(4):
            tic = perf_counter()
            sim_copy.update_neighbor_list(check=False)  # Force update by setting check=False
            toc = perf_counter()
            times.append(toc - tic)
            if verbose:
                print(f'Time to update neighbor list ({method}): {(toc - tic) * 1000:.3f} milliseconds')
        results[method] = min(times)
    fastest_time = min(results.values())
    fastest_method = list(results.keys())[list(results.values()).index(fastest_time)]
    if verbose:
        print(f'Fastest method to update neighbor list: {fastest_method}')
    sim.neighbor_list_method_str = fastest_method
    sim_copy.neighbor_list_method_str = fastest_method

    largest_allowd_skin = min(sim.box_vectors) / 2
    print(f'{largest_allowd_skin=}')
    skins = []
    times = []
    neighbor_list_updates = []
    skin = smallest_skin
    old_time = float('inf')
    while skin < largest_allowd_skin:
        old_num_updates = sim_copy.number_of_neighbor_list_updates
        sim_copy.set_neighbor_list(skin=skin)
        tic = perf_counter()
        sim_copy.run(steps)
        toc = perf_counter()
        skins.append(skin)
        times.append(toc - tic)
        neighbor_list_updates.append(sim_copy.number_of_neighbor_list_updates - old_num_updates)
        skin += step_skin
        if toc - tic > old_time:
            break
        old_time = toc - tic

    # Print table with skin and time values
    if verbose:
        print('Skin | Time per step (ms) | steps/updates')
        for skin, time, neighbor_list_update in zip(skins, times, neighbor_list_updates):
            print(f'{skin:4.1f} | {time/steps*1000:6.4f}            '
                  f' | {steps}/{neighbor_list_update} = {steps/neighbor_list_update:0.1f}')

    # Find skin value with minimum time
    fastest_time = min(times)
    skin = skins[times.index(min(times))]
    if verbose:
        print(f'Optimal parameters: {skin=}')

    sim.set_neighbor_list(skin=skin, method_str=fastest_method)


    # Time to compute force (if verbose)
    if verbose:
        sim_copy = sim.copy()
        for _ in range(4):
            tic = perf_counter()
            sim_copy.get_forces()
            toc = perf_counter()
            print(f'Time to compute forces ({sim_copy.neighbor_list_method_str}, '
                  f'skin={sim_copy.neighbor_list_skin:0.4f}): '
                  f'{(toc - tic) * 1000:.3f} milliseconds')

    # Test double loop method for force (multicore; no neighbor list)
    time_double_loop: float = None
    if test_double_loop:
        sim_copy = sim.copy()
        sim_copy.force_method_str = 'double loop'
        sim_copy.step()  # Run one step to initialize
        tic = perf_counter()
        sim_copy.run(steps)
        toc = perf_counter()
        time_double_loop = toc - tic
        if verbose:
            print(f'Time with double loop: {time_double_loop/steps*1000:0.4f} milliseconds')
        if time_double_loop < fastest_time:
            fastest_time = time_double_loop
            sim.force_method_str = 'double loop'


    # Test double loop single core method for force (no neighbor list)
    time_double_loop_single_core: float = None
    if test_double_loop:
        sim_copy = sim.copy()
        sim_copy.force_method_str = 'double loop single core'
        sim_copy.step()  # Run one step to initialize
        tic = perf_counter()
        sim_copy.run(steps)
        toc = perf_counter()
        time_double_loop_single_core = toc - tic
        if verbose:
            print(f'Time with double loop single core: {time_double_loop_single_core/steps*1000:0.4f} milliseconds')
        if time_double_loop_single_core < fastest_time:
            fastest_time = time_double_loop_single_core
            sim.force_method_str = 'double loop single core'


    # Test vectorized method for force (NumPy, no Numba)
    if test_double_loop:
        sim_copy = sim.copy()
        sim_copy.force_method_str = 'vectorized'
        sim_copy.step()  # Run one step to initialize
        tic = perf_counter()
        sim_copy.run(steps)
        toc = perf_counter()
        time_vectorized = toc - tic
        if verbose:
            print(f'Time with vectorized: {time_vectorized/steps*1000:0.4f} milliseconds')
        if time_vectorized < fastest_time:
            fastest_time = time_vectorized
            sim.force_method_str = 'vectorized'

    if verbose:
        print(f'Fastest method: {sim.force_method_str}')


    # Make plot
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(skins, times, 'o', label='Neighbour list')
        # Red for at fastest time
        plt.plot(skin, fastest_time, 'ro', label='Fastest time')
        if test_double_loop:
            plt.plot(skins, [time_double_loop] * len(skins), '--', label='Double loop')
        plt.xlabel('Skin')
        plt.ylabel('Time')
        plt.legend()
        plt.show()

    return sim


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

    sim = autotune(sim, verbose=verbose, plot=plot, test_double_loop=test_double_loop)
    assert 0.3 < sim.neighbor_list_skin < 1.7, f'{sim.neighbor_list_skin=}'


if __name__ == '__main__':
    test_autotune(verbose=True, plot=True, test_double_loop=True)
