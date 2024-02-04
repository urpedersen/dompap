from time import perf_counter

from dompap import Simulation
from dompap.tools import progress_bar, autotune

def run_test_simulation(verbose=False) -> Simulation:
    if verbose:
        print('  ..:: Run test simulation ::..')
    tic = perf_counter()
    sim = Simulation()
    toc = perf_counter()
    if verbose:
        print(f'Simulation initialization took {toc - tic:.3f} seconds')
    sim = autotune(sim, verbose=verbose)
    steps = 1000
    stride = 10
    for step in range(steps):
        sim.step()
        if step % stride == 0 and verbose:
            progress_bar(step, steps, stride)
    if verbose:
        progress_bar(steps, steps, stride, finalize=True)

    return sim


if __name__ == '__main__':
    run_test_simulation(verbose=True)
