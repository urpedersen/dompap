from dompap import Simulation
from time import perf_counter

tic = perf_counter()

sim = Simulation()

toc = perf_counter()
print(f'Time to setup: {toc - tic:.2f} seconds')

tic = perf_counter()
sim.step()
toc = perf_counter()
print(f'Time to make first step: {toc - tic:.2f} seconds')

tic = perf_counter()
sim.step()
toc = perf_counter()
print(f'Time to make second step: {(toc - tic)*1000:.3f} milliseconds')

tic = perf_counter()
# Equilibrate
steps_eq = 100
sim.run(steps_eq)
toc = perf_counter()
print(f'Time to equilibrate: {toc - tic:.2f} seconds')

run_time = 1.0  # seconds
tic = perf_counter()
steps_per_round = 1000
rounds = 0
while perf_counter() - tic < run_time:
    sim.run(steps_per_round)
    rounds += 1
toc = perf_counter()

steps_per_second = rounds * steps_per_round / (toc - tic)

print(f'Average steps per second: {steps_per_second:.2f}')
