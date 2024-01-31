""" Progress bar during simulation. """

from dompap import Simulation
from dompap.tools import progress_bar

sim = Simulation()

# Simulation parameters
steps = 6_000  # Number of steps to run
stride = 20  # Stride between evaluations

# Run simulation and print progress
print(f'Evaluate properties every {stride} steps for a total of {steps} steps.')
for step in range(steps):
    if step % stride == 0:
        progress_bar(step, steps, stride)
        ...  # Evaluate properties here
    sim.step()
progress_bar(steps, steps, stride, finalize=True)
