from dompap import Simulation
from dompap import print_progress

sim = Simulation()

# Simulation parameters
steps = 6_000  # Number of steps to run
stride = 20  # Stride between evaluations

# Run simulation and print progress
print(f'Evaluate properties every {stride} steps.', end='')
for step in range(steps):
    if step % stride == 0:
        print_progress(step, steps, stride)
        ...  # Evaluate properties here
    sim.step()
print_progress(steps, steps, stride, finalize=True)
