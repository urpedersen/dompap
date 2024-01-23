""" Generate a LAMMPS dump file. """

from dompap import Simulation, to_lammps_dump

# Setup default simulation
sim = Simulation()

# Set parameters
steps = 100  # Number of steps to run
number_of_dumps = 10  # Number of dumps to make
stride = steps // number_of_dumps  # Stride between dumps

# Run simulation
for step in range(steps):
    if step % stride == 0:
        print(to_lammps_dump(sim), file=open(f'dump.lammps', 'a'))  # Append to file
    sim.step()  # Make simulation step
