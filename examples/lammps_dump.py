from dompap import Simulation, to_lammps_dump

# Setup default simulation
sim = Simulation()

# Run simulation
steps = 100
for step in range(steps):
    if step % 1 == 0:
        lammps_str = to_lammps_dump(sim)
        with open(f'dump.lammps', 'a') as f:
            f.write(lammps_str)
    sim.step()
