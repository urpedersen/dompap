from dompap import Simulation

# Setup default simulation
sim = Simulation()

# Run simulation
steps = 100
for step in range(steps):
    sim.make_step()
    if step % 10 == 0:
        print(f'Energy after {step} steps: {sim.get_potential_energy()}')
