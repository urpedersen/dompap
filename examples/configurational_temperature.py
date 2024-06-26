from dompap import Simulation

# Setup default simulation
sim = Simulation()

# Run simulation
steps = 100
for step in range(steps):
    sim.step()
    T_conf = sim.get_configurational_temperature()
    if step % 10 == 0:
        print(f'Energy after {step} steps: {sim.get_potential_energy()}')
        print(f'Configurational temperature after {step} steps: {T_conf}')