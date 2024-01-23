""" Load simulation data from disk. """

from pprint import pprint

from dompap import Simulation

sim = Simulation()

# Load simulation data from disk.
metadata = sim.from_disk(particle_data='simulation.csv', meta_data='simulation.toml',
                         verbose=True, set_only_particle_data=False)

print('Meta data from simulation data loaded from disk:')
pprint(metadata)

