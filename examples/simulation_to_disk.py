""" Save simulation data to disk.
Particle data is saved as CSV file, and meta data is saved as TOML file.
"""

from dompap import Simulation

sim = Simulation()
sim.to_disk(particle_data='simulation.csv', meta_data='simulation.toml')
