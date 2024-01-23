""" Calculate the radial distribution function at runtime """

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dompap import Simulation

# Setup default simulation
sim = Simulation()

# Equilibrate
sim.set_random_velocities(temperature=0.02)
sim.set_integrator(target_temperature=0.02)
sim.run(steps=1000)

# Production run
r_bins = np.arange(0.01, 2.5, 0.01)
r, rdf = sim.get_radial_distribution_function(r_bins=r_bins)
rdf_evaluations = 1
steps = 80_000
stride = 40
for step in range(steps):
    sim.step()
    if step % stride == 0:
        _, this_rdf = sim.get_radial_distribution_function(r_bins=r_bins)
        rdf += this_rdf
        rdf_evaluations += 1
rdf /= rdf_evaluations + 1

# Plot radial distribution function
plt.figure(figsize=(6, 4))
plt.title(f'Made with {__file__.split("/")[-1]}')
plt.plot(r, rdf, '-')
plt.xlabel(r'Pair distance, $r$')
plt.ylabel(r'Radial distribution function, $g(r)$')
plt.xlim(0, 2.5)
plt.ylim(0, None)
plt.savefig('radial_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Save to csv-file
pd.DataFrame({'r': r, 'g(r)': rdf}).to_csv('radial_distribution.csv', index=False)
