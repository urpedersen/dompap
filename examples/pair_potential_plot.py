""" Plot the pair potential and force """

import matplotlib.pyplot as plt
import numpy as np

from dompap import Simulation

sim = Simulation()
sim.set_pair_potential(pair_potential_str='(1-r)**2', r_cut=1.0)

plt.figure()
r = np.linspace(0.95, 1.2, 1000)
plt.plot(r, sim.pair_potential(r), label='Pair potential.py')
plt.plot(r, sim.pair_force(r), label='Pair force')
plt.xlabel(r'Pair distance, $r$ [$\sigma$]')
plt.legend()
plt.show()
