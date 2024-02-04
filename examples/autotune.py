""" Automate finding the best parameters for an efficient simulation. """

from dompap import Simulation
from dompap.tools import autotune

sim = Simulation()
sim = autotune(sim, verbose=True, plot=True)
