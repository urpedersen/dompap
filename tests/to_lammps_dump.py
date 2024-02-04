from dompap import Simulation
from dompap.tools import to_lammps_dump


def test_to_lammps_dump():
    import numpy as np
    sim = Simulation()
    sim.number_of_steps = 0
    sim.box_vectors = np.array([5, 5, 5], dtype=np.float64)
    sim.particle_types = np.array([[0], [0]], dtype=np.int32)
    sim.masses = np.array([[1.0], [2.0]], dtype=np.float64)
    sim.positions = np.array([[0, 0, 0], [0.5, 0, 0]], dtype=np.float64)
    sim.image_positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    sim.velocities = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float64)
    sim.forces = sim.get_forces()
    out = to_lammps_dump(sim)
    assert out == """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 5.0
0.0 5.0
0.0 5.0
ITEM: ATOMS id type diameter mass x y z ix iy iz vx vy vz fx fy fz
0 0 1.0 1.0 0.0 0.0 0.0 0 0 0 0.0 0.0 0.0 -1.0 0.0 0.0
1 0 1.0 2.0 0.5 0.0 0.0 1 0 0 0.0 0.0 0.0 1.0 0.0 0.0"""
