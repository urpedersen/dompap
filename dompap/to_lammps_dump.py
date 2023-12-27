from dompap import Simulation

def to_lammps_dump(sim: Simulation) -> str:
    """ Return positions and box vectors to a LAMMPS dump file """
    number_of_steps = sim.number_of_steps
    box_vectors = sim.box_vectors

    particle_types = sim.particle_types
    masses = sim.masses
    positions = sim.positions
    image_positions = sim.image_positions
    velocities = sim.velocities
    forces = sim.get_forces()

    out = 'ITEM: TIMESTEP\n' + f'{number_of_steps:d}\n'
    number_of_atoms = positions.shape[0]
    out += f'ITEM: NUMBER OF ATOMS\n{number_of_atoms:d}\n'
    dimension_of_space = positions.shape[1]
    if dimension_of_space == 2:
        out += 'ITEM: BOX BOUNDS pp pp\n'
    if dimension_of_space == 3:
        out += 'ITEM: BOX BOUNDS pp pp pp\n'
    else:
        raise ValueError('Dimension of space must be 2 or 3')
    for dim in range(dimension_of_space):
        out += f'0.0 {box_vectors[dim]:.16f}\n'
    if dimension_of_space == 2:
        out += 'ITEM: ATOMS id type mass x y ix iy vx vy fx fy\n'
        index: int = 0
        for n in range(number_of_atoms):
            out += f'{index} {particle_types[n]} {masses[n]} '
            out += f'{positions[n, 0]} {positions[n, 1]} '
            out += f'{image_positions[n, 0]} {image_positions[n, 1]} '
            out += f'{velocities[n, 0]} {velocities[n, 1]} '
            out += f'{forces[n, 0]} {forces[n, 1]}\n'
    if dimension_of_space == 3:
        out += 'ITEM: ATOMS id type mass x y z ix iy iz vx vy vz fx fy fz\n'
        index: int = 0
        for n in range(number_of_atoms):
            out += f'{index} {particle_types[n]} {masses[n]} '
            out += f'{positions[n, 0]} {positions[n, 1]} {positions[n, 2]} '
            out += f'{image_positions[n, 0]} {image_positions[n, 1]} {image_positions[n, 2]} '
            out += f'{velocities[n, 0]} {velocities[n, 1]} {velocities[n, 2]} '
            out += f'{forces[n, 0]} {forces[n, 1]} {forces[n, 2]}\n'
    return out

TEST_STR="""ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 5.0000000000000000
0.0 5.0000000000000000
0.0 5.0000000000000000
ITEM: ATOMS id type mass x y z ix iy iz vx vy vz fx fy fz
0 1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0 2 2.0 1.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
"""

def test_to_lammps_dump():
    import numpy as np
    sim = Simulation()
    sim.number_of_steps = 0
    sim.box_vectors = np.array([5, 5, 5], dtype=np.float64)
    sim.particle_types = np.array([1, 2], dtype=np.int32)
    sim.masses = np.array([1.0, 2.0], dtype=np.float64)
    sim.positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    sim.image_positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    sim.velocities = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float64)
    sim.forces = np.array([[-1, 0, 0], [1, 0, 0]], dtype=np.float64)
    out = to_lammps_dump(sim)
    assert out == TEST_STR
