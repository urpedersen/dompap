from dompap import Simulation


def to_lammps_dump(sim: Simulation) -> str:
    """ Return positions and box vectors to a LAMMPS dump file """
    number_of_steps = sim.number_of_steps
    box_vectors = sim.box_vectors

    particle_types = sim.particle_types
    masses = sim.masses
    diameters = sim.get_diameters()
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
        out += f'0.0 {box_vectors[dim]}\n'
    if dimension_of_space == 2:
        out += 'ITEM: ATOMS id type diameter mass x y ix iy vx vy fx fy\n'
        index: int = 0
        for n in range(number_of_atoms):
            out += f'{index} {particle_types[n]} {float(diameters[n])} {masses[n]} '
            out += f'{positions[n, 0]} {positions[n, 1]} '
            out += f'{image_positions[n, 0]} {image_positions[n, 1]} '
            out += f'{velocities[n, 0]} {velocities[n, 1]} '
            out += f'{forces[n, 0]} {forces[n, 1]}\n'
    if dimension_of_space == 3:
        out += 'ITEM: ATOMS id type diameter mass x y z ix iy iz vx vy vz fx fy fz'
        index: int = 0
        for n in range(number_of_atoms):
            out += f'\n{index} {int(particle_types[n])} {float(diameters[n])} {float(masses[n])} '
            out += f'{positions[n, 0]} {positions[n, 1]} {positions[n, 2]} '
            out += f'{int(image_positions[n, 0])} {int(image_positions[n, 1])} {int(image_positions[n, 2])} '
            out += f'{velocities[n, 0]} {velocities[n, 1]} {velocities[n, 2]} '
            out += f'{forces[n, 0]} {forces[n, 1]} {forces[n, 2]}'
            index += 1
    return out


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
