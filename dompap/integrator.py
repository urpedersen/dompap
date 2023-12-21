import numpy as np
import numba


def _make_step(positions: np.ndarray,
               velocities: np.ndarray,
               betas: np.ndarray,
               forces: np.ndarray,
               masses: np.ndarray,
               time_step: np.float64,
               temperature_target: np.float64,
               temperature_damping_time: np.float64) -> tuple:
    """ Make one step in the simulation using Leap-Frog G-JF thermostat
    Reference: https://arxiv.org/1303.7011, Section II.C, Eqs. (16) and (17).
    """
    alphas = masses / temperature_damping_time
    # Make random numbers with normal distribution and same shape as velocities
    random_numbers = np.random.normal(size=velocities.shape)
    beta_variance = 2 * alphas * temperature_target
    new_betas = np.sqrt(beta_variance) * random_numbers
    numerator = 2.0 * masses - alphas * time_step
    denominator = 2.0 * masses + alphas * time_step
    a = numerator / denominator
    b = 1 / denominator
    velocities = a * velocities + b * time_step * forces / masses + b * (betas + new_betas) / 2 / masses
    positions = positions + velocities * time_step
    betas = new_betas
    return positions, velocities, betas


def test_make_step():
    positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    velocities = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float64)
    betas = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float64)
    forces = np.array([[-1, 0, 0], [1, 0, 0]], dtype=np.float64)
    masses = np.array([[1], [1]], dtype=np.float64)
    time_step = np.float64(0.01)
    temperature_target = np.float64(1.0)
    temperature_damping_time = np.float64(1.0)
    new_state = _make_step(positions, velocities, betas, forces, masses, time_step, temperature_target,
                           temperature_damping_time)
    positions_new, velocities_new, betas_new = new_state
    # Assert that the first particle has moved
    assert positions_new[0, 0] != 0.0
