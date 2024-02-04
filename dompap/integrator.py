import numpy as np
import numba


@numba.njit
def make_one_step_leap_frog(positions: np.ndarray,
                            velocities: np.ndarray,
                            forces: np.ndarray,
                            masses: np.ndarray,
                            time_step: np.float64) -> [np.ndarray, np.ndarray]:
    """ Make one step in the simulation using the Leap-Frog algorithm """
    velocities = velocities + time_step * forces / masses
    positions = positions + time_step * velocities
    return positions, velocities


@numba.njit
def make_one_step(positions: np.ndarray,
                  velocities: np.ndarray,
                  betas: np.ndarray,
                  forces: np.ndarray,
                  masses: np.ndarray,
                  time_step: np.float64,
                  temperature_target: np.float64,
                  temperature_damping_time: np.float64) -> [np.ndarray, np.ndarray, np.ndarray]:
    """ Make one step in the simulation using Leap-Frog G-JF thermostat
    Reference: https://arxiv.org/1303.7011, Section II.C, Eqs. (16) and (17).
    """
    alphas = masses / temperature_damping_time
    random_numbers = np.random.normal(0, 1, velocities.shape)
    beta_variance = 2 * alphas * temperature_target * time_step
    new_betas = np.sqrt(beta_variance) * random_numbers
    numerator = 1.0 - alphas * time_step / 2.0 / masses
    denominator = 1.0 + alphas * time_step / 2.0 / masses
    a = numerator / denominator
    b = 1 / denominator
    velocities = a * velocities + b * time_step * forces / masses + b * (betas + new_betas) / 2 / masses
    positions = positions + velocities * time_step
    betas = new_betas
    return positions, velocities, betas
