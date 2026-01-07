import numpy as np
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class HamiltionanParameters:
    u: float
    gamma: float
    f: float
    omega: float = 1


@dataclass
class EvolutionParameters:
    t_start: int = 0
    t_end: float = 200
    n_frames: int = 60
    n_config: int = 120


def main(h_params: HamiltionanParameters, evol_params: EvolutionParameters):
    # frequency and dt set up for the time evolution
    w_max = np.max(
        np.max(np.abs(np.array([h_params.u, h_params.gamma, h_params.omega])))
    )
    n_times = np.round(np.log2((evol_params.t_end - evol_params.t_start) * w_max))
    dt = (evol_params.t_end - evol_params.t_start) / (2 ** (n_times + 9))
    dt_obs = (evol_params.t_end - evol_params.t_start) / evol_params.n_frames
    t_vec = evol_params.t_start + dt * np.arange(2 ** (n_times + 9) + 1)
    frames = range(0, len(t_vec), int(np.floor(dt_obs / dt)))
    t_obs = t_vec[frames]

    # Rescaling constants
    u = dt * h_params.u
    omega = dt * h_params.omega
    gamma = dt * h_params.gamma
    f = dt * h_params.f

    def f_NL(beta, t):
        return 1j * u * (np.abs(beta) ** 2 - 1) * beta + 1j * f

    exp_elt = np.exp((-1j * omega - gamma / 2))
    beta_init = 0
    beta_tot = np.zeros((len(frames), evol_params.n_config), dtype=np.complex64)

    for k in tqdm(range(evol_params.n_config), desc="Evolving trajectories"):
        beta = beta_init + np.sqrt(h_params.gamma * dt / 4) * np.random.normal()
        i_frames = 0
        for t in t_vec:
            beta_1 = (
                exp_elt * beta
                + f_NL(beta, t)
                + np.sqrt(gamma / 4) * (np.random.normal() + 1j * np.random.normal())
            )

            beta = (
                exp_elt * beta
                + 1 / 2 * (f_NL(beta, t) + f_NL(beta_1, t + dt))
                + np.sqrt(gamma / 4) * (np.random.normal() + 1j * np.random.normal())
            )
            if t in t_obs:
                beta_tot[i_frames, k] = beta
                i_frames += 1
    return beta_tot


if __name__ == "__main__":
    h_params = HamiltionanParameters(u=0.1, gamma=1, f=5)
    evol_params = EvolutionParameters(t_end=30, n_frames=100, n_config=200)

    main(h_params=h_params, evol_params=evol_params)
