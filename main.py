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

    def f_NL(beta, t):
        return (
            1j * h_params.u * (np.abs(beta) ** 2 - 1) * dt * beta + 1j * h_params.f * dt
        )

    beta_init = 0
    beta_tot = np.zeros(
        (evol_params.n_frames, evol_params.n_config), dtype=np.complex64
    )

    for k in tqdm(range(evol_params.n_config), desc="Evolving trajectories"):
        beta = beta_init + np.sqrt(h_params.gamma * dt / 4) * np.random.normal()
        t = 0
        i_step = 0
        i_frame = 0
        while t < evol_params.t_end:
            beta_1 = (
                np.exp((-1j * h_params.omega - h_params.gamma / 2) * dt) * beta
                + f_NL(beta, t)
                + np.sqrt(h_params.gamma * dt / 4)
                * (np.random.normal() + 1j * np.random.normal())
            )

            beta = (
                np.exp((-1j * h_params.omega - h_params.gamma / 2) * dt) * beta
                + 1 / 2 * (f_NL(beta, t) + f_NL(beta_1, t))
                + np.sqrt(h_params.gamma * dt / 4)
                * (np.random.normal() + 1j * np.random.normal())
            )

            t += dt
            if i_step % np.floor(dt_obs / dt) == 0:
                if i_frame <= evol_params.n_frames:
                    beta_tot[i_frame - 1, k] = beta
                    i_frame += 1

            i_step += 1
    return beta_tot


if __name__ == "__main__":
    h_params = HamiltionanParameters(u=0.1, gamma=1, f=5)
    evol_params = EvolutionParameters(t_end=30, n_frames=100, n_config=200)

    main(h_params=h_params, evol_params=evol_params)
