from datetime import datetime
from bose_hubbard_integrator.models import (
    HamiltionanParameters,
    EvolutionParameters,
)
from bose_hubbard_integrator.utils import save_results
from bose_hubbard_integrator.basic_integrator import (
    compute_time_vectors,
    integrate_mc_bose_hubbard,
)


def main(h_params: HamiltionanParameters, evol_params: EvolutionParameters):
    # frequency and dt set up for the time evolution
    dt, t_vec, t_obs, frames = compute_time_vectors(evol_params, h_params)
    results = integrate_mc_bose_hubbard(
        dt, t_vec, t_obs, frames, evol_params.n_config, h_params
    )
    return results


if __name__ == "__main__":
    h_params = HamiltionanParameters(u=0.1, gamma=1, f=5)
    evol_params = EvolutionParameters(t_end=30, n_frames=100, n_config=20)

    results = main(h_params=h_params, evol_params=evol_params)
    save_path = f"data/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_results(save_path, results)
