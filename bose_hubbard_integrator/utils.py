from bose_hubbard_integrator.models import Results
import numpy as np


def save_results(path: str, r: Results):
    np.savez_compressed(path, t_obs=r.t_obs, beta=r.beta)


def load_results(path: str) -> Results:
    data = np.load(path)
    return Results(t_obs=data["t_obs"], beta=data["beta"])
