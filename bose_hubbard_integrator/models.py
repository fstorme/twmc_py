from dataclasses import dataclass
import numpy as np


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


@dataclass
class Results:
    t_obs: np.array
    beta: np.array
