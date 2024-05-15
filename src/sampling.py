from typing import Tuple
from torch import zeros, Tensor

from src.dynamics import RobotWorld


class Sampler:
    @staticmethod
    def estimate_V_rho(env: RobotWorld, P: Tensor, n: int) -> float:
        v = 0
        for _ in range(n):
            s = env.reset()
            v += s @ P @ s
        return (v / n).detach().item()

    @staticmethod
    def sample_trajectory(env: RobotWorld, K: Tensor, T: int) -> Tuple[Tensor, Tensor]:
        states = zeros([T, 4])
        actions = zeros([T, 2])

        s = env.reset()
        for i in range(0, T):
            u = s @ K.T
            sp = env.step(u)

            states[i] = s
            actions[i] = u

            s = sp
        return states, actions
