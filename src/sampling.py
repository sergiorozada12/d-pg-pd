from typing import Tuple
from torch import zeros, Tensor

from src.dynamics import RobotWorld


class Sampler:
    def __init__(self, env: RobotWorld, gamma: float) -> None:
        self.env = env
        self.gamma = gamma

    def sample_trajectory(self, K: Tensor, T: int) -> Tuple[Tensor, Tensor]:
        states = zeros([T, 4])
        actions = zeros([T, 2])

        s = self.env.reset()
        for i in range(0, T):
            u = s @ K.T
            sp = self.env.step(u)

            states[i] = s
            actions[i] = u

            s = sp
        return states, actions

    def rollout_V(self, s: Tensor, K: Tensor, n: int, G: Tensor, R: Tensor) -> float:
        self.env.s = s
        v = 0
        for i in range(n):
            a = s @ K.T
            sp = self.env.step(a)

            r = s @ G @ s + a @ R @ a
            v += (self.gamma**i) * r

            s = sp
        return v

    def rollout_Q(self, s: Tensor, a: Tensor, K: Tensor, n: int, G: Tensor, R: Tensor) -> float:
        self.env.s = s
        sp = self.env.step(a)
        r = s @ G @ s + a @ R @ a
        s = sp

        q = r
        for i in range(1, n):
            a = s @ K.T
            sp = self.env.step(a)

            r = s @ G @ s + a @ R @ a
            q += (self.gamma**i) * r

            s = sp
        return q.detach().item()

    def estimate_V_rho(self, P: Tensor, n: int) -> float:
        v = 0
        for _ in range(n):
            s = self.env.reset()
            v += s @ P @ s
        return (v / n).detach().item()

    def rollout_V_rho(self, K: Tensor, G: Tensor, R: Tensor, n_samples: int, n_rollout) -> float:
        v = 0
        for _ in range(n_samples):
            s = self.env.reset()
            v += self.rollout_V(s, K, n_rollout, G, R)
        return (v / n_samples).detach().item()
