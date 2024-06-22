from typing import Tuple, Callable
from torch import zeros, Tensor

from src.dynamics import RobotWorld


class Sampler:
    def __init__(self, env: RobotWorld, gamma: float) -> None:
        self.env = env
        self.gamma = gamma

    def sample_trajectory(self, K: Tensor, T: int) -> Tuple[Tensor, Tensor]:
        states = zeros([T, K.shape[1]])
        actions = zeros([T, K.shape[0]])

        s = self.env.reset()
        for i in range(0, T):
            u = s @ K.T
            sp = self.env.step(u)

            states[i] = s
            actions[i] = u

            s = sp
        return states, actions

    def rollout_V(self, s: Tensor, K: Tensor, n: int, reward_fn: Callable) -> float:
        self.env.reset(s.shape[0])
        self.env.s = s
        v = 0
        for i in range(n):
            a = s @ K.T
            v += (self.gamma ** i) * reward_fn(self.env, a)
            s = self.env.step(a)
        return v

    def rollout_Q(self, s: Tensor, a: Tensor, K: Tensor, n: int, reward_fn: Callable) -> float:
        self.env.reset(s.shape[0])
        self.env.s = s
        q = reward_fn(self.env, a)
        s = self.env.step(a)
        for i in range(1, n):
            a = s @ K.T
            q += (self.gamma**i) * reward_fn(self.env, a)
            s = self.env.step(a)
        return q.detach()

    def estimate_V_rho_closed(self, P: Tensor, n: int) -> float:
        s = self.env.reset(n)
        return ((s @ P) * s).sum(dim=1).mean().item()

    def estimate_V_rho_rollout(self, K: Tensor, n_samples: int, n_rollout: int, reward_fn: Callable) -> float:
        s = self.env.reset(n_samples)
        v = 0
        for i in range(n_rollout):
            a = s @ K.T
            v += (self.gamma ** i) * reward_fn(self.env, a)
            s = self.env.step(a)
        return v.mean().detach().item()
