from typing import List, Tuple, Callable
from numpy.random import default_rng
from torch import (
    eye, inverse, zeros, clamp, concat, cartesian_prod, tensor, ger, flatten, diag, einsum, Tensor
)
from torch.linalg import lstsq

from src.dynamics import RobotWorld
from src.sampling import Sampler


class ADpgpdSampled:
    def __init__(
            self,
            ds: int,
            da: int,
            env: RobotWorld,
            eta: float,
            tau: float,
            gamma: float,
            b: float,
            alpha: float,
            primal_reward_fn: Callable,
            primal_reward_reg_fn: Callable,
            dual_reward_fn: Callable,
            starting_pos_fn: Callable,
        ) -> None:
        self.env = env
        self.eta = eta
        self.tau = tau
        self.gamma = gamma
        self.b = b
        self.alpha = alpha
        self.ds, self.da = ds, da

        self.primal_reward_fn = primal_reward_fn
        self.primal_reward_reg_fn = primal_reward_reg_fn
        self.dual_reward_fn = dual_reward_fn
        self.starting_pos_fn = starting_pos_fn

        self.sampler = Sampler(env, gamma)

    def policy_evaluation(self, K: Tensor, lmbda: Tensor, n_samples: int, n_rollout: int) -> Tensor:
        s, a = self.starting_pos_fn(n_samples)
        s_a = concat([s, a], dim=1)
        X = einsum("bi,bj->bij", s_a, s_a).view(n_samples, (self.ds + self.da)**2)

        def reward_fn(env, action):
            return self.primal_reward_fn(env, action) + self.primal_reward_reg_fn(env, action) + lmbda * self.dual_reward_fn(env, action)

        a_pi = s @ K.T
        q = self.sampler.rollout_Q(s, a, K, n_rollout, reward_fn=reward_fn)
        y = q + (1 / self.eta) * diag(a_pi @ a.T)

        theta = lstsq(X, y, driver='gelsd').solution
        return theta

    def primal_update(self, theta: Tensor) -> Tensor:
        W_1 = zeros((self.da, self.ds))
        for i in range(self.da):
            for j in range(self.ds):
                s_idx, a_idx = zeros(self.ds), zeros(self.da)
                s_idx[j] = -1
                a_idx[i] = 1
                s_a_idx = concat([s_idx, a_idx])
                mask = - cartesian_prod(s_a_idx, s_a_idx).prod(dim=1).clip(-1, 0) 
                w = (theta * mask).sum()
                W_1[i, j] = w

        W_2 = zeros((self.da, self.da))
        for i in range(self.da):
            for j in range(self.da):
                s_idx, a_idx = zeros(self.ds), zeros(self.da)
                a_idx[i] = 1
                a_idx[j] = -1
                s_a_idx = concat([s_idx, a_idx])

                if i == j:
                    mask = cartesian_prod(s_a_idx, s_a_idx).prod(dim=1)
                    w = (theta * mask).sum()
                    W_2[i, j] = 2 * w * self.alpha
                else:
                    mask = - cartesian_prod(s_a_idx, s_a_idx).prod(dim=1).clip(-1, 0) 
                    w = (theta * mask).sum()
                    W_2[i, j] = w
        K = - inverse(W_2 - (self.tau + 1 / self.eta) * eye(self.da)) @ W_1
        return K.double()

    def dual_update(self, K: Tensor, lmbda: Tensor, n_samples: int, n_rollout) -> Tensor:
        v = self.sampler.estimate_V_rho_rollout(K, n_samples, n_rollout, self.dual_reward_fn)
        return clamp(lmbda - self.eta * (v - self.b + self.tau * lmbda), min=0), v

    def train_unconstrained(self, epochs: int, n_pe: int, n_rho: int, n_roll: int) -> Tuple[Tensor, List[float], List[float]]:
        losses_primal, losses_dual = [], []
        theta = zeros((self.ds + self.da)**2)
        K = self.primal_update(theta)
        for e in range(epochs):
            if e % 10 == 0:
                loss_primal = self.sampler.estimate_V_rho_rollout(K, n_rho, n_roll, self.primal_reward_fn)
                loss_dual = self.sampler.estimate_V_rho_rollout(K, n_rho, n_roll, self.dual_reward_fn)
                losses_primal.append(loss_primal)
                losses_dual.append(loss_dual)

            theta = self.policy_evaluation(K, 0, n_pe, n_roll)
            K = self.primal_update(theta)

            print(f"Episode {e}/{epochs} - Return {loss_primal} \r", end='')
        return K, losses_primal, losses_dual

    def train_constrained(self, epochs: int, n_pe: int, n_rho: int, n_roll: int) -> Tuple[Tensor, Tensor, List[float], List[float]]:
        losses_primal, losses_dual = [], []
        theta = zeros((self.ds + self.da)**2)
        K = self.primal_update(theta)
        lmbda = zeros(1)
        for e in range(epochs):
            if e % 10 == 0:
                loss_primal = self.sampler.estimate_V_rho_rollout(K, n_rho, n_roll, self.primal_reward_fn)
                losses_primal.append(loss_primal)

            theta = self.policy_evaluation(K, lmbda, n_pe, n_roll)
            K = self.primal_update(theta)
            lmbda, loss_dual = self.dual_update(K, lmbda, n_rho, n_roll)
            losses_dual.append(loss_dual)

            print(f'Episode {e}/{epochs} - Return {loss_primal} - Constrain {loss_dual} - Lambda {lmbda.detach().item()}\r', end='')
        return K, lmbda, losses_primal, losses_dual

    def resume_training(self, K, lmbda, losses_primal, losses_dual, epochs: int, n_pe: int, n_rho: int, n_roll: int):
        for e in range(epochs):
            if e % 10 == 0:
                loss_primal = self.sampler.estimate_V_rho_rollout(K, n_rho, n_roll, self.primal_reward_fn)
                losses_primal.append(loss_primal)

            theta = self.policy_evaluation(K, lmbda, n_pe, n_roll)
            K = self.primal_update(theta)
            lmbda, loss_dual = self.dual_update(K, lmbda, n_rho, n_roll)
            losses_dual.append(loss_dual)

            print(f'Episode {e}/{epochs} - Return {loss_primal} - Constrain {loss_dual} - Lambda {lmbda.detach().item()}\r', end='')
        return K, lmbda, losses_primal, losses_dual
