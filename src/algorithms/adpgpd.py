from typing import List, Tuple
import numpy as np
from numpy.random import default_rng
from torch import (
    eye, inverse, zeros, clamp, concat, cartesian_prod, tensor, ger, flatten, diag, Tensor
)
from torch.linalg import lstsq

from src.dynamics import RobotWorld
from src.lqr import Lqr
from src.sampling import Sampler


RNG_SAMPLING = [-10, 10]

# This class implements the AD-PGPD algorithm for the inexact case
class ADpgpd:
    def __init__(
            self,
            env: RobotWorld,
            eta: float,
            tau: float,
            gamma: float,
            b: float,
            alpha: float,
            G1: Tensor,
            G2: Tensor,
            R1: Tensor,
            R2: Tensor,
            H: Tensor,
        ) -> None:
        self.env = env
        self.eta = eta
        self.tau = tau
        self.I_eta = eye(env.B.shape[1]).double() * (1 / eta)
        self.gamma = gamma
        self.b = b
        self.alpha = alpha
        self.ds, self.da = env.B.shape

        self.G1 = G1
        self.G2 = G2
        self.R1 = R1
        self.R2 = R2
        self.H = H

        self.lqr = Lqr(env.A, env.B, gamma)
        self.sampler = Sampler(env, gamma)

    def get_initial_point(self) -> Tuple[Tensor, Tensor]:
        rng = default_rng()

        s = tensor([
            rng.uniform(*RNG_SAMPLING),
            rng.uniform(*RNG_SAMPLING),
            rng.uniform(*RNG_SAMPLING),
            rng.uniform(*RNG_SAMPLING),
        ]).double()

        a = tensor([
            rng.uniform(*RNG_SAMPLING),
            rng.uniform(*RNG_SAMPLING),
        ]).double()

        return s, a

    def policy_evaluation(self, K: Tensor, G: Tensor, R: Tensor, n_samples: int) -> Tensor:
        X, y = zeros((n_samples, 36)), zeros(n_samples)
        P = self.lqr.calculate_P(K, G, R)
        H = self.lqr.calculate_H(P, G, R)
        for n in range(n_samples):
            s, a = self.get_initial_point()
            s_a = concat([s, a])
            b = flatten(ger(s_a, s_a) + diag((self.alpha - 1) * s_a**2))

            a_pi = s @ K.T
            q = self.lqr.calculate_Q(s, a, H)
            l = q + (1 / self.eta) * a_pi @ a

            X[n] = b
            y[n] = l
        theta = lstsq(X, y, driver='gelsd').solution
        return theta

    # This method implements the primal update in equation 9a
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

    # This method implements the dual update in equation 9b
    def dual_update(self, P: Tensor, lmbda: Tensor, n: int) -> Tensor:
        v = self.sampler.estimate_V_rho_closed(P, n)
        return clamp(lmbda - self.eta * (v - self.b + self.tau * lmbda), min=0)

    def train_unconstrained(self, epochs: int, n_pe: int, n_rho: int) -> Tuple[Tensor, List[float], List[float]]:
        losses_primal, losses_dual = [], []
        theta = zeros((self.ds + self.da)**2)
        K = self.primal_update(theta)
        for e in range(epochs):
            P_primal = self.lqr.calculate_P(K, self.G1, self.R1 + self.H)
            P_dual = self.lqr.calculate_P(K, self.G2, self.R2)

            loss_primal = self.sampler.estimate_V_rho_closed(P_primal, n_rho)
            loss_dual = self.sampler.estimate_V_rho_closed(P_dual, n_rho)

            losses_primal.append(loss_primal)
            losses_dual.append(loss_dual)

            theta = self.policy_evaluation(K, self.G1, self.R1 + self.H, n_pe)
            K = self.primal_update(theta)

            print(f"Episode {e}/{epochs} - Return {loss_primal} \r", end='')
        return K, losses_primal, losses_dual

    # This method iterates the primal and the dual update
    def train_constrained(self, epochs: int, n_pe: int, n_rho: int) -> Tuple[Tensor, Tensor, List[float], List[float]]:
        losses_primal, losses_dual = [], []
        theta = zeros((self.ds + self.da)**2)
        K = self.primal_update(theta)
        lmbda = zeros(1)
        for e in range(epochs):
            Gl, Rl = self.G1 + lmbda * self.G2, self.R1 + self.H + lmbda * self.R2

            P_primal = self.lqr.calculate_P(K, self.G1, self.R1)
            P_dual = self.lqr.calculate_P(K, self.G2, self.R2)

            if e % 10 == 0:
                loss_primal = self.sampler.estimate_V_rho_closed(P_primal, n_rho)
                loss_dual = self.sampler.estimate_V_rho_closed(P_dual, n_rho)

                losses_primal.append(loss_primal)
                losses_dual.append(loss_dual)

            theta = self.policy_evaluation(K, Gl, Rl, n_pe)
            K = self.primal_update(theta)
            lmbda = self.dual_update(P_dual, lmbda, n_rho)

            print(f'Episode {e}/{epochs} - Return {loss_primal} - Constrain {loss_dual} - Lambda {lmbda.detach().item()} \r', end='')
        return K, lmbda, losses_primal, losses_dual

    def evaluate_duality_gap(self, V_primal_opt: float, epochs: int, n_pe: int, n_rho: int):
        theta = zeros((self.ds + self.da)**2)
        K = self.primal_update(theta)
        lmbda = zeros(1)
        gaps = []
        for e in range(epochs):
            Gl, Rl = self.G1 + lmbda * self.G2, self.R1 + self.H + lmbda * self.R2

            P_primal = self.lqr.calculate_P(K, self.G1, self.R1)
            P_dual = self.lqr.calculate_P(K, self.G2, self.R2)

            if e % 10 == 0:
                V_primal = self.sampler.estimate_V_rho_closed(P_primal, n_rho)
                g = np.linalg.norm(V_primal_opt - V_primal)
                gaps.append(g)
                print(f"Episode {e}/{epochs} - Dual Gap {g} - Lambda {lmbda.detach().item()} - V {V_primal} \r", end='')

            theta = self.policy_evaluation(K, Gl, Rl, n_pe)
            K = self.primal_update(theta)
            lmbda = self.dual_update(P_dual, lmbda, n_rho)
        return gaps
