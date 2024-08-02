from typing import List, Tuple
import numpy as np
from torch import eye, inverse, zeros, clamp, Tensor

from src.dynamics import RobotWorld
from src.lqr import Lqr
from src.sampling import Sampler


class Dpgpd:
    def __init__(
            self,
            env: RobotWorld,
            eta: float,
            tau: float,
            gamma: float,
            b: float,
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
        self.ds, self.da = env.B.shape

        self.G1 = G1
        self.G2 = G2
        self.R1 = R1
        self.R2 = R2
        self.H = H

        self.lqr = Lqr(env.A, env.B, gamma)
        self.sampler = Sampler(env, gamma)

    def primal_update(self, R: Tensor, K: Tensor, P: Tensor) -> Tensor:
        H_odiag = self.gamma * self.env.B.T @ P @ self.env.A
        H_diag = R + self.gamma * self.env.B.T @ P @ self.env.B
        return - inverse(2 * H_diag + 2 * R - self.I_eta) @ (2 * H_odiag + self.I_eta @ K)

    def dual_update(self, P: Tensor, lmbda: Tensor, n: int) -> Tensor:
        v = self.sampler.estimate_V_rho_closed(P, n)
        return clamp(lmbda - self.eta * (v - self.b + self.tau * lmbda), min=0)

    def train_unconstrained(self, epochs: int, n: int) -> Tuple[Tensor, List[float], List[float]]:
        losses_primal, losses_dual = [], []
        K = zeros(self.da, self.ds).double()
        for e in range(epochs):
            P_primal = self.lqr.calculate_P(K, self.G1, self.R1 + self.H)
            P_dual = self.lqr.calculate_P(K, self.G2, self.R2)

            loss_primal = self.sampler.estimate_V_rho_closed(P_primal, n)
            loss_dual = self.sampler.estimate_V_rho_closed(P_dual, n)

            losses_primal.append(loss_primal)
            losses_dual.append(loss_dual)

            K = self.primal_update(self.R1 + self.H, K, P_primal)
            print(f"Episode {e}/{epochs} - Return {loss_primal} \r", end='')
        return K, losses_primal, losses_dual

    def train_constrained(self, epochs: int, n: int) -> Tuple[Tensor, List[float], List[float]]:
        losses_primal, losses_dual = [], []
        K = zeros(self.da, self.ds).double()
        lmbda = zeros(1)
        for e in range(epochs):
            Gl, Rl = self.G1 + lmbda * self.G2, self.R1 + self.H + lmbda * self.R2

            P_primal = self.lqr.calculate_P(K, Gl, Rl)
            P_primal_unconstrained = self.lqr.calculate_P(K, self.G1, self.R1)
            P_dual = self.lqr.calculate_P(K, self.G2, self.R2)

            if e % 10 == 0:
                loss_primal = self.sampler.estimate_V_rho_closed(P_primal_unconstrained, n)
                loss_dual = self.sampler.estimate_V_rho_closed(P_dual, n)

                losses_primal.append(loss_primal)
                losses_dual.append(loss_dual)

            K = self.primal_update(Rl, K, P_primal)
            lmbda = self.dual_update(P_dual, lmbda, n)
            print(f"Episode {e}/{epochs} - Loss primal {loss_primal} - Loss dual {loss_dual} - Lambda {lmbda.detach().item()} \r", end='')
        return K, lmbda, losses_primal, losses_dual

    def evaluate_duality_gap(self, V_primal_opt: float, epochs: int, n: int):
        K = zeros(self.da, self.ds).double()
        lmbda = zeros(1)
        gaps = []
        for e in range(epochs):
            Gl, Rl = self.G1 + lmbda * self.G2, self.R1 + self.H + lmbda * self.R2

            P_primal = self.lqr.calculate_P(K, Gl, Rl)
            P_primal_unconstrained = self.lqr.calculate_P(K, self.G1, self.R1)
            P_dual = self.lqr.calculate_P(K, self.G2, self.R2)

            if e % 10 == 0:
                V_primal = self.sampler.estimate_V_rho_closed(P_primal_unconstrained, n)
                g = np.linalg.norm(V_primal_opt - V_primal)
                gaps.append(g)
                print(f"Episode {e}/{epochs} - Optm. Gap {g} - Lambda {lmbda.detach().item()} - V {V_primal} \r", end='')

            K = self.primal_update(Rl, K, P_primal)
            lmbda = self.dual_update(P_dual, lmbda, n)
        return gaps
