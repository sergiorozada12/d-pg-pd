import numpy as np
import torch
from scipy.linalg import solve_discrete_are


class Lqr:
    def __init__(
            self,
            A: np.ndarray,
            B: np.ndarray,
            G: np.ndarray,
            R: np.ndarray,
        ) -> None:
        P = torch.tensor(solve_discrete_are(A, B, G, R))
        self.K = - torch.inverse(R + B.T @ P @ B) @ B.T @ P @ A

    def pi(self, s: np.ndarray) -> np.ndarray:
        return s @ self.K.T
