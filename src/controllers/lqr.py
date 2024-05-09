import numpy as np
from scipy.linalg import solve_discrete_are


class Lqr:
    def __init__(
            self,
            A: np.ndarray,
            B: np.ndarray,
            Q: np.ndarray,
            R: np.ndarray,
            s_r: np.ndarray
        ) -> None:
        P = solve_discrete_are(A, B, Q, R)
        self.K = np.dot(np.dot(np.linalg.inv(R), B.T), P)
        self.s_r = s_r

    def pi(self, s: np.ndarray) -> np.ndarray:
        return (self.s_r - s) @ self.K.T
