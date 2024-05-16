from torch import tensor, eye, kron, inverse, cat, Tensor
from scipy.linalg import solve_discrete_are


class Lqr:
    def __init__(
            self,
            A: Tensor,
            B: Tensor,
            gamma: float,
        ) -> None:
        self.A = A
        self.B = B
        self.gamma = gamma

    def calculate_optimal_P(self, G: Tensor, R: Tensor) -> Tensor:
        A_hat = (self.gamma ** (1 / 2)) * self.A
        B_hat = (self.gamma ** (1 / 2)) * self.B
        return tensor(solve_discrete_are(A_hat, B_hat, G, R))

    def calculate_P(self, K: Tensor, G: Tensor, R: Tensor) -> Tensor:
        ds = self.A.shape[0]
        X = eye(ds ** 2) - self.gamma * kron((self.A + self.B @ K).T, (self.A + self.B @ K).T )
        b = (G + K.T @ R @ K).flatten()
        P = (inverse(X) @ b).reshape(ds, ds)
        return P

    def calculate_H(self, P: Tensor, G: Tensor, R: Tensor) -> Tensor:
        H_11 = G + self.gamma * self.A.T @ P @ self.A
        H_12 = self.gamma * self.A.T @ P @ self.B
        H_21 = self.gamma * self.B.T @ P @ self.A
        H_22 = R + self.gamma * self.B.T @ P @ self.B

        top = cat((H_11, H_12), dim=1)
        bot = cat((H_21, H_22), dim=1)

        H = cat((top, bot), dim=0)
        return H
    
    def calculate_optimal_K(self, R: Tensor, P: Tensor) -> Tensor:
        return - self.gamma * inverse(R + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A

    @staticmethod
    def calculate_V(s: Tensor, P: Tensor) -> float:
        return s @ P @ s

    @staticmethod
    def calculate_Q(s: Tensor, a: Tensor, H: Tensor) -> float:
        return cat((s, a), dim=0) @ H @ cat((s, a), dim=0) 
