from typing import Tuple
import numpy as np
import torch
from torch import Tensor

from src.config import Config


class RobotWorld:
    def __init__(self) -> None:
        self.s_r = torch.tensor(Config.s_r)
        self.rng = np.random.default_rng()
        self.A, self.B = self.generate_dynamics(Config.time_step)
        self.boundaries = torch.tensor(
            [
                Config.x_range,
                Config.y_range,
                Config.vx_range,
                Config.vy_range,
            ]
        ).double()

    def generate_initial_point(self, x_range, y_range, vx_range, vy_range) -> np.ndarray:
        s_0 = torch.tensor([
            self.rng.uniform(*x_range),
            self.rng.uniform(*y_range),
            self.rng.uniform(*vx_range),
            self.rng.uniform(*vy_range),
        ]).double()
        return (s_0 - self.s_r)

    def generate_dynamics(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        A = torch.tensor(
            [
                [1., 0, dt, 0],
                [0, 1., 0, dt],
                [0, 0, 1., 0],
                [0, 0, 0, 1.],
            ]
        ).double()

        B = torch.tensor(
            [
                [dt**2 / 2, 0.0],
                [0.0, dt**2 / 2],
                [dt, 0.0],
                [0.0, dt],
            ]
        ).double()

        return A, B

    def reset(self):
        self.s = self.generate_initial_point(
            Config.x_range,
            Config.y_range,
            Config.vx_range,
            Config.vy_range
        )
        return self.s

    def generate_noise(self, size: int) -> Tensor:
        return torch.tensor(
            self.rng.normal(
            scale=np.array(
                [
                    Config.noise_pos,
                    Config.noise_pos,
                    Config.noise_vel,
                    Config.noise_vel
                ]
            ) * Config.time_step,
            size=size,
        )
        )

    def step(self, u: np.ndarray) -> np.ndarray:
        noise = self.generate_noise(self.s.shape[0])
        self.s_noiseless = self.s @ self.A.T + u @ self.B.T 
        self.s = self.s_noiseless + noise
        #self.s = torch.clip(self.s, self.boundaries[:, 0], self.boundaries[:, 1])
        return self.s
