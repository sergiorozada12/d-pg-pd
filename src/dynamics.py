import numpy as np

from typing import Tuple

from src.config import Config


class RobotWorld:
    def __init__(self) -> None:
        self.rng = np.random.default_rng()
        self.A, self.B = self.generate_dynamics(Config.time_step)
        self.boundaries = np.array(
            [
                Config.x_range,
                Config.y_range,
                Config.vx_range,
                Config.vy_range,
            ]
        )

    def generate_initial_point(self, x_range, y_range) -> np.ndarray:
        return np.array([
            self.rng.uniform(*x_range),
            self.rng.uniform(*y_range),
            0,
            0,
        ])

    def generate_dynamics(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        A = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        B = np.array(
            [
                [dt**2 / 2, 0.0],
                [0.0, dt**2 / 2],
                [dt, 0.0],
                [0.0, dt],
            ]
        )

        return A, B

    def reset(self):
        self.s = self.generate_initial_point(Config.x_range, Config.y_range)
        return self.s

    def step(self, u: np.ndarray) -> np.ndarray:
        noise = self.rng.normal(
            scale=np.array(
                [
                    Config.noise_pos,
                    Config.noise_pos,
                    Config.noise_vel,
                    Config.noise_vel
                ]
            ) * Config.time_step,
            size=self.s.shape,
        )
        self.s = self.s @ self.A.T + u @ self.B.T + noise
        self.s = np.clip(self.s, self.boundaries[:, 0], self.boundaries[:, 1])
        return self.s
