from typing import Tuple
import numpy as np
import torch
from torch import Tensor

from src.config import Config


class RobotWorld:
    def __init__(self, range_pos, range_vel) -> None:
        self.s_r = torch.tensor(Config.s_r)
        self.rng = np.random.default_rng()
        self.range_pos = range_pos
        self.range_vel = range_vel
        self.A, self.B = self.generate_dynamics(Config.time_step)

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

    def reset(self, n_samples: int=1):
        s = np.stack([
            self.rng.uniform(self.range_pos[0], self.range_pos[1], n_samples),
            self.rng.uniform(self.range_pos[0], self.range_pos[1], n_samples),
            self.rng.uniform(self.range_vel[0], self.range_vel[1], n_samples),
            self.rng.uniform(self.range_vel[0], self.range_vel[1], n_samples),
        ])
        self.s = torch.from_numpy(s).double().T
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
        noise = self.generate_noise(self.s.shape)
        self.s_noiseless = self.s @ self.A.T + u @ self.B.T 
        self.s = self.s_noiseless + noise
        return self.s


class InventoryControl():
    def __init__(self, range_assets, range_demand, range_acq) -> None:
        self.rng = np.random.default_rng()
        self.range_assets = range_assets
        self.range_demand = range_demand
        self.range_acq = range_acq

    def reset(self, n_samples: int=1):
        self.demand = self.generate_noise([n_samples, 4])

        s = np.concatenate([
            self.rng.uniform(self.range_assets[0], self.range_assets[1], (n_samples, 4)),
            self.rng.uniform(self.range_demand[0], self.range_demand[1], (n_samples, 4)),
            self.rng.uniform(self.range_acq[0], self.range_acq[1], (n_samples, 4)),
        ], axis=1)
        self.s = torch.from_numpy(s).double()
        return self.s

    def generate_noise(self, size: int) -> Tensor:
        return torch.tensor(
            self.rng.normal(
                loc=10,
                scale=np.array(
                    [
                        Config.noise_asset,
                        Config.noise_asset,
                        Config.noise_asset,
                        Config.noise_asset,
                    ]
                ),
                size=size,
        )
        ).clip(min=0)

    def step(self, u: np.ndarray) -> np.ndarray:
        u = u.clip(min=0)

        self.s[:, :4] = torch.clip(self.s[:, :4] + u - self.demand, min=0.0)
        self.s[:, 4:8] = self.demand
        self.s[:, 8:] = u

        self.demand = self.generate_noise([self.s.shape[0], 4])

        return self.s
