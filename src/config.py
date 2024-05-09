from typing import Tuple


class Config:
    duration: float = 10.0
    time_step: float = 0.05
    x_range: Tuple[int, int] = [-10, 10]
    y_range: Tuple[int, int] = [-10, 10]
    vx_range: Tuple[int, int] = [-1, 1]
    vy_range: Tuple[int, int] = [-1, 1]
    noise_pos: float = 0.1
    noise_vel: float = 0.1
