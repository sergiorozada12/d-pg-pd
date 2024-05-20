from typing import Tuple, List


class Config:
    duration: float = 10.0
    time_step: float = 0.05
    x_range: Tuple[int, int] = [-10, 10]
    y_range: Tuple[int, int] = [-10, 10]
    vx_range: Tuple[int, int] = [-.1, .1]
    vy_range: Tuple[int, int] = [-.1, .1]
    noise_pos: float = 1.0
    noise_vel: float = 1.0
    s_r: List[int] = [0, 0, 0, 0]
