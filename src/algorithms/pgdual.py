from typing import Callable, List, Tuple

from torch import Tensor, zeros, clip, concat, einsum, no_grad
from torch.linalg import lstsq
from torch import optim
import torch.nn as nn

from src.dynamics import RobotWorld


class Actor(nn.Module):
    def __init__(self, ds: int, da: int) -> None:
        super(Actor, self).__init__()
        self.linear = nn.Linear(ds, da, bias=False)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.001)

    def forward(self, state: Tensor) -> Tensor:
        return self.linear(state)


class Critic(nn.Module):
    def __init__(self, ds: int, da: int) -> None:
        super(Critic, self).__init__()
        self.ds = ds
        self.da = da
        self.linear = nn.Linear((ds + da)**2, 1, bias=False)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        state_action = concat([state, action], dim=1)
        X = einsum("bi,bj->bij", state_action, state_action).view(state.shape[0], (self.ds + self.da)**2)
        q = self.linear(X)
        return q


class LinearDual:
    def __init__(
        self,
        ds: int,
        da: int,
        env: RobotWorld,
        lr_actor: float,
        lr_dual: float,
        gamma: float,
        b: float,
        starting_pos_fn: Callable,
        primal_reward_fn : Callable,
        dual_reward_fn: Callable,
    ) -> None:
        self.env = env
        self.ds = ds
        self.da = da
        self.starting_pos_fn = starting_pos_fn
        self.primal_reward_fn = primal_reward_fn
        self.dual_reward_fn = dual_reward_fn

        self.gamma = gamma
        self.b = b
        self.lr_dual = lr_dual

        self.actor = Actor(ds, da).double()
        self.critic = Critic(ds, da).double()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

    def rollout_Q(self, s: Tensor, a: Tensor, n_roll: int, lmbda: Tensor) -> Tensor:
        self.env.reset(s.shape[0])
        self.env.s = s
        q = self.primal_reward_fn(self.env, a) + lmbda * self.dual_reward_fn(self.env, a)
        s = self.env.step(a)
        for i in range(1, n_roll):
            a = self.actor(s)
            q += (self.gamma ** i) * (self.primal_reward_fn(self.env, a) + lmbda * self.dual_reward_fn(self.env, a))
            s = self.env.step(a)
        return q.detach()

    def rollout_V_primal(self, n_roll: int, n_samples: int) -> Tensor:
        s = self.env.reset(n_samples)
        v = 0
        for i in range(n_roll):
            a = self.actor(s)
            v += (self.gamma ** i) * self.primal_reward_fn(self.env, a)
            s = self.env.step(a)
        return v.detach().mean()

    def rollout_V_dual(self, n_roll: int, n_samples: int) -> Tensor:
        s = self.env.reset(n_samples)
        v = 0
        for i in range(n_roll):
            a = self.actor(s)
            v += (self.gamma ** i) * self.dual_reward_fn(self.env, a)
            s = self.env.step(a)
        return v.detach().mean()

    def train(
        self,
        num_epochs: int,
        num_samples: int,
        num_rollout: int,
        num_rho: int,
        num_dual_update: int,
        ) -> Tuple[List[float], List[float]]:

        lmbda = zeros(1)
        loss_primal, loss_dual = [], []

        for epoch in range(num_epochs):
            s, a = self.starting_pos_fn(num_samples)
            s_a = concat([s, a], dim=1)
            Phi = einsum("bi,bj->bij", s_a, s_a).view(num_samples, (self.ds + self.da)**2)
            q = self.rollout_Q(s, a, num_rollout, lmbda)
            theta = lstsq(Phi, q, driver='gelsy').solution

            with no_grad():
                self.critic.linear.weight.copy_(theta)

            actor_loss = - self.critic(s, self.actor(s)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if epoch % num_dual_update == 0:
                V_primal = self.rollout_V_primal(num_rollout, num_rho)
                V_dual = self.rollout_V_dual(num_rollout, num_rho)
                lmbda = clip(lmbda - self.lr_dual * (V_dual - self.b), min=0)
                loss_primal.append(V_primal.detach().item())
                loss_dual.append(V_dual.detach().item())
                print(f"Epoch {epoch} - Primal {V_primal.item()} - Dual {V_dual.item()} - Lambda {lmbda.item()}")
        return loss_primal, loss_dual
