from typing import Callable, List, Tuple

from torch import Tensor, cat, zeros, clip
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from src.dynamics import RobotWorld


class MLPActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int =256) -> None:
        super(MLPActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)

    def forward(self, state: Tensor) -> Tensor:
        x = F.relu(self.fc1(state))
        action = self.fc2(x)
        return action


class MLPCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int=256) -> None:
        super(MLPCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.1)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        x = cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        q_value = self.fc2(x)
        return q_value


class Nnpgpd:
    def __init__(
        self,
        ds: int,
        da: int,
        env: RobotWorld,
        hidden_dims: int,
        lr_actor: float,
        lr_critic: float,
        lr_dual: float,
        gamma: float,
        b: float,
        starting_pos_fn: Callable,
        primal_reward_fn : Callable,
        dual_reward_fn: Callable,
    ) -> None:
        self.env = env
        self.starting_pos_fn = starting_pos_fn
        self.primal_reward_fn = primal_reward_fn
        self.dual_reward_fn = dual_reward_fn

        self.gamma = gamma
        self.b = b
        self.lr_dual = lr_dual

        self.actor = MLPActor(ds, da, hidden_dims).double()
        self.critic = MLPCritic(ds, da, hidden_dims).double()

        self.actor_optimizer = optim.SGD(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.SGD(self.critic.parameters(), lr=lr_critic)

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
        num_episodes: int,
        num_samples: int,
        num_rollout: int,
        num_rho: int,
        num_init: int=10,
        ) -> Tuple[List[float], List[float]]:
        lmbda = zeros(1)
        loss_primal, loss_dual = [], []

        batch_size = 32
        state, action = self.starting_pos_fn(10_000)
        target_q = self.rollout_Q(state, action, num_rollout, lmbda)
        num_batches = (state.shape[0] + batch_size - 1) // batch_size
        for _ in range(num_init):
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size

                batch_state = state[start_idx:end_idx]
                batch_action = action[start_idx:end_idx]
                batch_q = target_q[start_idx:end_idx]

                q_value = self.critic(batch_state.detach(), batch_action.detach()).squeeze()
                critic_loss = F.mse_loss(q_value, batch_q.detach())
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
        print(f"Initial CLoss - {critic_loss.detach().item()}")

        for epoch in range(num_epochs):
            for episode in range(num_episodes):
                state, action = self.starting_pos_fn(num_samples)
                target_q = self.rollout_Q(state, action, num_rollout, lmbda)
                q_value = self.critic(state.detach(), action.detach()).squeeze()

                critic_loss = F.mse_loss(q_value, target_q.detach())
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                actor_loss = - self.critic(state, self.actor(state)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

            V_dual = self.rollout_V_dual(num_rollout, num_rho)
            lmbda = clip(lmbda - self.lr_dual * (V_dual - self.b), min=0)
            if epoch % 10 == 0:
                V_primal = self.rollout_V_primal(num_rollout, num_rho)
                loss_primal.append(V_primal.detach().item())
                loss_dual.append(V_dual.detach().item())
                print(f"Epoch {epoch} - Primal {V_primal.item()} - Dual {V_dual.item()} - Lambda {lmbda.item()}")
        return loss_primal, loss_dual
