import os
import sys
import random
import numpy as np
import torch
import multiprocessing as mp

d = os.getcwd()
p = os.path.dirname(d)

sys.path.append(p)

from src.dynamics import BurgersDynamics
from src.algorithms.addpgpd_sampled import ADpgpdSampled
from src.algorithms.pgdual import LinearDual

torch.set_num_threads(1)

ds = 10
da = 10

tau = 0.01
gamma = 0.9
alpha = 1.0

viscosity = 0.1
dt = 0.01
dx = 1.0 / (ds - 1)

b = -20

G = - torch.diag(torch.tensor([1.0] * ds)).double()
R =  - torch.diag(torch.tensor([0.1] * da)).double() * (tau / 2)
C = - torch.eye(da).double()

def primal_reward_fn(env, a):
    return ((env.u @ G) * env.u).sum(dim=1) + ((a @ R) * a).sum(dim=1)

def primal_reward_reg_fn(env, a):
    return -(tau / 2) * (a * a).sum(dim=1)

def dual_reward_fn(env, a):
    return (a.abs() @ C).sum(dim=1)

def starting_pos_fn(n_samples):
    rng = np.random.default_rng()

    x = torch.linspace(0, 1, ds).double()
    u = torch.sin(np.pi * x).repeat(n_samples, 1)
    noise = torch.normal(0, 0.01, size=u.shape).double()
    u += noise

    a = torch.tensor(rng.uniform(
        low=[-1] * da,
        high=[1] * da,
        size=[n_samples, da],
    )).double()

    return u, a

def run_experiment_pgpd(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    epochs = 10_000
    n_pe = 100
    n_rho = 1_000
    n_roll = 100

    gamma = 0.9
    eta = 0.001

    env = BurgersDynamics(ds, viscosity, dt, dx)
    dpgpd = ADpgpdSampled(ds, da, env, eta, tau, gamma, b, alpha, primal_reward_fn, primal_reward_reg_fn, dual_reward_fn, starting_pos_fn)

    K, lmbda, losses_primal, losses_dual = dpgpd.train_constrained(epochs, n_pe, n_rho, n_roll)
    return K, lmbda, losses_primal, losses_dual

def run_experiment_pgdual(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n_epochs = 10_000
    n_samples = 100
    n_rollout = 100
    n_rho = 1_000

    n_dual_update = 10
    lr_actor = 1e-3
    lr_dual = 1e-2

    env = BurgersDynamics(ds, viscosity, dt, dx)

    ld = LinearDual(ds, da, env, lr_actor, lr_dual, gamma, b, starting_pos_fn, primal_reward_fn, dual_reward_fn)
    losses_primal, losses_dual = ld.train(n_epochs, n_samples, n_rollout, n_rho, n_dual_update)
    return losses_primal, losses_dual


def run_parallel_pgpd(n_experiments):
    seeds = [np.random.randint(0, 1000000) for _ in range(n_experiments)]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(run_experiment_pgpd, seeds)
    return results

def run_parallel_pgdual(n_experiments):
    seeds = [np.random.randint(0, 1000000) for _ in range(n_experiments)]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(run_experiment_pgdual, seeds)
    return results

n_experiments = 50
results_pgpd = run_parallel_pgpd(n_experiments)
results_pgdual = run_parallel_pgdual(n_experiments)

if __name__ == "__main__":
    Ks, lambdas, losses_primals, losses_duals = zip(*results_pgpd)
    avg_K = np.mean(Ks, axis=0)
    avg_lambda = np.mean(lambdas, axis=0)
    avg_losses_primal = np.mean(losses_primals, axis=0)
    avg_losses_dual = np.mean(losses_duals, axis=0)

    np.save("../results/fluid_dpgpd_k.npy", Ks)
    np.save("../results/fluid_dpgpd_l.npy", lambdas)
    np.save("../results/fluid_dpgpd_ploss.npy", losses_primals)
    np.save("../results/fluid_dpgpd_dloss.npy", losses_duals)

    losses_primals, losses_duals = zip(*results_pgdual)
    avg_losses_primal = np.mean(losses_primals, axis=0)
    avg_losses_dual = np.mean(losses_duals, axis=0)

    np.save("../results/fluid_pgdual_ploss.npy", losses_primals)
    np.save("../results/fluid_pgdual_dloss.npy", losses_duals)
