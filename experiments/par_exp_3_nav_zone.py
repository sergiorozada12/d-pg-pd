import os
import sys
import random
import numpy as np
import torch
import multiprocessing as mp

d = os.getcwd()
p = os.path.dirname(d)

sys.path.append(p)

from src.dynamics import RobotWorld
from src.algorithms.addpgpd_sampled import ADpgpdSampled
from src.algorithms.pgdual import LinearDual

torch.set_num_threads(1)

ds = 4
da = 2

tau = 0.01
gamma = 0.95
b = - 200.0

G = - torch.tensor([
    [1.0, 0, 0, 0],
    [0, 1.0, 0, 0],
    [0, 0, 0.1, 0],
    [0, 0, 0, 0.1]
]).double()

R =  - torch.tensor([
    [0.1, 0],
    [0, 0.1],
]).double()

def primal_reward_fn(env, a):
    return ((env.s @ G) * env.s).sum(dim=1) + ((a @ R) * a).sum(dim=1)

def primal_reward_reg_fn(env, a):
    return -(tau / 2) * (a * a).sum(dim=1)

def dual_reward_fn(env, a):
    return 100 * (env.s[:, 0].clip(max=1.0) + env.s[:, 1].clip(max=1.0) - 2)

def starting_pos_fn(nsamples):
    rng = np.random.default_rng()

    s = torch.tensor(rng.uniform(
        low=[40, 40, -10, -10],
        high= [50, 50, 10, 10],
        size=[nsamples, 4],
    )).double()

    a = torch.tensor(rng.uniform(
        low=[-10, -10],
        high= [10, 10],
        size=[nsamples, 2],
    )).double()

    return s, a


def run_experiment_pgpd(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    epochs = 50_000
    n_pe = 100
    n_rho = 2_000
    n_roll = 200

    alpha = 1.0
    eta = 0.00005

    env = RobotWorld(range_pos=[40, 50], range_vel=[-.1, .1])
    dpgpd = ADpgpdSampled(ds, da, env, eta, tau, gamma, b, alpha, primal_reward_fn, primal_reward_reg_fn, dual_reward_fn, starting_pos_fn)

    K, lmbda, losses_primal, losses_dual = dpgpd.train_constrained(epochs, n_pe, n_rho, n_roll)
    return K, lmbda, losses_primal, losses_dual

def run_experiment_pgdual(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n_epochs = 50_000
    n_samples = 100
    n_rollout = 200
    n_rho = 2_000

    n_dual_update = 10
    lr_actor = 1e-4
    lr_dual = 1e-3

    env = RobotWorld(range_pos=[40, 50], range_vel=[-.1, .1])

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

    np.save("../results/nav_dpgpd_zone_k.npy", Ks)
    np.save("../results/nav_dpgpd_zone_l.npy", lambdas)
    np.save("../results/nav_dpgpd_zone_ploss.npy", losses_primals)
    np.save("../results/nav_dpgpd_zone_dloss.npy", losses_duals)

    losses_primals, losses_duals = zip(*results_pgdual)
    avg_losses_primal = np.mean(losses_primals, axis=0)
    avg_losses_dual = np.mean(losses_duals, axis=0)

    np.save("../results/nav_pgdual_zone_ploss.npy", losses_primals)
    np.save("../results/nav_pgdual_zone_dloss.npy", losses_duals)
