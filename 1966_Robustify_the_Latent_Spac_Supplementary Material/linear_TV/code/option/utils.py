import numpy as np
import scipy.optimize as opt

LOG_EPS = 1e-20
BETA_MIN = 1e-5
BETA_MAX = 1e3


def generate_traj(mdp, policy):
    traj = []
    state = mdp.reset()
    done = False
    while not done:
        action = policy.sample(state, mdp.h)
        next_state, reward, util, done = mdp.step(action)
        traj.append((state, action, reward, util, next_state))
        state = next_state
    return traj


def get_optimal_value(mdp):
    V = np.zeros((mdp.N_S,))
    for h in reversed(range(mdp.H)):
        Q = mdp.r + np.sum(mdp.P * V, axis=-1)
        V = Q.max(axis=-1)
    return np.dot(mdp.initial_state_dist, V)


def get_optimal_robust_value(mdp, rho):
    from itertools import product
    V = np.zeros((mdp.N_S,))
    for h in reversed(range(mdp.H)):
        Q = np.zeros((mdp.N_S, mdp.N_A))
        for s, a in product(range(mdp.N_S), range(mdp.N_A)):
            Q[s, a] = mdp.r[s, a] - opt.fminbound(
                lambda b: b * np.log(np.dot(mdp.P[s, a], np.exp(-(V - V.min()) / b)) + LOG_EPS) + b * rho - V.min(),
                BETA_MIN,
                BETA_MAX,
                full_output=True)[1]
        V = Q.max(axis=-1)
    return np.dot(mdp.initial_state_dist, V)
