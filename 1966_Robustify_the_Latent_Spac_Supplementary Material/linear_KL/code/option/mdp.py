import numpy as np


class OptionMDP():
    def __init__(self, H=20, K=100, d=12, tau=0.5, coef=1.02, decimal=1, normalize_reward=True):
        x_min, x_max = 80, 140
        self.x2s = lambda x: round((x - x_min) * 10**decimal)
        self.s2x = lambda s: s / 10**decimal + x_min

        self.N_S = (x_max - x_min) * 10**decimal + 2  # [x_min, x_max] + absorb state
        self.N_A = 2  # exercise or not
        self.d = d  # linear function dimension
        self.H = H  # time horizon
        c_u = 1.02
        c_d = 0.98

        phi = np.zeros((self.N_S, d))
        P = np.zeros((self.N_S, self.N_A, self.N_S))
        r = np.zeros((self.N_S, self.N_A))
        for s in range(self.N_S):
            if s == self.N_S - 1:
                P[s, :, s] = 1
                phi[s, :] = 1  # meaningless
            else:
                P[s, 0, min(self.x2s(c_u * self.s2x(s)), self.N_S - 2)] = tau
                P[s, 0, max(self.x2s(c_d * self.s2x(s)), 0)] = 1 - tau
                P[s, 1, -1] = 1
                r[s, 1] = max(0, K - self.s2x(s))
                # construct the feature
                # radius = (x_max - x_min) / d / 4
                # x_0 = np.linspace(self.s2x(1), x_max, d, endpoint=True)
                # phi[s] = np.exp(-(self.s2x(s) - x_0)**2 / (2 * radius**2))
                radius = (x_max - x_min) / (d - 1)
                x_0 = np.linspace(x_min, x_max, d, endpoint=True)
                phi[s] = np.maximum(1 - np.abs(self.s2x(s) - x_0) / radius, 0)
                
        self.phi = phi / phi.sum(-1, keepdims=True)
        self.P = P
        self.r = r
        if normalize_reward:
            self.r_scale = self.r.max() * 10
            self.r /= self.r_scale

        self.initial_state_dist = np.zeros(self.N_S)
        self.initial_state_dist[self.x2s(K - 5):self.x2s(K + 5)] = 1
        self.initial_state_dist /= self.initial_state_dist.sum()  # uniform reset

    def reset(self):
        self.h = 0
        self.state = np.random.choice(self.N_S, p=self.initial_state_dist)
        return self.state

    def step(self, action):
        self.h += 1
        reward = self.r[self.state, action]
        p = self.P[self.state, action]
        p /= p.sum()
        self.state = np.random.choice(self.N_S, p=p)
        return self.state, reward, self.h >= self.H


class OptionLinearPolicy():
    def __init__(self, mdp, omegas):
        self.mdp = mdp
        self.omegas = omegas

    def sample(self, state, h):
        omega = self.omegas[h]
        Q_0 = np.dot(self.mdp.phi[state], omega)
        Q_1 = self.mdp.r[state, 1]
        return 0 if Q_0 > Q_1 else 1

    def set_policy(self, omegas):
        self.omegas = omegas

    def V(self, state, h):
        omega = self.omegas[h]
        Q_0 = np.dot(self.mdp.phi[state], omega)
        Q_1 = self.mdp.r[state, 1]
        return max([Q_0, Q_1])

class FixedPolicy():

    def __init__(self, N_S, N_A):
        self.N_S = N_S
        self.N_A = N_A
        self.policy = np.zeros((self.N_S, self.N_A))
        self.policy[:, 0] = 1
        self.policy /= self.policy.sum(-1, keepdims=True)

    def sample(self, state, h):
        return np.random.choice(self.N_A, p=self.policy[state])

    def set_policy(self, mu):
        self.policy = mu
