import numpy as np
from algos import compute_penalty


class OptionMDP():
    def __init__(self, H=20, K=100, d=12, tau=0.5, decimal=1,
                 normalize_reward=True):
        x_min, x_max = 80, 140
        self.x2s = lambda x: round((x - x_min) * 10**decimal)
        self.s2x = lambda s: s / 10**decimal + x_min

        self.N_S = (x_max - x_min) * 10**decimal + 2
        self.N_A = 2
        self.d = d
        self.H = H
        c_u, c_d = 1.02, 0.98

        phi = np.zeros((self.N_S, d))
        P = np.zeros((self.N_S, self.N_A, self.N_S))
        r = np.zeros((self.N_S, self.N_A))
        g = np.zeros((self.N_S, self.N_A))

        # 安全带阈值（示例）
        safe_lo, safe_hi = K - 10, K + 10

        for s in range(self.N_S):
            if s == self.N_S - 1:
                P[s, :, s] = 1
                phi[s, :] = 1
                r[s, :] = 0
                g[s, :] = 1
            else:
                # 动作0：继续；动作1：行权吸收
                P[s, 0, min(self.x2s(c_u * self.s2x(s)), self.N_S - 2)] = tau
                P[s, 0, max(self.x2s(c_d * self.s2x(s)), 0)] = 1 - tau
                P[s, 1, -1] = 1

                # 奖励：看跌/看涨你原来怎么设就怎么用；这里沿用 r(s,1)=max(0, K - x)
                r[s, 1] = max(0, K - self.s2x(s))
                r[s, 0] = 0

                # 约束 utility：价格落在 [K-10,K+10] 视为“合规=1”，否则 0
                in_band = (safe_lo <= self.s2x(s) <= safe_hi)
                g[s, 0] = 1.0 if in_band else 0.0     # 继续时的合规性
                g[s, 1] = 1.0                         # 行权视为合规（可按需改）

                # 线性特征：帽函数基
                radius = (x_max - x_min) / (d - 1)
                x_0 = np.linspace(x_min, x_max, d, endpoint=True)
                phi[s] = np.maximum(1 - np.abs(self.s2x(s) - x_0) / radius, 0)

        self.phi = phi / np.clip(phi.sum(-1, keepdims=True), 1e-12, None)
        self.P = P
        self.r = r
        self.g = g

        if normalize_reward:
            self.r_scale = max(self.r.max(), 1.0) * 10.0
            self.r /= self.r_scale

        self.initial_state_dist = np.zeros(self.N_S)
        lo, hi = self.x2s(K - 5), self.x2s(K + 5)
        self.initial_state_dist[lo:hi] = 1
        self.initial_state_dist /= self.initial_state_dist.sum()

    def reset(self):
        self.h = 0
        self.state = np.random.choice(self.N_S, p=self.initial_state_dist)
        return self.state

    def step(self, action):
        self.h += 1
        reward = self.r[self.state, action]
        util   = self.g[self.state, action]
        p = self.P[self.state, action]
        p /= p.sum()
        self.state = np.random.choice(self.N_S, p=p)
        return self.state, reward, util, (self.h >= self.H)


class OptionLinearPolicy():
    def __init__(self, mdp, weights_r, weights_g, Lambda_inv, gamma0, beta, b):
        self.mdp = mdp
        self.wr = weights_r
        self.wg = weights_g
        self.beta = float(beta)
        self.b = float(b)
        self.state_penalty = np.array([
            compute_penalty(self.mdp.phi[s], Lambda_inv, gamma0)
            for s in range(self.mdp.N_S)
        ])

    def _Q_r(self, s, h):
        # 悲观奖励 Q
        Q0 = float(self.mdp.phi[s] @ self.wr[h] - self.state_penalty[s])
        Q1 = float(self.mdp.r[s, 1])
        return (Q0, Q1)

    def _Q_g(self, s, h):
        # 乐观约束 Q
        Q0 = float(self.mdp.phi[s] @ self.wg[h] + self.state_penalty[s])
        Q1 = float(self.mdp.g[s, 1])
        return (Q0, Q1)

    def sample(self, state, h):
        Qr0, Qr1 = self._Q_r(state, h)
        Qg0, Qg1 = self._Q_g(state, h)
        # 整流优化： max_a Qr - beta * (b - Qg)_+
        obj0 = Qr0 - self.beta * max(self.b - Qg0, 0.0)
        obj1 = Qr1 - self.beta * max(self.b - Qg1, 0.0)
        return 0 if obj0 >= obj1 else 1

    def V(self, state, h):
        Qr0, Qr1 = self._Q_r(state, h)
        Qg0, Qg1 = self._Q_g(state, h)
        obj0 = Qr0 - self.beta * max(self.b - Qg0, 0.0)
        obj1 = Qr1 - self.beta * max(self.b - Qg1, 0.0)
        # 这里返回的是“整流后的目标值”，若你想看纯 V_r/V_g 可单独返回
        return max(obj0, obj1)

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
