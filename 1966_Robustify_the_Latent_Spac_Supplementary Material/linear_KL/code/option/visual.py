# -*- coding: utf-8 -*-
# Reproduce Fig. 2 in:
# Ma et al., "Distributionally Robust Offline RL with Linear Function Approximation" (American Option)
# 环境与设置见论文正文与附录C。

import numpy as np
import time
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ---------- 数值常量 ----------
LOG_EPS   = 1e-12      # 防止 log(0)
BETA_MIN  = 1e-6       # 对偶变量 β 的搜索下界
BETA_MAX  = 50.0       # 对偶变量 β 的搜索上界（ρ→0 时 β*→∞，用有限上界近似）
TOL       = 1e-8       # 一维搜索容差

# ---------- 一维有界最小化：黄金分割法（自包含，无需 SciPy） ----------
def fminbound(func, a, b, tol=TOL, max_iter=200):
    """在 [a,b] 上用黄金分割法最小化 func，返回 (x_min, f_min)。"""
    invphi = (math.sqrt(5) - 1) / 2  # 1/phi
    invphi2 = (3 - math.sqrt(5)) / 2  # 1/phi^2
    (a, b) = (min(a, b), max(a, b))
    h = b - a
    if h <= tol: 
        x = (a + b) / 2.0
        return x, func(x)
    # 需的迭代步数
    n = int(math.ceil(math.log(tol / h) / math.log(invphi)))
    c = a + invphi2 * h
    d = a + invphi * h
    yc = func(c)
    yd = func(d)
    for _ in range(n):
        if yc < yd:
            b, d, yd = d, c, yc
            h = invphi * h
            c = a + invphi2 * h
            yc = func(c)
        else:
            a, c, yc = c, d, yd
            h = invphi * h
            d = a + invphi * h
            yd = func(d)
    x = (a + b) / 2.0
    return x, func(x)

# ---------- DRO 对偶：带 value shift 的 g(β) ----------
def g_with_d(beta, psi, v, rho):
    """
    论文中 KL-DRO 的对偶目标（带 value shift）：
    g(β) = β * ρ + β * log( 1 + E_ψ[exp(- (v - min(v)) / β) - 1] ) - min(v)
    其中 ψ 是 d-rectangular 下的第 i 个因子的权重（这里已折叠为对后继状态的权重向量）。
    """
    v_min = v.min()
    v_span = v.max() - v_min
    # 数值稳定：先平移 v，再用 expm1；y ∈ [-1, 0]
    y = np.dot(psi, np.expm1(-(v - v_min) / max(beta, 1e-12)))
    # 裁剪避免 log(1 + y) 的参数落在 (-∞, -1]
    y = np.clip(y, np.expm1(-v_span / max(beta, 1e-12)), 0.0)
    return beta * rho + beta * np.log1p(y + LOG_EPS) - v_min

# ---------- American Option 环境 ----------
class OptionMDP:
    def __init__(self, H=20, K=100, d=61, tau=0.5, decimal=1, normalize_reward=True):
        # 价格离散
        x_min, x_max = 80, 140
        self.x2s = lambda x: round((x - x_min) * 10**decimal)
        self.s2x = lambda s: s / 10**decimal + x_min

        self.N_S = (x_max - x_min) * 10**decimal + 2  # 含 1 个吸收态
        self.N_A = 2  # 0=继续持有, 1=行权（终止）
        self.d = d
        self.H = H
        c_u, c_d = 1.02, 0.98  # 价格上下因子

        # 特征、转移、奖励
        phi = np.zeros((self.N_S, d))
        P = np.zeros((self.N_S, self.N_A, self.N_S))
        r = np.zeros((self.N_S, self.N_A))
        for s in range(self.N_S):
            if s == self.N_S - 1:
                # 吸收态
                P[s, :, s] = 1.0
                phi[s, :] = 1.0
            else:
                # 动作0：以 tau 上涨，否则下跌；保持在区间内
                su = min(self.x2s(c_u * self.s2x(s)), self.N_S - 2)
                sd = max(self.x2s(c_d * self.s2x(s)), 0)
                P[s, 0, su] = tau
                P[s, 0, sd] = 1 - tau
                # 动作1：行权->吸收态
                P[s, 1, -1] = 1.0
                # 行权收益（看跌期权）
                r[s, 1] = max(0.0, K - self.s2x(s))
                # 分段线性（帽函数）特征
                radius = (x_max - x_min) / (d - 1)
                x_anchors = np.linspace(x_min, x_max, d, endpoint=True)
                phi[s] = np.maximum(1.0 - np.abs(self.s2x(s) - x_anchors) / radius, 0.0)

        # 特征按行归一化
        self.phi = phi / phi.sum(-1, keepdims=True)
        self.P = P
        self.r = r

        # 奖励缩放（和论文示例代码保持一致，便于数值稳定）
        if normalize_reward:
            self.r_scale = self.r.max() * 10.0 if self.r.max() > 0 else 1.0
            self.r = self.r / self.r_scale
        else:
            self.r_scale = 1.0

        # 初始分布：K±5 附近均匀
        self.initial_state_dist = np.zeros(self.N_S)
        self.initial_state_dist[self.x2s(K - 5): self.x2s(K + 5)] = 1.0
        self.initial_state_dist /= self.initial_state_dist.sum()

        self.reset()

    def reset(self):
        self.h = 0
        self.state = np.random.choice(self.N_S, p=self.initial_state_dist)
        return self.state

    def step(self, action):
        self.h += 1
        reward = self.r[self.state, action]
        p = self.P[self.state, action].copy()
        p /= p.sum()
        self.state = np.random.choice(self.N_S, p=p)
        done = self.h >= self.H
        return self.state, reward, done

# 固定采集策略（始终选择动作0）
class FixedPolicy:
    def __init__(self, N_S, N_A):
        self.N_S, self.N_A = N_S, N_A
        self.policy = np.zeros((N_S, N_A))
        self.policy[:, 0] = 1.0

    def sample(self, state, h):
        return 0

# 线性策略（用 DRVI 学到的 ω_h）
class OptionLinearPolicy:
    def __init__(self, mdp, omegas):
        self.mdp = mdp
        self.omegas = omegas  # list of (d,) for each h

    def sample(self, state, h):
        omega = self.omegas[h]
        Q0 = float(np.dot(self.mdp.phi[state], omega))
        Q1 = float(self.mdp.r[state, 1])
        return 0 if Q0 > Q1 else 1

    def V(self, state, h):
        omega = self.omegas[h]
        return max(float(np.dot(self.mdp.phi[state], omega)),
                   float(self.mdp.r[state, 1]))

# 生成一条轨迹（长度 H），记录 (s, a, r, s')
def generate_traj(mdp, policy):
    traj = []
    s = mdp.reset()
    for h in range(mdp.H):
        a = policy.sample(s, h)
        s_next, r, done = mdp.step(a)
        traj.append((s, a, float(r), s_next))
        s = s_next
        if done:
            # 吸收态后仍会继续迭代，但我们的构造已是吸收转移，不影响
            pass
    return traj

# ---------- 经验线性模型 + DRVI（d-rectangular, KL, value shift） ----------
def DRVI(mdp: OptionMDP, dataset, rho: float):
    K = len(dataset)
    # 统计矩阵（所有阶段共享）
    Lambda = np.identity(mdp.d)
    psi_hat = np.zeros((mdp.d, mdp.N_S))
    theta_hat = np.zeros((mdp.d,))

    # 按论文实现：遍历数据集中每条轨迹的每一步
    for k in range(K):
        for h in range(mdp.H):
            s, a, r, next_s = dataset[k][h]
            phi_s = mdp.phi[s]               # (d,)
            psi_hat[:, next_s] += phi_s
            theta_hat += phi_s * r
            Lambda += np.outer(phi_s, phi_s) # φ φ^T

    Lambda_inv = np.linalg.inv(Lambda)
    psi   = Lambda_inv @ psi_hat            # (d, S)
    theta = Lambda_inv @ theta_hat          # (d,)

    # 反向迭代得到每个阶段的 ω
    omegas = []
    omega = np.zeros((mdp.d,))
    next_v = np.zeros((mdp.N_S,))

    for h in reversed(range(mdp.H)):
        # 先计算下一阶段价值 next_v(s')
        for s in range(mdp.N_S):
            if h < mdp.H - 1:
                next_v[s] = max(float(np.dot(mdp.phi[s], omega)),
                                float(mdp.r[s, 1]))
            else:
                next_v[s] = float(mdp.r[s, 1])

        # 按维度 i 求解一维 β 的最优并更新 ω_i
        for i in range(mdp.d):
            if abs(psi[i].sum()) < 1e-12:
                omega[i] = 0.0
            else:
                # 最小化 g(β)（等价于原问题的 sup 的负号），并做 value shift（见 g_with_d）
                _, gmin = fminbound(
                    lambda b: g_with_d(b, psi[i], next_v, rho),
                    BETA_MIN, BETA_MAX
                )
                # 论文式：θ_i + min_v - min_β g(β)；注意我们在 g 中已减了 min_v
                omega[i] = theta[i] - gmin
        # 裁剪（防止震荡/数值溢出）
        omega = np.clip(omega, 0.0, mdp.r.max())
        omegas.append(omega.copy())

    omegas.reverse()
    return omegas

# ---------- 训练/评估工具 ----------
def collect_dataset(env_for_data: OptionMDP, N_traj: int, seed=None):
    if seed is not None:
        np.random.seed(seed)
    policy = FixedPolicy(env_for_data.N_S, env_for_data.N_A)
    return [generate_traj(env_for_data, policy) for _ in range(N_traj)]

def train_policy(mdp_train: OptionMDP, dataset, rho: float):
    omegas = DRVI(mdp_train, dataset, rho)
    return OptionLinearPolicy(mdp_train, omegas)

def evaluate_avg_return(env_eval: OptionMDP, policy: OptionLinearPolicy, episodes=2000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    total = 0.0
    for _ in range(episodes):
        s = env_eval.reset()
        ret = 0.0
        for h in range(env_eval.H):
            a = policy.sample(s, h)
            s, r, done = env_eval.step(a)
            ret += r
            if done:
                break
        total += ret
    # 还原奖励尺度
    return (total / episodes) * env_eval.r_scale

def compute_value_vector(mdp: OptionMDP, policy: OptionLinearPolicy, h=0):
    # 直接读取阶段 h 的“一步贪心”价值（由反向传回保证一致性）
    V = np.array([policy.V(s, h) for s in range(mdp.N_S)], dtype=float)
    return V

def sup_norm(a, b):
    return float(np.max(np.abs(a - b)))

# ================== Fig. 2 (a) 平均总收益 ==================
def figure2a(seed=666):
    np.random.seed(seed)
    d = 61
    N = 1000
    p0_train = 0.5
    # 用训练环境采集数据（p0 = 0.5）
    env_data = OptionMDP(d=d, tau=p0_train)
    dataset = collect_dataset(env_data, N_traj=N, seed=seed)

    # 在相同 d、相同（名义）环境上训练不同 ρ 的策略
    rhos = [0.0, 0.01, 0.02, 0.05, 0.10]  # ρ=0 视作非鲁棒
    labels = ["non robust", r"$\rho=0.01$", r"$\rho=0.02$", r"$\rho=0.05$", r"$\rho=0.10$"]

    policies = []
    for rho in rhos:
        mdp_train = OptionMDP(d=d, tau=p0_train)  # 训练时的名义模型
        pol = train_policy(mdp_train, dataset, rho)
        policies.append(pol)

    # 在扰动环境上评估：p0 ∈ [0.3, 0.7]
    p0_grid = np.linspace(0.3, 0.7, 9)
    avg_returns = [[] for _ in rhos]
    for p0 in p0_grid:
        mdp_eval = OptionMDP(d=d, tau=float(p0))
        for idx, pol in enumerate(policies):
            # 将策略映射到评估环境（同一组 ω_h 在不同 tau 下执行）
            pol_eval = OptionLinearPolicy(mdp_eval, pol.omegas)
            avg = evaluate_avg_return(mdp_eval, pol_eval, episodes=2000, seed=seed+1)
            avg_returns[idx].append(avg)

    # 画图
    plt.figure(figsize=(5.2, 4.0))
    for idx, y in enumerate(avg_returns):
        plt.plot(p0_grid, y, marker='o', label=labels[idx])
    plt.xlabel(r"$p_0$")
    plt.ylabel("Average Total Return")
    plt.title("(a) Average Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ================== Fig. 2 (b) 估计误差 ||V̂1 - V*1|| ==================
def figure2b(seed=666):
    np.random.seed(seed)
    rhos = 0.01
    d_list = [31, 61, 121, 301, 601]
    N_list = [100, 200, 400, 800, 1600, 3200, 6400]
    TRIES = 20  # 论文中用 20 次重复

    p0_train = 0.5
    lgN = np.log10(N_list)

    # 准备画布
    plt.figure(figsize=(5.2, 4.0))

    for d in d_list:
        # 用大样本构造“近似真值” V*（固定一次，避免高方差），可按需调小 BIG_N
        BIG_N = 20000
        env_star = OptionMDP(d=d, tau=p0_train)
        dataset_star = collect_dataset(env_star, BIG_N, seed=seed)
        pol_star = train_policy(env_star, dataset_star, rhos)
        V_star = compute_value_vector(env_star, pol_star, h=0)

        errs = []
        for N in N_list:
            e_sum = 0.0
            for t in range(TRIES):
                env_train = OptionMDP(d=d, tau=p0_train)
                dataset = collect_dataset(env_train, N, seed=seed + 1000 + t)
                pol_hat = train_policy(env_train, dataset, rhos)
                V_hat = compute_value_vector(env_train, pol_hat, h=0)
                # 使用 ∞-范数（与理论常用范数一致）
                e = sup_norm(V_hat, V_star)
                e_sum += e
            errs.append(e_sum / TRIES)
        plt.plot(lgN, errs, marker='o', label=f"d = {d}")

    plt.xlabel(r"$\lg N$")
    plt.ylabel(r"$\|\hat V_1 - V^*_1\|$")
    plt.title("(b) " + r"$\|\hat V_1 - V^*_1\|$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ================== Fig. 2 (c) 执行时间（秒） ==================
def figure2c(seed=666):
    np.random.seed(seed)
    rhos = 0.01
    d_list = [31, 61, 121]  # 与图中一致
    N_list = [100, 200, 400, 800, 1600, 3200, 6400]
    TRIES = 5  # 为节省时间，时间统计重复次数可以较小；如需更平滑可设为 10 或 20

    p0_train = 0.5
    lgN = np.log10(N_list)

    plt.figure(figsize=(5.2, 4.0))
    for d in d_list:
        ts = []
        for N in N_list:
            t_sum = 0.0
            for t in range(TRIES):
                env = OptionMDP(d=d, tau=p0_train)
                dataset = collect_dataset(env, N, seed=seed + 2000 + t)
                t0 = time.perf_counter()
                _ = DRVI(env, dataset, rhos)
                t1 = time.perf_counter()
                t_sum += (t1 - t0)
            ts.append(t_sum / TRIES)
        plt.plot(lgN, ts, marker='o', label=f"d = {d}")
    plt.xlabel(r"$\lg N$")
    plt.ylabel("Time (s)")
    plt.title("(c) Execution time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ================== 主入口 ==================
if __name__ == "__main__":
    # Fig. 2 (a)
    figure2a(seed=666)
    # Fig. 2 (b)
    figure2b(seed=666)
    # Fig. 2 (c)
    figure2c(seed=666)
