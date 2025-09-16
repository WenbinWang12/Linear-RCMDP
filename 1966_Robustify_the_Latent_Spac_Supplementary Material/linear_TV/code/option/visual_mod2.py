# -*- coding: utf-8 -*-
# Reproduce Fig. 2 in:
# Ma et al., "Distributionally Robust Offline RL with Linear Function Approximation" (American Option)
# 本实现加入：
# 1) 每个绘图点多次独立重复并取均值
# 2) DRVI(calibrate 开关, 固定 lam_eff)
# 3) Fig.2(b) 使用更干净的“真值”，禁用校准
# 4) Fig.2(c) 计时用中位数 + 两次预热，仅计 _fit_once（禁校准）
# 5) 预计算 alphas（每个 h 一次），移除“||psi||_1 很小就跳过”的分支，减少路径差异与计时抖动

# —— 固定 BLAS/OpenMP 线程，降低系统层面的计时噪声（需在 import numpy 之前）——
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import time
import matplotlib.pyplot as plt

# ---------- 数值常量 ----------
LOG_EPS   = 1e-12      # 防止 log(0)
BETA_MIN  = 1e-6       # 对偶变量 β 的搜索下界
BETA_MAX  = 50.0       # 对偶变量 β 的搜索上界（ρ→0 时 β*→∞，用有限上界近似）
TOL       = 1e-8       # 一维搜索容差

# ---------- 实验重复参数（可按需调节；数值越大越稳、但越耗时） ----------
REPS_A   = 5   # Fig.2(a) 每个 (rho, p0) 点的独立重复次数
TRIES_B  = 20  # Fig.2(b) 每个 (d, N) 点的独立重复次数（论文用 20）
TRIES_C  = 12  # Fig.2(c) 每个 (d, N) 点的计时重复次数（稍增大以稳中位数）

# ========== 基础工具 ==========
def compute_penalty(phi_s, Lambda_inv, gamma0):
    """
    精确置信半径：Gamma(s) = gamma0 * ||phi(s)||_{Lambda^{-1}}
                 = gamma0 * sqrt( phi(s)^T Lambda_inv phi(s) )
    """
    phi_s = np.asarray(phi_s)
    val = float(phi_s @ (Lambda_inv @ phi_s))
    val = max(val, 0.0)
    return float(gamma0 * np.sqrt(val))

def tv_nu_with_d(psi_i, v, rho):
    """保留原版函数（若其他实验需要），但 DRVI 中将使用预计算版以稳定耗时。"""
    v = np.asarray(v); psi_i = np.asarray(psi_i)
    v_min, v_max = float(v.min()), float(v.max())
    if v_max - v_min < 1e-15:
        return float(np.dot(psi_i, v))
    alphas = np.unique(np.concatenate([v, [v_min, v_max]]))
    best = -np.inf
    for alpha in alphas:
        v_clip = np.minimum(v, alpha)
        obj = np.dot(psi_i, v_clip) - rho * (alpha - float(v_clip.min()))
        if obj > best: best = obj
    return float(best)

# 预计算版（每个 h 一次性生成 alphas，内部仅用它）
def tv_nu_precomp(psi_i, v, alphas, rho):
    psi_i = np.asarray(psi_i, dtype=float)
    v = np.asarray(v, dtype=float)
    best = -np.inf
    for alpha in alphas:
        v_clip = np.minimum(v, alpha)
        obj = float(np.dot(psi_i, v_clip)) - rho * (float(alpha) - float(np.min(v_clip)))
        if obj > best:
            best = obj
    return best

# ========== 环境 ==========
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

                # 奖励：看跌期权 r(s,1)=max(0, K - x)；(s,0)=0
                r[s, 1] = max(0, K - self.s2x(s))
                r[s, 0] = 0

                # 约束 utility：价格落在 [K-10,K+10] 视为“合规=1”，否则 0
                in_band = (safe_lo <= self.s2x(s) <= safe_hi)
                g[s, 0] = 1.0 if in_band else 0.0
                g[s, 1] = 1.0

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
        done = (self.h >= self.H)
        return self.state, reward, util, done  # 返回四元组

# ========== 策略 ==========
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

class OptionLinearPolicy():
    def __init__(self, mdp, weights_r, weights_g, Lambda_inv, gamma0, beta, b):
        self.mdp = mdp
        self.wr = weights_r  # list/array of shape (H, d) 等价
        self.wg = weights_g
        self.beta = float(beta)
        self.b = float(b)
        self.state_penalty = np.array([
            compute_penalty(self.mdp.phi[s], Lambda_inv, gamma0)
            for s in range(self.mdp.N_S)
        ])
        # 保存权重与杂项，便于跨环境映射
        self.weights_r = weights_r
        self.weights_g = weights_g
        self.misc = {'Lambda_inv': Lambda_inv, 'gamma0': gamma0, 'beta': beta, 'b': b}

    def _Q_r(self, s, h):
        Q0 = float(self.mdp.phi[s] @ self.wr[h] - self.state_penalty[s])
        Q1 = float(self.mdp.r[s, 1])
        return (Q0, Q1)

    def _Q_g(self, s, h):
        Q0 = float(self.mdp.phi[s] @ self.wg[h] + self.state_penalty[s])
        Q1 = float(self.mdp.g[s, 1])
        return (Q0, Q1)

    def sample(self, state, h):
        Qr0, Qr1 = self._Q_r(state, h)
        Qg0, Qg1 = self._Q_g(state, h)
        obj0 = Qr0 - self.beta * max(self.b - Qg0, 0.0)
        obj1 = Qr1 - self.beta * max(self.b - Qg1, 0.0)
        return 0 if obj0 >= obj1 else 1

    def V(self, state, h):
        Qr0, Qr1 = self._Q_r(state, h)
        Qg0, Qg1 = self._Q_g(state, h)
        obj0 = Qr0 - self.beta * max(self.b - Qg0, 0.0)
        obj1 = Qr1 - self.beta * max(self.b - Qg1, 0.0)
        return max(obj0, obj1)

# ========== 采样轨迹 ==========
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

# ========== DRVI ==========
def DRVI(
    mdp,
    dataset,
    rho=0.003,
    b=0.7,
    eps=0.2,
    delta=0.05,
    lam0=1.0,
    c_gamma=0.5,
    cap_ratio=0.25,
    max_calib=3,
    calibrate=True,   # 新增：是否进行基于 frac_exercise 的校准
):
    """
    Constrained Robust VI under TV divergence with pessimism/optimism and Mahalanobis penalty.
    返回：
    weights_r : list 长度 H，每个是 (d,)
    weights_g : list 长度 H，每个是 (d,)
    misc      : dict {'Lambda_inv','gamma0','beta','b','frac_exercise'}
    """
    H, d, S = mdp.H, mdp.d, mdp.N_S
    K = len(dataset)
    beta = H / max(float(eps), 1e-12)

    def _pen_mahalanobis(phi_s, Lambda_inv, gamma0_):
        phi_s = np.asarray(phi_s, dtype=float)
        val = float(phi_s @ (Lambda_inv @ phi_s))
        if val < 0.0:
            val = 0.0
        return float(gamma0_ * np.sqrt(val))

    # ---------- 单次拟合（不含校准循环） ----------
    def _fit_once(rho_, gamma0_):
        # 固定 lam_eff 为常数，避免随 N 改变任务
        lam_eff = float(lam0)
        Lambda = lam_eff * np.eye(d)
        psi_hat = np.zeros((d, S), dtype=float)
        theta_r_hat = np.zeros(d, dtype=float)
        theta_g_hat = np.zeros(d, dtype=float)

        for traj in dataset:
            for (s, a, r, g, ns) in traj:
                phi_s = mdp.phi[s]
                psi_hat[:, ns] += phi_s
                theta_r_hat += phi_s * float(r)
                theta_g_hat += phi_s * float(g)
                Lambda += np.outer(phi_s, phi_s)

        Lambda_inv = np.linalg.inv(Lambda)
        psi = Lambda_inv @ psi_hat
        theta_r = Lambda_inv @ theta_r_hat
        theta_g = Lambda_inv @ theta_g_hat

        r_max = float(np.max(mdp.r)) if hasattr(mdp, "r") else 1.0
        cap_pen = float(cap_ratio) * max(r_max, 1e-6) if cap_ratio is not None else np.inf

        state_penalty = np.empty(S, dtype=float)
        for s in range(S):
            pen = _pen_mahalanobis(mdp.phi[s], Lambda_inv, gamma0_)
            state_penalty[s] = pen if pen < cap_pen else cap_pen

        w_r = np.zeros(d, dtype=float)
        w_g = np.zeros(d, dtype=float)
        V_r_next = np.zeros(S, dtype=float)
        V_g_next = np.zeros(S, dtype=float)
        weights_r, weights_g = [], []

        for h in reversed(range(H)):
            if h < H - 1:
                Qr0_next = mdp.phi @ w_r - state_penalty
                Qr1_next = mdp.r[:, 1]
                V_r_next = np.maximum(Qr0_next, Qr1_next)

                Qg0_next = mdp.phi @ w_g + state_penalty
                Qg1_next = mdp.g[:, 1]
                V_g_next = np.maximum(Qg0_next, Qg1_next)
            else:
                V_r_next = mdp.r[:, 1].astype(float).copy()
                V_g_next = mdp.g[:, 1].astype(float).copy()

            # —— 预计算 alphas（每个 h 一次） —— #
            vr_min, vr_max = float(np.min(V_r_next)), float(np.max(V_r_next))
            vg_min, vg_max = float(np.min(V_g_next)), float(np.max(V_g_next))
            vr_shift = V_r_next - vr_min
            vg_shift = V_g_next - vg_min

            alphas_r = np.unique(np.concatenate([vr_shift, [0.0, float(vr_max - vr_min)]]))
            alphas_g = np.unique(np.concatenate([vg_shift, [0.0, float(vg_max - vg_min)]]))

            for i in range(d):
                # 移除“||psi||_1 很小就跳过”的分支，统一总是计算，减少路径差异
                nu_r_i = tv_nu_precomp(psi[i], vr_shift, alphas_r, rho_)
                nu_g_i = tv_nu_precomp(psi[i], vg_shift, alphas_g, rho_)
                w_r[i] = theta_r[i] + nu_r_i
                w_g[i] = theta_g[i] + nu_g_i

            # 放宽裁剪：仅约束为非负，避免上界裁剪导致分段行为
            w_r = np.maximum(w_r, 0.0)
            w_g = np.maximum(w_g, 0.0)

            weights_r.append(w_r.copy())
            weights_g.append(w_g.copy())

        weights_r.reverse()
        weights_g.reverse()

        Qr0_h0 = mdp.phi @ weights_r[0] - state_penalty
        Qr1_h0 = mdp.r[:, 1]
        frac_exercise = float(np.mean(Qr1_h0 >= Qr0_h0))

        misc = dict(
            Lambda_inv=Lambda_inv,
            gamma0=gamma0_,
            beta=H / max(float(eps), 1e-12),
            b=float(b),
            frac_exercise=frac_exercise,
        )
        return weights_r, weights_g, misc

    # ---------- gamma0 初值 ----------
    xi0 = np.log(max(3.0 * mdp.H * max(K, 1) / max(float(delta), 1e-12), 1.0))
    gamma0 = float(c_gamma) * np.sqrt(max(mdp.d * xi0 / max(K, 1), 0.0))
    rho_curr = float(rho)

    if not calibrate:
        return _fit_once(rho_curr, gamma0)

    # ---------- 校准（如启用） ----------
    for attempt in range(max_calib + 1):
        wr, wg, misc = _fit_once(rho_curr, gamma0)
        frac_ex = misc["frac_exercise"]
        if frac_ex < 0.95:
            return wr, wg, misc
        gamma0 *= 0.5
        rho_curr *= 0.5

    return wr, wg, misc

# ========== 训练/评估工具 ==========
def collect_dataset(env_for_data: OptionMDP, N_traj: int, seed=None):
    if seed is not None:
        np.random.seed(seed)
    policy = FixedPolicy(env_for_data.N_S, env_for_data.N_A)
    return [generate_traj(env_for_data, policy) for _ in range(N_traj)]

def train_policy(mdp_train: OptionMDP, dataset, rho: float, calibrate=True):
    weights_r, weights_g, misc = DRVI(mdp_train, dataset, rho, calibrate=calibrate)
    return OptionLinearPolicy(mdp_train, weights_r, weights_g,
                              misc['Lambda_inv'], misc['gamma0'],
                              misc['beta'], misc['b'])

def evaluate_avg_return(env_eval: OptionMDP, policy: OptionLinearPolicy,
                        episodes=2000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    total = 0.0
    for _ in range(episodes):
        s = env_eval.reset()
        ret = 0.0
        done = False
        for h in range(env_eval.H):
            a = policy.sample(s, h)
            s, r, util, done = env_eval.step(a)
            ret += r
            if done:
                break
        total += ret
    # 还原奖励尺度
    return (total / episodes) * env_eval.r_scale

def compute_value_vector(mdp: OptionMDP, policy: OptionLinearPolicy, h=0):
    V = np.array([policy.V(s, h) for s in range(mdp.N_S)], dtype=float)
    return V

def sup_norm(a, b):
    return float(np.max(np.abs(a - b)))

# ========== Fig. 2 (a) 平均总收益：每点多次独立重复 ==========
def figure2a(seed=666):
    np.random.seed(seed)
    d = 61
    N = 1000
    p0_train = 0.5

    # ρ 列表
    rhos = [0.0, 0.01, 0.02, 0.05, 0.10]
    labels = ["non robust", r"$\rho=0.01$", r"$\rho=0.02$", r"$\rho=0.05$", r"$\rho=0.10$"]

    # 评估环境网格：p0 ∈ [0.3, 0.7]
    p0_grid = np.linspace(0.3, 0.7, 9)

    avg_returns = [np.zeros(len(p0_grid), dtype=float) for _ in rhos]

    # 对每个点进行 REPS_A 次完全独立重复：重采数据、重训、重评估
    for rep in range(REPS_A):
        # 每次重复都基于同一名义训练环境 tau=0.5，但数据重新采集
        env_data = OptionMDP(d=d, tau=p0_train)
        data_seed = seed + 100000 * rep + 123  # 确保不同 rep 不同随机数
        dataset = collect_dataset(env_data, N_traj=N, seed=data_seed)

        # 训练多组策略（每个 rho 一组）
        trained_policies = []
        for ri, rho in enumerate(rhos):
            mdp_train = OptionMDP(d=d, tau=p0_train)
            # Fig2(a) 保留 calibrate=True（若希望完全固定可改 False）
            pol = train_policy(mdp_train, dataset, rho, calibrate=True)
            trained_policies.append(pol)

        # 在每个评估 p0 上评估
        for j, p0 in enumerate(p0_grid):
            mdp_eval = OptionMDP(d=d, tau=float(p0))
            for ri, pol in enumerate(trained_policies):
                # 将策略映射到评估环境
                pol_eval = OptionLinearPolicy(
                    mdp_eval,
                    pol.weights_r, pol.weights_g,
                    pol.misc['Lambda_inv'], pol.misc['gamma0'],
                    pol.misc['beta'], pol.misc['b']
                )
                eval_seed = seed + 100000 * rep + 1000 * ri + j
                avg = evaluate_avg_return(mdp_eval, pol_eval,
                                          episodes=2000, seed=eval_seed)
                avg_returns[ri][j] += avg

    # 取每点均值
    for ri in range(len(rhos)):
        avg_returns[ri] /= REPS_A

    # 画图
    plt.figure(figsize=(5.6, 4.2))
    for idx, y in enumerate(avg_returns):
        plt.plot(p0_grid, y, marker='o', label=labels[idx])
    plt.xlabel(r"$p_0$")
    plt.ylabel("Average Total Return")
    plt.title("(a) Average Return (mean over {} runs/pt)".format(REPS_A))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figure2a.png", dpi=600)
    plt.show()
    plt.close()

# ========== Fig. 2 (b) 估计误差 ||V̂1 - V*1||：保留多次独立重复 ==========
def figure2b(seed=666):
    np.random.seed(seed)
    rho_fixed = 0.01
    d_list = [31, 61, 121, 301, 601]
    N_list = [100, 200, 400, 800, 1600, 3200, 6400]
    TRIES = TRIES_B  # 可调

    p0_train = 0.5
    lgN = np.log10(N_list)

    plt.figure(figsize=(5.6, 4.2))

    for d in d_list:
        # —— 构造更干净的“真值” V*（更大样本 + 多次平均；并关闭校准与训练端一致） —— #
        BIG_N = 100000
        R_STAR = 3
        V_star_runs = []
        env_star = OptionMDP(d=d, tau=p0_train)
        for r in range(R_STAR):
            dataset_star = collect_dataset(env_star, BIG_N, seed=seed + d * 7 + 10000 * r + 1)
            pol_star = train_policy(env_star, dataset_star, rho_fixed, calibrate=False)
            V_star_runs.append(compute_value_vector(env_star, pol_star, h=0))
        V_star = np.mean(np.stack(V_star_runs, axis=0), axis=0)

        errs = []
        for N in N_list:
            e_sum = 0.0
            for t in range(TRIES):
                env_train = OptionMDP(d=d, tau=p0_train)
                dataset = collect_dataset(env_train, N, seed=seed + 1000 + d * 37 + t)
                # 关闭校准：保持不同 N 的任务一致（同一组超参）
                pol_hat = train_policy(env_train, dataset, rho_fixed, calibrate=False)
                V_hat = compute_value_vector(env_train, pol_hat, h=0)
                e_sum += sup_norm(V_hat, V_star)
            errs.append(e_sum / TRIES)
        plt.plot(lgN, errs, marker='o', label=f"d = {d}")

    plt.xlabel(r"$\lg N$")
    plt.ylabel(r"$\|\hat V_1 - V^*_1\|$")
    plt.title("(b) " + r"$\|\hat V_1 - V^*_1\|$" + f" (mean over {TRIES} runs/pt)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figure2b.png", dpi=600)
    plt.show()
    plt.close()

# ========== Fig. 2 (c) 执行时间（秒）：每点重复计时取“中位数” ==========
def figure2c(seed=666):
    np.random.seed(seed)
    rho_fixed = 0.01
    d_list = [31, 61, 121]  # 与论文图一致
    N_list = [100, 200, 400, 800, 1600, 3200, 6400]
    TRIES = TRIES_C  # 次数略大一些以稳定中位数

    p0_train = 0.5
    lgN = np.log10(N_list)

    plt.figure(figsize=(5.6, 4.2))
    for d in d_list:
        ts = []
        for j, N in enumerate(N_list):
            times = []
            for t in range(TRIES):
                env = OptionMDP(d=d, tau=p0_train)
                dataset = collect_dataset(env, N, seed=seed + 2000 + d * 97 + j * 13 + t)
                # 预热两次（不计时），尽量稳定缓存/内存池等
                _ = DRVI(env, dataset, rho_fixed, calibrate=False)
                _ = DRVI(env, dataset, rho_fixed, calibrate=False)
                # 计时：只做一次 _fit_once（calibrate=False）
                t0 = time.perf_counter()
                _ = DRVI(env, dataset, rho_fixed, calibrate=False)
                t1 = time.perf_counter()
                times.append(t1 - t0)
            # 用中位数降低极端值影响；如需可改为“去极值均值”
            ts.append(float(np.median(np.array(times))))
        plt.plot(lgN, ts, marker='o', label=f"d = {d}")

    plt.xlabel(r"$\lg N$")
    plt.ylabel("Time (s)")
    plt.title("(c) Execution time (median over {} runs/pt)".format(TRIES))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figure2c.png", dpi=600)
    plt.show()
    plt.close()

# ========== 主入口 ==========
if __name__ == "__main__":
    # Fig. 2 (a)
    figure2a(seed=666)
    # Fig. 2 (b)
    figure2b(seed=666)
    # Fig. 2 (c)
    figure2c(seed=666)
