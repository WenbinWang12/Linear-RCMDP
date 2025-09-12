import numpy as np
import scipy.optimize as opt

LOG_EPS = 1e-20
BETA_MIN = 1e-2
BETA_MAX = 1e2

#######
# VI
#######


def VI(mdp, dataset):
    K = len(dataset)
    omegas = []
    omega = np.zeros((mdp.d,))  # (d,)
    for h in reversed(range(mdp.H)):
        Lambda = np.identity(mdp.d)
        target_v = 0
        for k in range(K):
            s, a, r, next_s = dataset[k][h]
            if h < mdp.H - 1:
                next_v = np.maximum(np.dot(mdp.phi[next_s], omega), mdp.r[next_s, 1])
            else:
                next_v = mdp.r[next_s, 1]
            phi_k = mdp.phi[s]  # (d,)
            Lambda += phi_k[:, np.newaxis] * phi_k[np.newaxis, :]
            target_v += phi_k * (r + next_v)
        Lambda_inv = np.linalg.inv(Lambda)
        omega = np.matmul(Lambda_inv, target_v)
        omega = np.clip(omega, 0, mdp.r.max())
        omegas.append(omega)
    omegas.reverse()
    return omegas


#######
# DRVI
#######


import numpy as np

# ==== TV dual for d-rectangular (Wang et al., 2024) ====

def tv_nu_with_d(psi_i, v, rho):
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

def compute_penalty(phi_s, Lambda_inv, gamma0):
    """
    精确置信半径：Gamma(s) = gamma0 * ||phi(s)||_{Lambda^{-1}}
                 = gamma0 * sqrt( phi(s)^T Lambda_inv phi(s) )
    """
    phi_s = np.asarray(phi_s)
    val = float(phi_s @ (Lambda_inv @ phi_s))
    val = max(val, 0.0)
    return float(gamma0 * np.sqrt(val))




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
):
    """
    Constrained Robust VI under TV divergence with pessimism/optimism and Mahalanobis penalty.

    参数
    ----
    mdp      : 环境（需含属性 phi (S,d), r (S,2), g (S,2), H, N_S, d）
    dataset  : list of trajectories，每条为 [(s,a,r,g,next_s), ...]，长度约为 H
    rho      : TV 球半径（越大越保守）
    b        : 约束阈值（希望 V_g >= b）
    eps      : 约束容忍（策略用到：beta = H/eps）
    delta    : 置信水平（影响 gamma0 的尺度）
    lam0     : 基础岭正则（实际会放大为 max(lam0, 0.1*K/d)）
    c_gamma  : 惩罚系数的倍率（默认 0.5，温和）
    cap_ratio: 惩罚的硬上限相对 r_max 的比例（默认 0.25）
    max_calib: 若退化为“总是行权”，自动把 gamma0 和 rho 各减半并重训的最大轮数

    返回
    ----
    weights_r : list, 长度 H，每个元素 shape (d,) —— 奖励侧权重（悲观）
    weights_g : list, 长度 H，每个元素 shape (d,) —— 约束侧权重（乐观）
    misc      : dict，含 {'Lambda_inv','gamma0','beta','b'}
    """

    H, d, S = mdp.H, mdp.d, mdp.N_S
    K = len(dataset)
    beta = H / max(float(eps), 1e-12)

    # ---------- 内部小工具：TV 对偶 & 马氏范数惩罚 ----------
    def _tv_nu_with_d(psi_i, v, rho_):
        """nu_i = max_{alpha in [min v, max v]} { psi_i^T min(v,alpha) - rho*(alpha - min(min(v,alpha))) }"""
        v = np.asarray(v, dtype=float)
        psi_i = np.asarray(psi_i, dtype=float)
        v_min = float(np.min(v)); v_max = float(np.max(v))
        if v_max - v_min < 1e-15:
            return float(np.dot(psi_i, v))
        alphas = np.unique(np.concatenate([v, [v_min, v_max]]))
        best = -np.inf
        for alpha in alphas:
            v_clip = np.minimum(v, alpha)
            obj = float(np.dot(psi_i, v_clip)) - rho_ * (float(alpha) - float(np.min(v_clip)))
            if obj > best:
                best = obj
        return float(best)

    def _pen_mahalanobis(phi_s, Lambda_inv, gamma0_):
        """Gamma(s) = gamma0 * sqrt( phi(s)^T Lambda^{-1} phi(s) )"""
        phi_s = np.asarray(phi_s, dtype=float)
        val = float(phi_s @ (Lambda_inv @ phi_s))
        if val < 0.0:  # 数值防御
            val = 0.0
        return float(gamma0_ * np.sqrt(val))

    # ---------- 单次拟合：回归 + 反向VI ----------
    def _fit_once(rho_, gamma0_):
        # ---- 线性回归（自适应加大岭正则） ----
        lam_eff = max(float(lam0), 0.1 * max(K, 1) / max(d, 1))  # 自适应更强的岭正则
        Lambda = lam_eff * np.eye(d)
        psi_hat = np.zeros((d, S), dtype=float)
        theta_r_hat = np.zeros(d, dtype=float)
        theta_g_hat = np.zeros(d, dtype=float)

        for traj in dataset:
            for (s, a, r, g, ns) in traj:
                phi_s = mdp.phi[s]  # (d,)
                psi_hat[:, ns] += phi_s
                theta_r_hat += phi_s * float(r)
                theta_g_hat += phi_s * float(g)
                Lambda += np.outer(phi_s, phi_s)

        Lambda_inv = np.linalg.inv(Lambda)
        psi = Lambda_inv @ psi_hat          # (d, S)
        theta_r = Lambda_inv @ theta_r_hat  # (d,)
        theta_g = Lambda_inv @ theta_g_hat  # (d,)

        # ---- 惩罚（马氏范数）+ 硬帽 ----
        r_max = float(np.max(mdp.r)) if hasattr(mdp, "r") else 1.0
        cap_pen = float(cap_ratio) * max(r_max, 1e-6)

        state_penalty = np.empty(S, dtype=float)
        for s in range(S):
            pen = _pen_mahalanobis(mdp.phi[s], Lambda_inv, gamma0_)
            state_penalty[s] = pen if pen < cap_pen else cap_pen

        # ---- 反向阶段迭代（奖励悲观 / 约束乐观）----
        w_r = np.zeros(d, dtype=float)
        w_g = np.zeros(d, dtype=float)
        V_r_next = np.zeros(S, dtype=float)
        V_g_next = np.zeros(S, dtype=float)
        weights_r, weights_g = [], []

        for h in reversed(range(H)):
            if h < H - 1:
                # 奖励侧悲观：动作0扣惩罚、动作1=行权（不扣罚）
                Qr0_next = mdp.phi @ w_r - state_penalty
                Qr1_next = mdp.r[:, 1]
                V_r_next = np.maximum(Qr0_next, Qr1_next)

                # 约束侧乐观：动作0加惩罚、动作1=行权（不加罚）
                Qg0_next = mdp.phi @ w_g + state_penalty
                Qg1_next = mdp.g[:, 1]
                V_g_next = np.maximum(Qg0_next, Qg1_next)
            else:
                # 末步：下一阶段价值由行权确定
                V_r_next = mdp.r[:, 1].astype(float).copy()
                V_g_next = mdp.g[:, 1].astype(float).copy()

            # 为数值稳定，对 V 做非负平移（对偶只差常数）
            vr_shift = V_r_next - float(np.min(V_r_next))
            vg_shift = V_g_next - float(np.min(V_g_next))

            # 坐标更新：nu_i(TV) + theta_i
            for i in range(d):
                if np.linalg.norm(psi[i], 1) < 1e-12:
                    w_r[i] = 0.0
                    w_g[i] = 0.0
                else:
                    nu_r_i = _tv_nu_with_d(psi[i], vr_shift, rho_)
                    nu_g_i = _tv_nu_with_d(psi[i], vg_shift, rho_)
                    w_r[i] = theta_r[i] + nu_r_i
                    w_g[i] = theta_g[i] + nu_g_i

            # 合理裁剪（防止偶发外溢）
            w_r = np.clip(w_r, 0.0, float(max(1.0, np.max(mdp.r))))
            w_g = np.clip(w_g, 0.0, float(max(1.0, np.max(mdp.g))))

            weights_r.append(w_r.copy())
            weights_g.append(w_g.copy())

        weights_r.reverse()
        weights_g.reverse()

        # 退化诊断：h=0 时“继续 vs 行权”
        Qr0_h0 = mdp.phi @ weights_r[0] - state_penalty
        Qr1_h0 = mdp.r[:, 1]
        frac_exercise = float(np.mean(Qr1_h0 >= Qr0_h0))

        misc = dict(
            Lambda_inv=Lambda_inv,
            gamma0=gamma0_,
            beta=beta,
            b=float(b),
            frac_exercise=frac_exercise,
        )
        return weights_r, weights_g, misc

    # ---------- 设置 gamma0（温和、随 K 衰减） ----------
    # xi0 ~ log(3HK/delta)，常见的自信半径项；gamma0 = c_gamma * sqrt(d * xi0 / K)
    xi0 = np.log(max(3.0 * H * max(K, 1) / max(float(delta), 1e-12), 1.0))
    gamma0 = float(c_gamma) * np.sqrt(max(d * xi0 / max(K, 1), 0.0))
    rho_curr = float(rho)

    # ---------- 校准循环：若几乎总是行权，则减半 gamma0、rho 重训 ----------
    for attempt in range(max_calib + 1):
        wr, wg, misc = _fit_once(rho_curr, gamma0)
        frac_ex = misc["frac_exercise"]
        # 若不是几乎全行权（例如 < 0.95），接受本次结果
        if frac_ex < 0.95:
            return wr, wg, misc
        # 否则降强度并重训
        gamma0 *= 0.5
        rho_curr *= 0.5

    # 多次仍退化，就返回最后一次（极少发生）
    return wr, wg, misc


#####################
# DRVI2, Tamar13'
#####################


def g_with_L(beta, phi, Lambda_inv, Phi, V, rho):
    '''
    beta: scalar
    phi: (d,)
    Lambda_inv: (d, d)
    Phi: (K, d)
    V: (K,)
    '''
    V_mm = V.max() - V.min()
    EV = np.expm1(-(V / beta))  # (K,)
    y = np.sum(Phi * EV[:, np.newaxis], axis=0)  # (d,)
    w = np.matmul(Lambda_inv, y)  # (d,)
    x = np.dot(phi, w).clip(np.expm1(-V_mm / beta), 0)
    g = beta * rho + beta * np.log1p(x + LOG_EPS)
    return g


def get_next_value(phi, Lambda_inv, Phi, V, rho):
    V_max, V_min = V.max(), V.min()
    if V_max == V_min:
        return V_min
    V = (V - V_min) / (V_max - V_min)
    g = -opt.fminbound(lambda b: g_with_L(b, phi, Lambda_inv, Phi, V, rho), BETA_MIN, BETA_MAX, full_output=True)[1]
    return V_min + g * (V_max - V_min)


def DRVI2(mdp, dataset, rho, lcb_coef=0.):
    """
    Episodic variant of Tamer13'
    """
    K = len(dataset)
    omegas = []
    omega = np.zeros((mdp.d,))  # (d,)
    for h in reversed(range(mdp.H)):
        Lambda = np.identity(mdp.d)
        NEXT_V = np.zeros((K,))
        Phi = np.zeros((K, mdp.d))
        for k in range(K):
            s, a, r, next_s = dataset[k][h]
            phi_k = mdp.phi[s]  # (d,)
            Phi[k] = phi_k
            Lambda += phi_k[:, np.newaxis] * phi_k[np.newaxis, :]
            if h < mdp.H - 1:
                next_v = np.dot(mdp.phi[next_s], omega)
                NEXT_V[k] = np.maximum(next_v, mdp.r[next_s, 1])
            else:
                NEXT_V[k] = mdp.r[next_s, 1]

        Lambda_inv = np.linalg.inv(Lambda)
        # Next robust value
        target_v = np.zeros((mdp.d,))
        for k in range(K):
            s, a, r, next_s = dataset[k][h]
            phi_k = mdp.phi[s]  # (d,)
            next_v = get_next_value(phi_k, Lambda_inv, Phi, NEXT_V, rho)
            target_v += phi_k * (r + next_v)

        omega = np.matmul(Lambda_inv, target_v)
        omega = np.clip(omega, 0, mdp.r.max())
        omegas.append(omega.copy())
    omegas.reverse()
    return omegas
