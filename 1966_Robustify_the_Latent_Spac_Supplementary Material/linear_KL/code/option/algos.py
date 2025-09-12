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


def g_with_d(beta, psi, v, rho):
    '''
    beta: scalar
    psi: (S,)
    v: (S,)
    rho: scalar
    '''
    v_mm = v.max() - v.min()
    y = np.dot(psi, np.expm1(-(v - v.min()) / beta)).clip(np.expm1(-v_mm / beta), 0)
    g = beta * rho + beta * np.log1p(y + LOG_EPS) - v.min()
    return g


def DRVI(mdp, dataset, rho):
    '''
    Learn an emprical model.
    '''
    K = len(dataset)
    omegas = []

    # share the features over all stages
    Lambda = np.identity(mdp.d)
    psi_hat = np.zeros((mdp.d, mdp.N_S))  # (d, S)
    theta_hat = np.zeros((mdp.d,))  # (d,)
    from itertools import product
    for k, h in product(range(K), range(mdp.H)):
        s, a, r, next_s = dataset[k][h]
        phi_k = mdp.phi[s]  # (d,)
        psi_hat[:, next_s] += phi_k
        theta_hat += phi_k * r
        Lambda += phi_k[:, np.newaxis] * phi_k[np.newaxis, :]
    Lambda_inv = np.linalg.inv(Lambda)
    # Learn the emprical model
    psi = np.matmul(Lambda_inv, psi_hat)  # (d, S)
    theta = np.matmul(Lambda_inv, theta_hat)  # (d,)

    omega = np.zeros((mdp.d,))  # (d,)
    next_v = np.zeros((mdp.N_S,))  # (d,)
    for h in reversed(range(mdp.H)):
        for s in range(mdp.N_S):
            if h < mdp.H - 1:
                next_v[s] = np.maximum(np.dot(mdp.phi[s], omega), mdp.r[s, 1])
            else:
                next_v[s] = mdp.r[s, 1]
        for i in range(mdp.d):
            if np.abs(psi[i].sum()) < 1e-10:
                omega[i] = 0
            else:
                omega[i] = theta[i] + next_v.min() - opt.fminbound(
                    lambda b: g_with_d(b, psi[i], next_v - next_v.min(), rho), BETA_MIN, BETA_MAX, full_output=True)[1]
        omega = np.clip(omega, 0, mdp.r.max())
        omegas.append(omega.copy())
    omegas.reverse()
    return omegas


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
