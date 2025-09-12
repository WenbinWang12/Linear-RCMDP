import numpy as np
import scipy.optimize as opt
import time 
from multiprocessing import Pool
from itertools import product
LOG_EPS = 1e-10
BETA_MIN = 1e-1
BETA_MAX = 1e2
data_prefix = '/home/liangzhp/dr_func_approx/envs/offline_data/'
model_prefix = '/home/liangzhp/dr_func_approx/envs/cartpole/models/'

class FeatureConstructor():
    def __init__(self, centers=None, feature_dim=60):
        self.feature_dim = feature_dim
        if centers is None:
            '''
            first random select some representative states in the state space
            each feature position is the gaussian distance from the state to the representative states
            '''
            centers = np.random.uniform(low=-1, high=1, size=(feature_dim // 2, 4))
            np.save("cartpole/models/centers", centers)
        else:
            centers = np.load(centers)
        self.centers = centers

    def get_phi(self, state, action):
        feature = np.zeros((self.feature_dim // 2,))
        for i, center in enumerate(self.centers):
            feature[i] = self.gaussian_kernel(state, center)
        if action == 0:
            feature = np.hstack([feature, np.zeros((self.feature_dim // 2,))])
        else:
            feature = np.hstack([np.zeros((self.feature_dim // 2,)), feature])
        return feature / feature.sum()

    @staticmethod
    def gaussian_kernel(x1, x2, sigma=0.3):
        return np.exp(-np.sum((x1 - x2)**2) / (2 * sigma**2))


def g_with_L(beta, index, Lambda_inv, Phi, V, rho):
    '''
    beta: scalar
    index: int
    Lambda_inv: (d, d)
    Phi: (K, d)
    V: (K,)
    '''
    EV = np.expm1(-(V / beta))  # (K,)
    y = np.sum(Phi * EV[:, np.newaxis], axis=0)  # (d,)
    w = np.matmul(Lambda_inv, y).clip(np.expm1(-1 / beta), 0)  # (d,)
    w = w[index]
    return beta * rho + beta * np.log1p(w + LOG_EPS)


def get_next_value(index, Lambda_inv, Phi, V, rho):
    V_max, V_min = V.max(), V.min()
    if V_max == V_min:
        return V_min
    V = (V - V_min) / (V_max - V_min)
    g = -opt.fminbound(lambda b: g_with_L(b, index, Lambda_inv, Phi, V, rho), BETA_MIN, BETA_MAX, full_output=True)[1]
    return V_min + g * (V_max - V_min)


def DRVI(featurer, dataset, rho, lcb_coef=0.02, gamma=0.95):
    state, action, next_state, reward, not_done = dataset
    reward_normed = reward * (1 - gamma)  # make sure value in [0, 1]
    K = len(state)
    d = featurer.feature_dim  # feature dimension

    Lambda = np.identity(d)
    Phi = np.zeros((K, d))
    for k in range(K):
        s, a = state[k], action[k]
        phi_k = featurer.get_phi(s, a)  # (d,)
        Phi[k] = phi_k
        Lambda += phi_k[:, np.newaxis] * phi_k[np.newaxis, :]

    # def compute_Lambda(input):
    #     s, a = input[0], input[1]
    #     phi_k = featurer.get_phi(s, a)
    #     Lambda_k = phi_k[:, np.newaxis] * phi_k[np.newaxis, :]
    #     return Lambda_k
    
    # Lambda = np.sum(np.vectorize(compute_Lambda)(zip(state, action)))
    
    Lambda_inv = np.linalg.inv(Lambda)
    np.save(model_prefix + f"Linv_rho({rho})_lcb_coef({lcb_coef})", Lambda_inv)
    print("Finish the preparation")

    omega = np.zeros((d,))  # (d,)
    omega_old = -np.ones((d,))  # (d,)
    one_vecs = np.identity(d)
    for itr in range(200):
        TARGET_V = np.zeros((K,))
        temp = time.time()
        for k in range(K):
            r, next_s, nd = reward_normed[k], next_state[k], not_done[k]
            next_qs = []
            for next_a in range(2):
                next_phi = featurer.get_phi(next_s, next_a)
                # lcb_term = next_phi[np.newaxis, :] @ Lambda_inv @ next_phi[:, np.newaxis] # could change to our lcb term
                lcb_term = np.sum(np.power(next_phi,2) * np.diag(Lambda_inv)) # np.sum([(next_phi[i]*one_vecs[i]) @ Lambda_inv @ (next_phi[i]*one_vecs[i].T) for i in range(d)])
                lcb_term = lcb_coef * np.sqrt(lcb_term)
                next_qs.append(np.dot(next_phi, omega) - lcb_term)
            next_v = max(next_qs)
            TARGET_V[k] = r + gamma * nd * next_v
        print(f"The iteration takes {time.time() - temp} s")

        temp = time.time()
        for i in range(d):
            # Next robust value
            omega[i] = get_next_value(i, Lambda_inv, Phi, TARGET_V, rho)
        print(f"The second iteration takes {time.time() - temp} s")

        omega = np.clip(omega, 0, 1)  # Suppose value in [0, 1]
        error = np.abs(omega - omega_old).max()

        print("itr =", itr, "error =", error)
        if itr % 10 == 0:
            print(omega)
        if error < 1e-4:
            break

        omega_old = omega.copy()
    np.save(model_prefix + f"omega_rho({rho})_lcb_coef({lcb_coef})", omega)    
    return None


def VI(featurer, dataset, gamma=0.95, lcb_coef=0.3):
    state, action, next_state, reward, not_done = dataset
    reward_normed = reward * (1 - gamma)  # make sure value in [0, 1]
    K = len(state)
    d = featurer.feature_dim  # feature dimension

    Lambda = np.identity(d)
    Phi = np.zeros((K, d))
    for k in range(K):
        s, a = state[k], action[k]
        phi_k = featurer.get_phi(s, a)  # (d,)
        Phi[k] = phi_k
        Lambda += phi_k[:, np.newaxis] * phi_k[np.newaxis, :]
    Lambda_inv = np.linalg.inv(Lambda)
    np.save("./models/Linv_nonrobust", Lambda_inv)
    print("Finish the preparation")

    omega = np.zeros((d,))  # (d,)
    omega_old = -np.ones((d,))  # (d,)
    for itr in range(200):
        TARGET_V = np.zeros((K,))
        for k in range(K):
            r, next_s, nd = reward_normed[k], next_state[k], not_done[k]
            next_qs = []
            for next_a in range(2):
                next_phi = featurer.get_phi(next_s, next_a)
                lcb_term = next_phi[np.newaxis, :] @ Lambda_inv @ next_phi[:, np.newaxis]
                lcb_term = lcb_coef * np.sqrt(lcb_term[0, 0])
                next_qs.append(np.dot(next_phi, omega) - lcb_term)
            next_v = max(next_qs)
            TARGET_V[k] = r + gamma * nd * next_v

        y = np.sum(Phi * TARGET_V[:, np.newaxis], axis=0)  # (d,)
        omega = np.matmul(Lambda_inv, y)
        omega = np.clip(omega, 0, 1)  # Suppose value in [0, 1]
        error = np.abs(omega - omega_old).max()

        print("itr =", itr, "error =", error)
        # if itr % 5 == 0:
        #     print(omega)
        if error < 1e-4:
            break

        omega_old = omega.copy()

    np.save(model_prefix + f"omega_nonrobust)", omega)
    return None


if __name__ == "__main__":
    np.random.seed(666)

    state = np.load(data_prefix+"CartPole-v0_ppo_e0.3_state.npy")
    action = np.load(data_prefix+"CartPole-v0_ppo_e0.3_action.npy")
    next_state = np.load(data_prefix+"CartPole-v0_ppo_e0.3_next_state.npy")
    reward = np.load(data_prefix+"CartPole-v0_ppo_e0.3_reward.npy")
    not_done = np.load(data_prefix+"CartPole-v0_ppo_e0.3_not_done.npy")

    ind = np.random.choice(len(state), 10000)
    dataset = (state[ind], action[ind], next_state[ind], reward[ind], not_done[ind])

    fc = FeatureConstructor(feature_dim=512, centers='/home/liangzhp/dr_func_approx/envs/cartpole/models/centers.npy')
    # for rho in [0.01, 0.1]:
        # print("rho =", rho)
    rhos, lcb_coefs = [0.01, 0.1, 0.2], [0.02, 0.03, 0.01]
    with Pool(processes=3) as pool:
        pool.starmap(DRVI, product([fc], [dataset], rhos, lcb_coefs))

    print("Nonrobust")
    VI(fc, dataset)
    # np.save(model_prefix + "omega_nonrobust", w)
