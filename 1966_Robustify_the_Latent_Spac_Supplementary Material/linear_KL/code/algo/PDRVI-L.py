import numpy as np
import scipy.optimize as opt
import time 
import glob
from multiprocessing import Pool
from itertools import product
import argparse

LOG_EPS = 1e-10
BETA_MIN = 1e-1
BETA_MAX = 1e2

data_prefix = '/home/liangzhp/dr_func_approx/envs/offline_data/'
params = {
        #   "acrobot": [6, 3], 
          "cartpole": [4, 2, 0.3, 256],
          "mountaincar": [2, 3, 1]
        }

offline_data_names = {"acrobot": "Acrobot", 
          "cartpole": "CartPole",
          "mountaincar": "MountainCar"}

class FeatureConstructor():
    def __init__(self, env, state_dim, n_action, centers=None, feature_dim=60, sigma = 0.3):
        self.feature_dim = feature_dim
        center_files = list(glob.glob(f"envs/{env}/models/centers.npy"))
        print(center_files)
        if len(center_files) == 0:
            '''
            first random select some representative states in the state space
            each feature position is the gaussian distance from the state to the representative states
            '''
            centers = np.random.uniform(low=-1, high=1, size=(feature_dim // n_action, state_dim))
            np.save(f"envs/{env}/models/centers", centers)
            print('---save centers---')
        else:
            centers = np.load(center_files[0])
        self.centers = centers
        self.n_action = n_action
        self.sigma = sigma

    def get_phi(self, state, action):
        feature = np.zeros((self.feature_dim,))
        base_shift = action * self.feature_dim // self.n_action
        for i, center in enumerate(self.centers):
            feature[base_shift + i] = self.gaussian_kernel(state, center)
        return feature / feature.sum()

    # @staticmethod
    def gaussian_kernel(self, x1, x2):
        return np.exp(-np.sum((x1 - x2)**2) / (2 * self.sigma**2))


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
    ###### initializaiton
    state, action, next_state, reward, not_done = dataset
    reward_normed = reward * (1 - gamma)  # make sure value in [0, 1]
    K, d, n_action = len(state), featurer.feature_dim, featurer.n_action  # feature dimension
    Lambda = np.identity(d)
    Phi, Next_Phi, Lcb_meta_term = np.zeros((K, d)), np.zeros((K, n_action, d)), np.zeros((K, n_action, 1))

    ##### init Lambda_inv
    for k in range(K):
        s, a = state[k], action[k]
        phi_k = featurer.get_phi(s, a)  # (d,)
        Phi[k] = phi_k
        Lambda += phi_k[:, np.newaxis] * phi_k[np.newaxis, :]
    
    Lambda_inv = np.linalg.inv(Lambda)
    Lambda_inv_diag = np.diag(Lambda_inv)
    np.save(model_prefix + f"PDRVI_Linv_rho({rho})_lcb_coef({lcb_coef})", Lambda_inv)

    ##### init next phi
    for k in range(K):
        r, next_s, nd = reward_normed[k], next_state[k], not_done[k]
        next_qs = []
        for next_a in range(n_action):
            next_phi = featurer.get_phi(next_s, next_a)
            Next_Phi[k][next_a] = nd * next_phi
            Lcb_meta_term[k][next_a] = nd * np.sqrt(np.sum(np.power(next_phi,2) * Lambda_inv_diag))

    print("Finish the preparation")
    # start to update omega
    omega = np.zeros((d,))  # (d,)
    omega_old = -np.ones((d,))  # (d,)
    for itr in range(200):
        TARGET_V = np.zeros((K,))
        temp = time.time()
        for k in range(K):
            next_qs = []
            for next_a in range(n_action):
                next_phi = Next_Phi[k][next_a]
                lcb_term = Lcb_meta_term[k][next_a]
                lcb_term = lcb_coef * lcb_term
                next_qs.append(np.dot(next_phi, omega) - lcb_term)
            next_v = max(next_qs)
            TARGET_V[k] = r + gamma * next_v
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
    np.save(model_prefix + f"PDRVI_omega_rho({rho})_lcb_coef({lcb_coef})", omega)
    return None


def VI(featurer, dataset, gamma=0.95, lcb_coef=0.3):
    state, action, next_state, reward, not_done = dataset
    reward_normed = reward * (1 - gamma)  # make sure value in [0, 1]
    K = len(state)
    d, n_action = featurer.feature_dim, featurer.n_action  # feature dimension

    Lambda = np.identity(d)
    Phi = np.zeros((K, d))
    for k in range(K):
        s, a = state[k], action[k]
        phi_k = featurer.get_phi(s, a)  # (d,)
        Phi[k] = phi_k
        Lambda += phi_k[:, np.newaxis] * phi_k[np.newaxis, :]
    Lambda_inv = np.linalg.inv(Lambda)
    Lambda_inv_diag = np.diag(Lambda_inv)
    np.save("./models/Linv_nonrobust", Lambda_inv)
    print("Finish the preparation")

    omega = np.zeros((d,))  # (d,)
    omega_old = -np.ones((d,))  # (d,)
    for itr in range(200):
        TARGET_V = np.zeros((K,))
        for k in range(K):
            r, next_s, nd = reward_normed[k], next_state[k], not_done[k]
            next_qs = []
            for next_a in range(n_action):
                next_phi = featurer.get_phi(next_s, next_a)
                lcb_term = np.sqrt(np.sum(np.power(next_phi,2) * Lambda_inv_diag)) # next_phi[np.newaxis, :] @ Lambda_inv @ next_phi[:, np.newaxis]
                lcb_term = lcb_coef * lcb_term # np.sqrt (lcb_term[0, 0])
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
    parser = argparse.ArgumentParser(description='PDRVI-L')
    parser.add_argument('--env')
    parser.add_argument('--feature_dim', default= 512) # 128 * 3
    parser.add_argument('--num_process', default= 3, type = int)
    parser.add_argument('--random_seed', default= 666)
    args = parser.parse_args()
    env, feature_dim, random_seed, num_process = args.env, args.feature_dim, args.random_seed, args.num_process
    model_prefix = f'/home/liangzhp/dr_func_approx/envs/{env}/models/'
    state_dim, n_action, sigma, feature_dim = params[env]

    np.random.seed(random_seed)
    print(f"searching for ", f"envs/offline_data/{offline_data_names[env]}*.npy")
    prefix = '_'.join(sorted(list(glob.glob(f"envs/offline_data/{offline_data_names[env]}*.npy")))[0].split('_')[:-1])
    state_name, action_name, next_state_name, reward_name, not_done_name = prefix+ '_state.npy', \
                                                                            prefix+ '_action.npy', \
                                                                            prefix+ '_next_state.npy', \
                                                                            prefix+ '_reward.npy', \
                                                                            prefix+ '_not_done.npy'
    state, action, next_state, reward, not_done = np.load(state_name), np.load(action_name), np.load(next_state_name), \
                                                    np.load(reward_name), np.load(not_done_name)
    if reward.max() - reward.min()>0:
        reward = (reward - reward.min())/(reward.max() - reward.min())
    else:
        reward = 1 + (reward - reward.min())

    ind = np.random.choice(len(state), 10000)
    dataset = (state[ind], action[ind], next_state[ind], reward[ind], not_done[ind])

    fc = FeatureConstructor(env = env, feature_dim=feature_dim, state_dim = state_dim, n_action= n_action, sigma = sigma)

    rhos, lcb_coefs = [0.01, 0.1, 0.2], [0.01, 0.02, 0.03]
    # for rho in [0.01, 0.1, 1.0]:
    #     DRVI(fc, dataset, rho = rho, lcb_coef=0.03)

    with Pool(processes=num_process) as pool:
        pool.starmap(DRVI, product([fc], [dataset], rhos, lcb_coefs))

    # VI(fc, dataset, rhos[0], lcb_coef = lcb_coefs[0])