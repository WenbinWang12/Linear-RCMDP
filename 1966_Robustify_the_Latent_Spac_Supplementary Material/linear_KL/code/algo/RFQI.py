import numpy as np
import scipy.optimize as opt
import time 
import glob
from multiprocessing import Pool
from itertools import product
import argparse

EPS = 1e-5
LOG_EPS = 1e-10
BETA_MIN = 1e-1
BETA_MAX = 1e2

feature_dim =  128 * 3
data_prefix = '/home/liangzhp/dr_func_approx/envs/offline_data/'
params = {"acrobot": [6, 3], 
          "cartpole": [4, 2],
          "mountaincar": [2, 3]}

offline_data_names = {"acrobot": "Acrobot", 
          "cartpole": "CartPole",
          "mountaincar": "MountainCar"}

class FeatureConstructor():
    def __init__(self, env, state_dim, n_action, centers=None, feature_dim=60):
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

    def get_phi(self, state, action):
        feature = np.zeros((self.feature_dim,))
        base_shift = action * self.feature_dim // self.n_action
        for i, center in enumerate(self.centers):
            feature[base_shift + i] = self.gaussian_kernel(state, center)
        return feature / feature.sum()

    @staticmethod
    def gaussian_kernel(x1, x2, sigma=1):
        return np.exp(-np.sum((x1 - x2)**2) / (2 * sigma**2))

def update_g_omega(K, lcb_coef, rho, omega, g_omega, Phi, action, Next_Phi, Lcb_meta_term, Phi_mean ,gamma = 0.95, eta = 10):
    Next_v = np.zeros((K, 1))
    for k in range(K):
        next_qs = []
        for next_a in range(n_action):
            next_phi = Next_Phi[k][next_a]
            lcb_meta_term = Lcb_meta_term[k][next_a]
            lcb_term = lcb_coef * lcb_meta_term
            next_qs.append(np.dot(next_phi, omega) - lcb_term)
        Next_v[k] = max(next_qs)
            
    g_omega_old = g_omega.copy()
    for itr in range(100):
        g_update = list()
        for k in range(K):
            a, next_v = action[k], Next_v[k]
            if np.clip(Phi[k] @ g_omega, 0, 2/(rho*(1-gamma))) > next_v:
                g_update.append(Phi[k][a])

        # g_update = np.asarray(g_update)
        if len(g_update):
            g_omega = g_omega - eta/((1 - 0.95) * itr + 1) * (len(g_update)/K * np.mean(g_update, axis = 0) - (1-rho) * Phi_mean).reshape((-1,1))
        else:
            g_omega = g_omega - eta/((1 - 0.95) * itr + 1) * (- (1-rho) * Phi_mean).reshape((-1,1))

        # print(f"g error {np.linalg.norm(g_omega_old - g_omega) / np.linalg.norm(g_omega)}")
        if np.linalg.norm(g_omega_old - g_omega) / (np.linalg.norm(g_omega) + EPS) < 1e-2:
            break

        g_omega_old = g_omega.copy()
    return Next_v, g_omega

def update_f_omega(gamma, rho, Phi, Next_v, rs, nds, omega, g_omega, Lambda_inv, eta = 10):
    K = len(Next_v)
    omega_old = omega.copy()
    for itr in range(10):
        TARGET_V = np.zeros((K,1))
        for k in range(K):
            r, next_v, nd = rs[k], Next_v[k], nds[k]
            TARGET_V[k] = r + gamma * nd * (- np.max(np.clip(Phi[k] @ g_omega, 0, 2/(rho*(1-gamma)))  - next_v, 0) + (1-rho) * np.clip(Phi[k] @ g_omega, 0, 2/(rho*(1-gamma))))

        omega = omega - eta/(itr + 1) * (Phi.T @ Phi @ omega - Phi.T @ TARGET_V)
        # omega = Lambda_inv @ (Phi.T @ TARGET_V)
        if np.linalg.norm(omega_old - omega) / (np.linalg.norm(omega) + EPS) < 1e-2:
            break
        omega_old = omega.copy()
    return omega

def RFQI(featurer, dataset, rho, lcb_coef=0.02, gamma=0.95, eta = 2):
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
    np.save(model_prefix + f"RFQI_Linv_rho({rho})_lcb_coef({lcb_coef})", Lambda_inv)
    Phi_mean = np.mean(Phi, axis = 0)
    ##### init next phi
    for k in range(K):
        r, next_s, nd = reward_normed[k], next_state[k], not_done[k]
        for next_a in range(n_action):
            next_phi = featurer.get_phi(next_s, next_a)
            Next_Phi[k][next_a] = nd * next_phi
            Lcb_meta_term[k][next_a] = nd * np.sqrt(np.sum(np.power(next_phi,2) * np.diag(Lambda_inv)))
            Next_Phi[k][next_a] = next_phi
    print("Finish the preparation")

    # start to fitting Q table
    omega, g_omega = np.zeros((d,1)), np.zeros((d,1))  # (d,)
    omega_old, g_omega_old = -np.ones((d,1)), -np.ones((d,1))  # (d,)
    for itr in range(30):
        temp = time.time()
        Next_v, g_omega = update_g_omega(K, lcb_coef, rho, omega, g_omega, Phi, action, Next_Phi, Lcb_meta_term, Phi_mean = Phi_mean, gamma =  gamma, eta = eta)
        omega = update_f_omega(gamma, rho, Phi, Next_v, reward, not_done, omega, g_omega, Lambda_inv, eta = 1/10)

        print(f"The iteration takes {time.time() - temp} s")
        omega = np.clip(omega, 0, 1)  # Suppose value in [0, 1]
        error, error_g = np.abs(omega - omega_old).max(), np.abs(g_omega - g_omega_old).max()

        print("itr =", itr, "error =", error)
        if itr % 10 == 0:
            print('-----f-----')
            print(omega)
            print('-----g-----')
            print(g_omega)
        if error < 1e-4 and error_g<1e-4:
            break

        omega_old, g_omega_old = omega.copy(), g_omega.copy()
    np.save(model_prefix + f"RFQI_omega_rho({rho})_lcb_coef({lcb_coef})", omega)    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RFQI')
    parser.add_argument('--env')
    parser.add_argument('--feature_dim', default= 512)
    parser.add_argument('--num_process', default= 3, type = int)
    parser.add_argument('--random_seed', default= 666)
    args = parser.parse_args()
    env, feature_dim, random_seed, num_process = args.env, args.feature_dim, args.random_seed, args.num_process
    model_prefix = f'/home/liangzhp/dr_func_approx/envs/{env}/models/'

    state_dim, n_action = params[env]

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

    fc = FeatureConstructor(env = env, feature_dim=feature_dim, state_dim = state_dim, n_action= n_action)

    RFQI(fc, dataset, rho = 0.1, lcb_coef = 0.03)
    # rhos, lcb_coefs = [0.01, 0.1, 0.2], [0, 0.01, 0.02, 0.03]

    # RFQI(fc, dataset, rhos[0], lcb_coef = lcb_coefs[0])
    # with Pool(processes=num_process) as pool:
    #     pool.starmap(RFQI, product([fc], [dataset], rhos, lcb_coefs))
