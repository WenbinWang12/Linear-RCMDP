import numpy as np
import scipy.optimize as opt
from itertools import product
import argparse
import glob 
from multiprocessing import Pool

LOG_EPS = 1e-20
BETA_MIN = 1e-2
BETA_MAX = 1e2
data_prefix = '/home/liangzhp/dr_func_approx/envs/offline_data/'
params = {
        #   "acrobot": [6, 3], 
          "cartpole": [4, 2, 0.3, 512],
          "mountaincar": [2, 3, 1]
        }

offline_data_names = {"acrobot": "Acrobot", 
          "cartpole": "CartPole",
          "mountaincar": "MountainCar"}

#####################
# DRVI2, Tamar13'
#####################

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
        else:
            centers = np.load(center_files[0])
        self.centers = centers
        self.n_action = n_action

    def get_phi(self, state, action):
        feature = np.zeros((self.feature_dim,))
        base_shift = action * self.feature_dim // self.n_action
        try:
            for i, center in enumerate(self.centers):
                feature[base_shift + i] = self.gaussian_kernel(state, center)
        except:
            print('base_shift', base_shift, i)
        return feature / feature.sum()

    @staticmethod
    def gaussian_kernel(x1, x2, sigma=1):
        return np.exp(-np.sum((x1 - x2)**2) / (2 * sigma**2))
    

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


# def get_next_value(phi, Lambda_inv, Phi, V, rho):
#     V_max, V_min = V.max(), V.min()
#     if V_max == V_min:
#         return V_min
#     V = (V - V_min) / (V_max - V_min)
#     g = -opt.fminbound(lambda b: g_with_L(b, phi, Lambda_inv, Phi, V, rho), BETA_MIN, BETA_MAX, full_output=True)[1]
#     return V_min + g * (V_max - V_min)

def get_next_value(phi, Lambda_inv, Phi, V, rho):
    V_max, V_min = V.max(), V.min()
    if V_max == V_min:
        return V_min
    V = (V - V_min) / (V_max - V_min)
    g = -opt.fminbound(lambda b: g_with_L(b, phi, Lambda_inv, Phi, V, rho), BETA_MIN, BETA_MAX, full_output=True)[1]
    return V_min + g * (V_max - V_min)


def DRVI2(featurer, dataset, rho, lcb_coef=0., gamma=0.95):
    """
    Episodic variant of Tamer13'
    """
    state, action, next_state, reward, not_done = dataset
    K, d, n_action = len(state), featurer.feature_dim, featurer.n_action  # feature dimension
    reward_normed = reward * (1 - gamma)  # make sure value in [0, 1]
    Phi, Next_Phi, Lcb_meta_term,  = np.zeros((K, d)), np.zeros((K, n_action, d)), np.zeros((K, 1))
    Lambda = np.identity(d)
    R = lcb_coef

    # init Lambda
    for k in range(K):
        s, a = state[k], action[k]
        phi_k = featurer.get_phi(s, a)  # (d,)
        Phi[k] = phi_k
        Lambda += phi_k[:, np.newaxis] * phi_k[np.newaxis, :]
    Lambda_inv = np.linalg.inv(Lambda)
    np.save(model_prefix + f"DRVI2_Linv_rho({rho})_lcb_coef({lcb_coef})", Lambda_inv)

    ##### init next phi
    for k in range(K):
        r, next_s, nd = reward_normed[k], next_state[k], not_done[k]
        next_qs = []
        for next_a in range(n_action):
            next_phi = featurer.get_phi(next_s, next_a)
            # Lcb_meta_term[k] = np.sqrt(np.sum(np.power(next_phi,2) * np.diag(Lambda_inv)))
            Next_Phi[k][next_a] = next_phi
    print("Finish the preparation")

    omega = np.zeros((d,))  # (d,)
    omega_old = -np.ones((d,))  # (d,)
    for itr in range(200):
        NEXT_V = np.zeros((K,))
        worst_V = np.inf
        for k in range(K):
            r, next_s, nd = reward_normed[k], next_state[k], not_done[k]
            phi_k = Phi[k]  # (d,)
            next_v = np.dot(Next_Phi[k], omega)
            NEXT_V[k] = max(np.max(next_v), r)
            if worst_V>NEXT_V[k]:
                worst_V = NEXT_V[k]

            if worst_V>phi_k.dot(omega):
                worst_V = phi_k.dot(omega)

        # Next robust value
        target_v = np.zeros((d,))
        for k in range(K):
            r, nd = reward_normed[k], not_done[k]
            phi_k = Phi[k]  # (d,)
            # next_v = get_next_value(phi_k, Lambda_inv, Phi, NEXT_V, rho)
            next_v = R * worst_V + (1 - R) * NEXT_V[k]
            target_v += phi_k * (r + gamma * nd * next_v)

        omega = np.matmul(Lambda_inv, target_v)

        err = np.abs(omega - omega_old).max()
        print(f"round {itr} has error {err}")
        if err<1e-4:
            break
        if itr % 10:
            print(omega)
        omega = np.clip(omega, 0, 1)
        omega_old = omega.copy()
        np.save(model_prefix + f"DRVI2_omega_rho({rho})_lcb_coef({lcb_coef})", omega)    
    return omega


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DRVI2')
    parser.add_argument('--env')
    parser.add_argument('--feature_dim', default= 512)
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

    rhos, lcb_coefs = [0.01, 0.1, 0.2], [0.01, 0.02, 0.03]

    # DRVI(fc, dataset, rho = rhos[0], lcb_coef=lcb_coefs[0])
    fc = FeatureConstructor(env = env, feature_dim=feature_dim, state_dim = state_dim, n_action= n_action)

    # DRVI2(fc, dataset, rhos[0], lcb_coefs[0])

    with Pool(processes=num_process) as pool:
        pool.starmap(DRVI2, product([fc], [dataset], rhos, lcb_coefs))