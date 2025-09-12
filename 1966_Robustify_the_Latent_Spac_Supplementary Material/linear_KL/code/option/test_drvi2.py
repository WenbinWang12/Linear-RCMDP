from algos import DRVI2
from mdp import OptionMDP, FixedPolicy, OptionLinearPolicy
from utils import generate_traj
import numpy as np
import time

import multiprocessing


def test(t, d, dataset):
    test_mdp = OptionMDP(d=d)
    start = time.perf_counter()
    rob_police = OptionLinearPolicy(test_mdp, DRVI2(test_mdp, dataset, 0.01))
    running_time = time.perf_counter() - start
    V_rob = np.array([rob_police.V(s, 0) for s in range(test_mdp.N_S)])
    rob_value = np.dot(test_mdp.initial_state_dist, V_rob) * mdp.r_scale
    print("finish, t={}, d={}, K={}".format(t, test_mdp.d, len(dataset)))
    return rob_value, running_time


if __name__ == "__main__":
    np.random.seed(666)
    mdp = OptionMDP(d=31)
    # Dataset collection
    fixed_policy = FixedPolicy(mdp.N_S, mdp.N_A)
    ds = [31, 61, 121, 201, 301]
    Ks = [100, 200, 400, 800, 1600, 3200, 6400]
    num_tries = 20
    # ds = [61]
    # Ks = [100, 200, 400, 800, 1600, 3200, 6400]
    # num_tries = 20

    res30 = np.zeros((len(ds), len(Ks), num_tries))
    res31 = np.zeros((len(ds), len(Ks), num_tries))

    res = []

    p = multiprocessing.Pool(128)
    p_conn, c_conn = multiprocessing.Pipe()

    for t in range(num_tries):
        dataset = [generate_traj(mdp, fixed_policy) for _ in range(int(1e4))]
        for i, d in enumerate(ds):
            for j, K in enumerate(Ks):
                print(f"start, t={t}, d={d}, K={K}")
                res.append(p.apply_async(test, (t, d, dataset[:K].copy())))
    p.close()
    p.join()

    print('output:')
    while p_conn.poll():
        print(p_conn.recv())
    k = 0
    for t in range(num_tries):
        for i, d in enumerate(ds):
            for j, K in enumerate(Ks):
                res30[i, j, t], res31[i, j, t] = res[k].get()
                k += 1

    print('finish all')
    np.save("result/res30.npy", res30)
    np.save("result/res31.npy", res31)
