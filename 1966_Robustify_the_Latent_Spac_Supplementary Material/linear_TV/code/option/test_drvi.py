from algos import DRVI
from mdp import OptionMDP, FixedPolicy, OptionLinearPolicy
from utils import generate_traj
import numpy as np
import time
from pathlib import Path
from algos import compute_penalty

import multiprocessing


# ==== test 函数：在给定 d 与数据集上训练 CROP-VI(TV) 并返回目标价值与耗时 ====
import time
import numpy as np


def test(t, d, dataset, rho=0.003, b=0.7, eps=0.2, delta=0.05):
    """
    参数：
      t      : 第几次重复（仅用于日志）
      d      : 线性特征维度
      dataset: 约束轨迹数据集，元素为 (s, a, r, g, next_s)
      rho    : TV 球半径
      b      : 约束阈值（希望 V_g >= b）
      eps    : 约束容忍（β = H / eps）
      delta  : 置信水平（用于 penalty 系数）
    返回：
      (objective_value, running_time)
      其中 objective_value 是“整流目标”在初始分布下的期望值（单位：按 r_scale 还原）
    """
    # 用目标维度 d 构建评测环境
    test_mdp = OptionMDP(d=d)

    # 计时开始
    start = time.perf_counter()

    # 训练（TV + penalty + constraint）
    weights_r, weights_g, misc = DRVI(
        test_mdp, dataset, rho=rho, b=b, eps=eps, delta=delta
    )

    # 构造带约束策略（悲观奖励 / 乐观约束）
    pol = OptionLinearPolicy(
        test_mdp, weights_r, weights_g,
        misc['Lambda_inv'], misc['gamma0'], misc['beta'], misc['b']
    )

    # 计时结束
    running_time = time.perf_counter() - start

    # 计算 h=0 时的“整流目标”价值 V(s,0)，对初始分布取期望
    V0 = np.array([pol.V(s, 0) for s in range(test_mdp.N_S)])
    # 若训练时对奖励做了归一化，这里乘回 r_scale（与 OptionMDP 保持一致）
    scale = getattr(test_mdp, "r_scale", 1.0)
    objective_value = float(np.dot(test_mdp.initial_state_dist, V0) * scale)

    print(f"finish: t={t}, d={d}, K={len(dataset)} | value={objective_value:.6f}, time={running_time:.3f}s")
    # Qr0 = test_mdp.phi @ weights_r[0] - np.array([compute_penalty(test_mdp.phi[s], misc['Lambda_inv'], misc['gamma0']) for s in range(test_mdp.N_S)])
    # Qr1 = test_mdp.r[:, 1]
    # print(f"[diag] at h=0: mean(Qr0)={Qr0.mean():.4f}, mean(Qr1)={Qr1.mean():.4f}, exercise_ratio={(Qr1>=Qr0).mean():.3f}")
    return objective_value, running_time



if __name__ == "__main__":
    np.random.seed(666)
    mdp = OptionMDP(d=31)
    # Dataset collection
    fixed_policy = FixedPolicy(mdp.N_S, mdp.N_A)
    ds = [31, 61, 121, 201, 301]
    Ks = [100, 200, 400, 800, 1600, 3200, 6400]
    num_tries = 20

    res20 = np.zeros((len(ds), len(Ks), num_tries))
    res21 = np.zeros((len(ds), len(Ks), num_tries))

    res = []

    p = multiprocessing.Pool(24)
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
                res20[i, j, t], res21[i, j, t] = res[k].get()
                k += 1

    print('finish all')
    OUT_DIR = Path(__file__).parent / "result"   # 与脚本同级的新建 result 文件夹
    OUT_DIR.mkdir(parents=True, exist_ok=True)   # 若不存在则创建
    print(res20)
    np.save(OUT_DIR / "res20.npy", res20)
    print(res21)
    np.save(OUT_DIR / "res21.npy", res21)
