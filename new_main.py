import time
from rsma_env import UplinkRSMAEnv
from ppo_agent import PPOAgent
from maddpg_baseline import LiteratureMADDPGAgent
# 核心修改：导入所有三种对比算法
from traditional_baselines import TraditionalBCDRSMA, TraditionalSDMA, EPARSMA
import scipy.linalg as la
from itertools import combinations
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np

def calculate_true_mac_symmetric_limit(H_true_list, P_max, sigma2, K, Nr):
    """
    计算真正的多用户 MAC 信道对称速率上限 (Max-Min Shannon Limit)
    通过遍历所有 2^K - 1 个子集，寻找系统的“瓶颈”子集。
    """
    # 假设每个用户使用最优的单流特征模传输 (对系统容量贡献最大)
    w_opt_list = []
    for k in range(K):
        H_k = H_true_list[k]
        U, S, Vh = la.svd(H_k, full_matrices=False)
        w_opt_list.append(Vh[0, :].conj() * np.sqrt(P_max))

    min_symmetric_rate = float('inf')

    # 遍历所有可能的子集 (例如 K=4 时，有 15 个子集)
    for subset_size in range(1, K + 1):
        for subset in combinations(range(K), subset_size):
            # 计算当前子集 S 的协方差矩阵
            R_subset = np.eye(Nr, dtype=complex)
            for k in subset:
                H_k = H_true_list[k]
                w_k = w_opt_list[k]
                R_subset += (1.0 / sigma2) * (H_k @ np.outer(w_k, w_k.conj()) @ H_k.conj().T)

            # 计算当前子集的容量 C(S)
            C_subset = np.real(np.log2(np.linalg.det(R_subset)))

            # 均分给子集内的用户: C(S) / |S|
            sym_rate_bound = C_subset / subset_size

            # 真正的系统对称极限由最严苛的那个子集决定
            if sym_rate_bound < min_symmetric_rate:
                min_symmetric_rate = sym_rate_bound

    return min_symmetric_rate




def test_epsilon_boxplot(env, agent, agent_mlp, agent_maddpg,bcd_solver):
    print("📦 开始生成 CSI 误差鲁棒性箱线图数据...")

    # 设定固定的测试条件
    fixed_snr = 20.0
    test_steps_per_eps = 100  # 每个箱子收集 100 个样本点
    epsilon_test_grid = [0.1, 0.2, 0.3]  # 轻、中、重三种误差

    # 用于存储所有数据的列表，方便转为 pandas DataFrame
    data_records = []

    for eps in epsilon_test_grid:
        print(f"正在测试 epsilon = {eps}...")

        for step in range(test_steps_per_eps):
            # 强制环境使用固定的 SNR 和 epsilon 重置
            state, _ = env.reset(options={'snr_db': fixed_snr, 'epsilon': eps})

            # --------------------------------------------------
            # 1. Proposed MAPPO (Transformer)
            # --------------------------------------------------
            state_ts = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                dist, _ = agent.policy(state_ts)
                action = dist.mean.squeeze(0).numpy()
            _, _, _, _, info = env.step(action, eta=0.0)
            data_records.append(
                {'Epsilon': f'$\epsilon={eps}$', 'Algorithm': 'SA-GAT PPO', 'Rate': info['min_rate_true']})


            state_ts_local = torch.FloatTensor(state_ts).view(1, env.K, agent_maddpg.obs_dim_per_agent)
            with torch.no_grad():
                action_maddpg = agent_maddpg.actor(state_ts_local).squeeze(0).numpy()
            state_maddpg, _, _, _, info_maddpg = env.step(action_maddpg, eta=0.0)
            data_records.append(
                {'Epsilon': f'$\epsilon={eps}$', 'Algorithm': 'MA-DDPG [23]', 'Rate': info_maddpg['min_rate_true']})

            # --------------------------------------------------
            # 2. MAPPO (Pure MLP)
            # --------------------------------------------------
            with torch.no_grad():
                dist_mlp, _ = agent_mlp.policy(state_ts)
                action_mlp = dist_mlp.mean.squeeze(0).numpy()
            _, _, _, _, info_mlp = env.step(action_mlp, eta=0.0)
            data_records.append(
                {'Epsilon': f'$\epsilon={eps}$', 'Algorithm': 'MLP-PPO [7]', 'Rate': info_mlp['min_rate_true']})

            # --------------------------------------------------
            # 3. Traditional BCD-RSMA
            # --------------------------------------------------
            _, true_rate_bcd, _ = bcd_solver.optimize_and_evaluate(env.H_hat, env.epsilon, env.sigma2,
                                                                   H_true=env.H_true)
            data_records.append({'Epsilon': f'$\epsilon={eps}$', 'Algorithm': 'BCD [21]', 'Rate': true_rate_bcd})

    # 将数据转为 DataFrame 供 seaborn 使用
    df = pd.DataFrame(data_records)

    # ==========================================
    # 绘制高级分组箱线图
    # ==========================================
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'Times New Roman'

    # 使用 seaborn 画图，调色板选择顶刊常用的冷暖对比色
    solid_palette = ["#335ca3", "#e9333f", "#993399", "#6c9c4d"]
    ax = sns.boxplot(x="Epsilon", y="Rate", hue="Algorithm", data=df,
                     palette=["#0033a0", "#e3000f", "#800080", "#478321"],  # 蓝，红，紫
                     width=0.6, boxprops=dict(alpha=0.8), fliersize=3)

    # 美化图表
    plt.xlabel('Imperfect CSI Error Bound', fontsize=14, fontweight='bold')
    plt.ylabel('Minimum User Rate (bps/Hz)', fontsize=14, fontweight='bold')

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(title='Scheduling Algorithm', title_fontsize='13', fontsize='12', loc='upper right')

    plt.tight_layout()
    plt.savefig('robustness_boxplot.png', dpi=600)
    plt.show()

def test_benchmark_snr(env, agent, test_steps_per_snr=500):
    print("🔬 开始顶刊级别 Benchmark 对比测试 (MAPPO vs Traditional BCD)...")

    agent.policy.load_state_dict(torch.load("rsma_ppo_transformer_weights.pth", weights_only=True))
    agent.policy.eval()

    agent_mlp = PPOAgent(env.K, env.Nt, env.Nr, env.action_space.shape[0], model_type='mlp')
    try:
        agent_mlp.policy.load_state_dict(torch.load("rsma_ppo_mlp_weights.pth", weights_only=True))
        agent_mlp.policy.eval()
        print("✅ 成功加载纯 MLP 权重用于消融对比！")
    except FileNotFoundError:
        print("⚠️ 未找到 rsma_ppo_mlp_weights.pth，请先训练 MLP 模型！")
        return
    # 3. 加载 文献基线 MA-DDPG
    agent_maddpg = LiteratureMADDPGAgent(env.K, env.Nt, env.Nr, env.action_space.shape[0])
    try:
        agent_maddpg.actor.load_state_dict(torch.load("rsma_maddpg_weights.pth", weights_only=True))
        agent_maddpg.actor.eval()
        print("✅ 成功加载文献 MA-DDPG 权重！")
    except FileNotFoundError:
        print("⚠️ 未找到 rsma_maddpg_weights.pth，请先训练  MADDPG 。")

    bcd_solver = TraditionalBCDRSMA(env.K, env.Nt, env.Nr, env.P_max)
    sdma_solver = TraditionalSDMA(env.K, env.Nt, env.Nr, env.P_max)
    epa_solver = EPARSMA(env.K, env.Nt, env.Nr, env.P_max)

    test_epsilon_boxplot(env,agent,agent_mlp,agent_maddpg,bcd_solver)
    snr_grid = [0, 5, 10, 15, 20, 25, 30]

    # 性能记录列表
    mappo_results, mlp_results,maddpg_results,shannon_results,bcd_results, sdma_results, epa_results = [],[],[], [], [], [],[]
    # 耗时记录列表
    mappo_times,mlp_times, maddpg_times,bcd_times, sdma_times, epa_times = [],[], [], [], [],[]

    for snr in snr_grid:
        print(f"\n--- 测试 SNR: {snr} dB ---")
        state, _ = env.reset(options={'snr_db': snr})

        mappo_rates_snr = []
        bcd_rates_snr = []
        sdma_rates_snr = []
        shannon_bound_snr = []
        epq_rates_snr = []

        time_mappo_snr = 0
        time_bcd_snr = 0
        time_sdma_snr = 0
        time_epa_snr = 0

        # 为了节约传统算法的龟速测试时间，每个 SNR 跑 50 个信道快照取平均
        for step in range(test_steps_per_snr):
            cap_k = calculate_true_mac_symmetric_limit(env.H_true, env.P_max, env.sigma2, env.K, env.Nr)
            shannon_bound_snr.append(cap_k)
            # --------------------------------------------------
            # 1. 我们的方法：MAPPO-RSMA (极速前向传播)
            # --------------------------------------------------
            t0 = time.time()
            state_ts = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                dist, _ = agent.policy(state_ts)
                action = dist.mean.squeeze(0).numpy()

            # 环境步进，获取 MAPPO 在真实信道中的物理速率
            state, _, _, _, info = env.step(action, eta=0.0)
            time_mappo_snr += (time.time() - t0)
            mappo_rates_snr.append(info['min_rate_true'])

            # --------------------------------------------------
            # 2. 对比方法：传统 BCD-RSMA (繁重的迭代优化)
            # --------------------------------------------------
            H_hat = env.H_hat
            epsilon = env.epsilon
            sigma2 = env.sigma2
            H_true = env.H_true

            # 调用传统迭代算法，传入 H_true 以计算其真实的物理层速率
            robust_rate_bcd, true_rate_bcd, opt_time = bcd_solver.optimize_and_evaluate(H_hat, epsilon, sigma2,
                                                                                        H_true=H_true)

            time_bcd_snr += opt_time
            # 直接将基于真实信道算出的速率加入统计，彻底抛弃此前的折算 hack！
            bcd_rates_snr.append(true_rate_bcd)

            # 3. 传统 SDMA (证明 RSMA 架构优越性)
            # --------------------------------------------------
            _, true_rate_sdma, opt_time_sdma = sdma_solver.optimize_and_evaluate(H_hat, epsilon, sigma2, H_true=H_true)
            time_sdma_snr += opt_time_sdma
            sdma_rates_snr.append(true_rate_sdma)

            # --------------------------------------------------
            # 4. EPA-RSMA (均分功率，证明 AI 调度的必要性)
            # --------------------------------------------------
            _, true_rate_epa, opt_time_epa = epa_solver.optimize_and_evaluate(H_hat, epsilon, sigma2, H_true=H_true)
            time_epa_snr += opt_time_epa
            epq_rates_snr.append(true_rate_epa)
        # -----------------------------------------------------------
        # 第二阶段：平行测试 MLP 消融基线
        # -----------------------------------------------------------
        state_mlp, _ = env.reset(options={'snr_db': snr})
        mlp_rates_snr = []
        time_mlp_snr = 0

        for step in range(test_steps_per_snr):
            t0 = time.time()
            state_ts = torch.FloatTensor(state_mlp).unsqueeze(0)
            with torch.no_grad():
                dist_mlp, _ = agent_mlp.policy(state_ts)
                action_mlp = dist_mlp.mean.squeeze(0).numpy()

            state_mlp, _, _, _, info_mlp = env.step(action_mlp, eta=0.0)
            time_mlp_snr += (time.time() - t0)
            mlp_rates_snr.append(info_mlp['min_rate_true'])

        # ==================================================
        # 第三阶段：平行测试 文献基线 MA-DDPG
        # ==================================================
        state_maddpg, _ = env.reset(options={'snr_db': snr})
        maddpg_rates_snr = []
        time_maddpg_snr = 0

        for step in range(test_steps_per_snr):
            t0 = time.time()
            # MADDPG Actor 需要局部状态视角 [batch, K, obs_dim_per_agent]
            state_ts_local = torch.FloatTensor(state_maddpg).view(1, env.K, agent_maddpg.obs_dim_per_agent)
            with torch.no_grad():
                action_maddpg = agent_maddpg.actor(state_ts_local).squeeze(0).numpy()
            state_maddpg, _, _, _, info_maddpg = env.step(action_maddpg, eta=0.0)
            time_maddpg_snr += (time.time() - t0)
            maddpg_rates_snr.append(info_maddpg['min_rate_true'])

        # 统计平均速率
        mappo_mean_rate = np.mean(mappo_rates_snr)
        mlp_results.append(np.mean(mlp_rates_snr))
        shannon_results.append(np.mean(shannon_bound_snr))
        maddpg_results.append(np.mean(maddpg_rates_snr))
        bcd_mean_rate = np.mean(bcd_rates_snr)  # 加上与 MAPPO 类似的安全裕度转为 True Rate

        mappo_results.append(mappo_mean_rate)
        bcd_results.append(bcd_mean_rate)
        sdma_results.append(np.mean(sdma_rates_snr))
        epa_results.append(np.mean(epq_rates_snr))

        avg_time_mappo = (time_mappo_snr / test_steps_per_snr) * 1000  # 毫秒
        avg_time_bcd = (time_bcd_snr / test_steps_per_snr) * 1000  # 毫秒
        maddpg_times.append((time_maddpg_snr / test_steps_per_snr) * 1000)
        sdma_times.append((time_sdma_snr / test_steps_per_snr) * 1000)
        epa_times.append((time_epa_snr / test_steps_per_snr) * 1000)

        mappo_times.append(avg_time_mappo)
        mlp_times.append((time_mlp_snr / test_steps_per_snr) * 1000)
        bcd_times.append(avg_time_bcd)

        print(f"MAPPO 平均速率: {mappo_mean_rate:.4f} bps/Hz | 平均耗时: {avg_time_mappo:.2f} ms")
        print(f"MAPPO (Pure MLP)   : {mlp_results[-1]:.4f} bps/Hz | 耗时: {mlp_times[-1]:.2f} ms")
        print(f"理论极限   : {shannon_results[-1]:.4f} bps/Hz")
        print(f"Literature MA-DDPG : {maddpg_results[-1]:.4f} bps/Hz | 耗时: {maddpg_times[-1]:.2f} ms")
        print(f"BCD   平均速率: {bcd_mean_rate:.4f} bps/Hz | 平均耗时: {avg_time_bcd:.2f} ms")
        print(f"SDMA     : {sdma_results[-1]:.4f} bps/Hz | 耗时: {sdma_times[-1]:.2f} ms")
        print(f"EPA-RSMA : {epa_results[-1]:.4f} bps/Hz | 耗时: {epa_times[-1]:.2f} ms")
        print(f"-> 速度提升: {avg_time_bcd / avg_time_mappo:.1f} 倍！")

    # ==========================================
    # 绘制核心卖点 1：五大算法性能对比图 (包含消融)
    # ==========================================
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(10, 7))
    plt.plot(snr_grid, mappo_results, marker='o', color='b', linewidth=2.5, markersize=8,
             label='SA-GAT PPO')
    # plt.plot(snr_grid, shannon_results, marker='*', color='black', linewidth=3.0, markersize=10, linestyle='-',
    #          label='Theoretical Limit (Interference-Free Shannon Capacity)')
    plt.plot(snr_grid, mlp_results, marker='v', color='m', linewidth=2.0, markersize=8, linestyle='-',
             label='MLP-PPO [7]')
    plt.plot(snr_grid, bcd_results, marker='s', color='r', linewidth=2.5, markersize=8, linestyle='--',
             label='BCD [21]')
    plt.plot(snr_grid, maddpg_results, marker='X', color='c', linewidth=2.0, markersize=8, linestyle='-',
             label='MA-DDPG [23]')
    plt.plot(snr_grid, sdma_results, marker='^', color='g', linewidth=2.5, markersize=8, linestyle='-.',
             label='SDMA [2]')
    plt.plot(snr_grid, epa_results, marker='d', color='darkorange', linewidth=2.5, markersize=8, linestyle=':',
             label='EPA-RSMA')

    plt.xlabel('SNR (dB)', fontsize=13)
    plt.ylabel('Minimum User Rate (bps/Hz)', fontsize=13)
    plt.xticks(snr_grid, fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    plt.savefig('performance_comparison_5curves.png', dpi=300)
    plt.show()

    # ==========================================
    # 绘制核心卖点 2：计算复杂度 (耗时) 对比图
    # ==========================================
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(10, 7))
    plt.plot(snr_grid, mappo_times, marker='o', color='b', linewidth=2.5, markersize=8,
             label='SA-GAT PPO')
    plt.plot(snr_grid, mlp_times, marker='v', color='m', linewidth=2.0, markersize=8, linestyle='-',
             label='MLP-PPO [7]')
    plt.plot(snr_grid, maddpg_times, marker='X', color='c', linewidth=2.0, markersize=8, linestyle='-',
             label='MA-DDPG [23]')
    plt.plot(snr_grid, bcd_times, marker='s', color='r', linewidth=2.5, markersize=8, linestyle='--',
             label='BCD [21]')
    plt.plot(snr_grid, sdma_times, marker='^', color='g', linewidth=2.0, markersize=7, linestyle='-.',
             label='SDMA [2]')
    plt.plot(snr_grid, epa_times, marker='d', color='darkorange', linewidth=2.0, markersize=7, linestyle=':',
             label='EPA-RSMA')

    plt.yscale('log')
    plt.xlabel('SNR (dB)', fontsize=13)
    plt.ylabel('Average Execution Time per Slot (ms)', fontsize=13)
    plt.xticks(snr_grid, fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.legend(fontsize=12, loc='center right')
    plt.tight_layout()
    plt.savefig('complexity_comparison_5curves.png', dpi=300)
    plt.show()


if __name__ == "__main__":

    env = UplinkRSMAEnv(K=4, Nt=2, Nr=4)
    agent = PPOAgent(K=4, Nt=2, Nr=4, action_dim=env.action_space.shape[0])

    test_benchmark_snr(env, agent)