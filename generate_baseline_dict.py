import numpy as np
import time
from rsma_env import UplinkRSMAEnv
from traditional_baselines import TraditionalBCDRSMA


def generate_baseline_table():
    print("🚀 开始离线生成 BCD 基线查找表 (0-30dB, 步长 0.1dB)...")

    # 初始化环境和求解器 (参数必须与你训练时完全一致)
    K, Nt, Nr = 4, 2, 4
    env = UplinkRSMAEnv(K=K, Nt=Nt, Nr=Nr)
    bcd_solver = TraditionalBCDRSMA(K, Nt, Nr, env.P_max)

    snr_grid = np.arange(-5, 35, 0.1)
    baseline_dict = {}

    # 每个 SNR 点做 20 次蒙特卡洛平均 (平衡时间与精度)
    num_mc_samples = 20

    start_total = time.time()

    for snr in snr_grid:
        snr_rounded = round(snr, 1)
        rates_at_snr = []

        for _ in range(num_mc_samples):
            # 强制环境在当前 SNR 下重置
            env.reset(options={'snr_db': snr_rounded})

            # 获取环境参数
            H_hat = env.H_hat
            epsilon = env.epsilon
            sigma2 = env.sigma2
            H_true = env.H_true

            # 使用传统算法求解并评估真实速率
            robust_min_rate, true_min_rate, _ = bcd_solver.optimize_and_evaluate(H_hat, epsilon, sigma2, H_true)
            rates_at_snr.append(robust_min_rate)

        # 记录平均性能，为了防止除以 0，加一个极小的保护值
        avg_rate = np.mean(rates_at_snr)
        baseline_dict[snr_rounded] = max(avg_rate, 1e-3)

        if snr_rounded % 5.0 == 0:
            print(f"✅ 已完成 SNR = {snr_rounded:4.1f} dB | Baseline Rate = {baseline_dict[snr_rounded]:.4f} bps/Hz")

    # 保存为 numpy 字典文件
    np.save('bcd_baseline_lut.npy', baseline_dict)
    print(f"🎉 查找表生成完毕！耗时: {(time.time() - start_total) / 60:.2f} 分钟。已保存至 'bcd_baseline_lut.npy'")


if __name__ == "__main__":
    generate_baseline_table()