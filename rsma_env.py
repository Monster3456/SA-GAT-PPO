
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import scipy.linalg as la
from scipy.special import j0


class UplinkRSMAEnv(gym.Env):
    def __init__(self, K=8, Nt=4, Nr=8, P_max=1.0, fd=240, Ts=1e-3, snr_range=(-5, 35)):
        super(UplinkRSMAEnv, self).__init__()
        self.K, self.Nt, self.Nr = K, Nt, Nr
        self.P_max = P_max
        self.snr_range = snr_range  # 训练时的 SNR 随机化范围 (dB)

        self.rho_corr = j0(2 * np.pi * fd * Ts)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5 * self.K,), dtype=np.float32)

        # 状态空间扩维：H_hat 实部虚部 (2*Nr*Nt), epsilon (1), SNR_dB (1)
        # 每个节点特征从 33 维变成 34 维 (假设 Nt=4, Nr=8)
        obs_dim = (2 * self.Nr * self.Nt + 2) * self.K
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # ==========================================
        # 新增：加载离线基线查找表 (Look-Up Table)
        # ==========================================
        try:
            self.baseline_lut = np.load('bcd_baseline_lut.npy', allow_pickle=True).item()
            print("📦 成功加载 BCD 基线查找表！")
        except FileNotFoundError:
            print("⚠️ 未找到 bcd_baseline_lut.npy，将使用常数 1.0 作为 fallback (请先运行生成脚本)")
            self.baseline_lut = {}

    def _generate_rayleigh(self):
        return (np.random.randn(self.Nr, self.Nt) + 1j * np.random.randn(self.Nr, self.Nt)) / np.sqrt(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 1. 强制指定或随机 SNR
        if options is not None and 'snr_db' in options:
            self.current_snr_db = options['snr_db']
        else:
            self.current_snr_db = np.random.uniform(self.snr_range[0], self.snr_range[1])

        self.sigma2 = self.P_max / (10 ** (self.current_snr_db / 10.0))
        self.H_true = [self._generate_rayleigh() for _ in range(self.K)]

        # ==========================================
        # 核心修改：允许测试时强制固定 epsilon
        # ==========================================
        if options is not None and 'epsilon' in options:
            # 如果外部指定了 epsilon，则所有用户都使用这个误差界进行严格测试
            fixed_eps = options['epsilon']
            self.epsilon = np.ones(self.K) * fixed_eps
        else:
            # 训练时保持随机，增加网络泛化能力
            self.epsilon = np.random.uniform(0.05, 0.3, self.K)

        self._update_csi()
        return self._get_obs(), {}

    def _update_csi(self):
        self.H_hat = []
        for k in range(self.K):
            E_dir = self._generate_rayleigh()
            E = self.epsilon[k] * E_dir / la.norm(E_dir, 'fro')
            self.H_hat.append(self.H_true[k] + E)

    def _evolve_channel(self):
        for k in range(self.K):
            innovation = self._generate_rayleigh()
            self.H_true[k] = self.rho_corr * self.H_true[k] + np.sqrt(1 - self.rho_corr ** 2) * innovation
        self._update_csi()

    def _get_obs(self):
        # 将当前 SNR 归一化到 [0, 1] 附近
        snr_norm = (self.current_snr_db-self.snr_range[0] )/ (self.snr_range[1]-self.snr_range[0])

        obs_list = []
        # 核心修复：按“用户(Node)”维度，逐个封装特征
        for k in range(self.K):
            # 将每个用户独有的特征拼成一个一维向量
            user_features = np.concatenate([
                np.real(self.H_hat[k]).flatten(),  # 该用户的信道实部
                np.imag(self.H_hat[k]).flatten(),  # 该用户的信道虚部
                [self.epsilon[k]],  # 该用户的误差界 (转为数组)
                [snr_norm]  # 全局 SNR (转为数组)
            ])
            obs_list.append(user_features)

        # 最终拼接成 [User1_feats, User2_feats, User3_feats, User4_feats]
        obs = np.concatenate(obs_list)
        return obs.astype(np.float32)

    def step(self, action, eta=0.05):
        # 1. 动作映射
        action = np.clip(action, -1.0, 1.0)
        a_tilde = (action[0:self.K] + 1) / 2
        rho = (action[self.K:2 * self.K] + 1) / 2
        alpha = (action[2 * self.K:3 * self.K] + 1) / 2
        z_scores = action[3 * self.K:5 * self.K]

        a_k = (a_tilde > 0.5).astype(int)
        P_c = a_k * alpha * rho * self.P_max
        P_p = (1 - a_k * alpha) * rho * self.P_max

        mask = np.concatenate([a_k, np.ones(self.K)])
        masked_scores = z_scores * mask - 1e9 * (1 - mask)
        sic_order = np.argsort(masked_scores)[::-1]
        M = self.K + np.sum(a_k)
        active_sic = sic_order[:M]

        # 2. 预编码方向闭式解 (基于误差信道 H_hat)
        W_c = np.zeros((self.Nt, self.K), dtype=complex)
        W_p = np.zeros((self.Nt, self.K), dtype=complex)

        H_cat = np.concatenate(self.H_hat, axis=1)
        RZF_mat = H_cat.conj().T @ la.inv(H_cat @ H_cat.conj().T + self.sigma2 * np.eye(self.Nr, dtype=complex))

        for k in range(self.K):
            if a_k[k] == 1:
                U, S, Vh = la.svd(self.H_hat[k], full_matrices=False)
                W_c[:, k] = np.sqrt(P_c[k]) * Vh[0, :].conj()

            w_p_dir = RZF_mat[k * self.Nt: (k + 1) * self.Nt, k]
            norm_val = la.norm(w_p_dir)
            if norm_val > 1e-12:
                W_p[:, k] = np.sqrt(P_p[k]) * (w_p_dir / norm_val)

        # ==========================================
        # 3. 鲁棒 MMSE 接收与 SINR 计算 (基于 H_hat)
        # ==========================================
        rates_robust = np.zeros(2 * self.K)
        # 存储计算出的组合器，用于后续真实环境测试
        combiners = {}

        for idx in range(M):
            stream_id = active_sic[idx]
            user_idx = stream_id % self.K
            is_common = stream_id < self.K

            w_i = W_c[:, user_idx] if is_common else W_p[:, user_idx]
            H_i_hat = self.H_hat[user_idx]
            eps_i = self.epsilon[user_idx]

            R_interf_robust = self.sigma2 * np.eye(self.Nr, dtype=complex)

            for j_id in active_sic[idx + 1:]:
                j_user = j_id % self.K
                w_j = W_c[:, j_user] if j_id < self.K else W_p[:, j_user]
                H_j_hat = self.H_hat[j_user]
                eps_j = self.epsilon[j_user]
                R_interf_robust += H_j_hat @ np.outer(w_j, w_j.conj()) @ H_j_hat.conj().T
                R_interf_robust += (eps_j ** 2 * la.norm(w_j) ** 2) * np.eye(self.Nr)

            R_total_robust = R_interf_robust + H_i_hat @ np.outer(w_i, w_i.conj()) @ H_i_hat.conj().T + (
                        eps_i ** 2 * la.norm(w_i) ** 2) * np.eye(self.Nr)
            try:
                u_i = la.inv(R_total_robust) @ H_i_hat @ w_i
            except la.LinAlgError:
                u_i = np.zeros(self.Nr, dtype=complex)

            combiners[stream_id] = u_i

            signal_power_rob = np.abs(u_i.conj().T @ H_i_hat @ w_i) ** 2
            interf_power_rob = np.real(u_i.conj().T @ R_interf_robust @ u_i) + eps_i ** 2 * la.norm(u_i) ** 2 * la.norm(
                w_i) ** 2

            if interf_power_rob > 0:
                rates_robust[stream_id] = np.log2(1 + signal_power_rob / interf_power_rob)

        # ==========================================
        # 新增 4：真实物理空间速率计算 (基于 H_true)
        # ==========================================
        rates_true = np.zeros(2 * self.K)
        for idx in range(M):
            stream_id = active_sic[idx]
            user_idx = stream_id % self.K
            is_common = stream_id < self.K

            w_i = W_c[:, user_idx] if is_common else W_p[:, user_idx]
            u_i = combiners[stream_id]  # 直接使用基站算好的组合器
            H_i_true = self.H_true[user_idx]  # 真实的电磁波传播信道

            R_interf_true = self.sigma2 * np.eye(self.Nr, dtype=complex)
            for j_id in active_sic[idx + 1:]:
                j_user = j_id % self.K
                w_j = W_c[:, j_user] if j_id < self.K else W_p[:, j_user]
                H_j_true = self.H_true[j_user]
                R_interf_true += H_j_true @ np.outer(w_j, w_j.conj()) @ H_j_true.conj().T

            signal_power_true = np.abs(u_i.conj().T @ H_i_true @ w_i) ** 2
            interf_power_true = np.real(u_i.conj().T @ R_interf_true @ u_i)

            if interf_power_true > 0:
                rates_true[stream_id] = np.log2(1 + signal_power_true / interf_power_true)

        # 5. 汇总与奖励计算
        R_users_robust = [a_k[k] * rates_robust[k] + rates_robust[self.K + k] for k in range(self.K)]
        R_users_true = [a_k[k] * rates_true[k] + rates_true[self.K + k] for k in range(self.K)]

        min_rate_rob = np.min(R_users_robust)
        min_rate_true = np.min(R_users_true)

        # 纯粹的原始物理奖励 (Raw Reward)
        raw_reward = min_rate_rob + eta * np.sum(R_users_robust)

        # ==========================================
        # 终极防御：绝对就近查表 (解决浮点数丢失)
        # ==========================================
        closest_snr = min(self.baseline_lut.keys(), key=lambda k: abs(k - self.current_snr_db))
        baseline_rate = self.baseline_lut[closest_snr]

        # ==========================================
        # 终极塑形：平滑百分比增益 (Smoothed Percentage Gain)
        # 完美解决高低 SNR 下奖励尺度不统一与低 SNR 爆炸的问题
        # ==========================================
        delta = 1.0  # 速率平滑因子 (bps/Hz)
        reward = (raw_reward - baseline_rate) / (baseline_rate + delta)

        self._evolve_channel()
        return self._get_obs(), float(reward), False, False, {
            "rates_robust": R_users_robust,
            "min_rate_robust": min_rate_rob,
            "min_rate_true": min_rate_true,
            "split_decisions": a_k,
            "raw_reward": raw_reward
        }