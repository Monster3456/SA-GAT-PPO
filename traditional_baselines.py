import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
import time


class TraditionalBCDRSMA:
    def __init__(self, K, Nt, Nr, P_max):
        self.K = K
        self.Nt = Nt
        self.Nr = Nr
        self.P_max = P_max

    def optimize_and_evaluate(self, H_hat, epsilon, sigma2, H_true=None):
        """
        基于 SLSQP 的传统迭代优化，并严格在真实信道下评估最终性能
        """
        start_time = time.time()

        # 1. 预编码方向 (基于估计信道 H_hat)
        W_c_dir = np.zeros((self.Nt, self.K), dtype=complex)
        W_p_dir = np.zeros((self.Nt, self.K), dtype=complex)

        H_cat = np.concatenate(H_hat, axis=1)
        RZF_mat = H_cat.conj().T @ la.inv(H_cat @ H_cat.conj().T + sigma2 * np.eye(self.Nr, dtype=complex))

        for k in range(self.K):
            U, S, Vh = la.svd(H_hat[k], full_matrices=False)
            W_c_dir[:, k] = Vh[0, :].conj()

            w_p = RZF_mat[k * self.Nt: (k + 1) * self.Nt, k]
            norm_val = la.norm(w_p)
            if norm_val > 1e-12:
                W_p_dir[:, k] = w_p / norm_val

        # 传统 Baseline 假设固定的解码顺序 (先解公共流，再解私有流)
        active_sic = list(range(self.K)) + list(range(self.K, 2 * self.K))

        # 辅助函数：根据给定功率，计算鲁棒组合器 u_i 以及鲁棒速率
        def compute_robust_combiners_and_rate(powers):
            P_c = powers[:self.K]
            P_p = powers[self.K:]

            W_c = W_c_dir * np.sqrt(P_c)[None, :]
            W_p = W_p_dir * np.sqrt(P_p)[None, :]

            rates = np.zeros(2 * self.K)
            combiners = {}  # 必须保存算出来的组合器，用于后续真实信道验证

            for idx, stream_id in enumerate(active_sic):
                user_idx = stream_id % self.K
                is_common = stream_id < self.K

                w_i = W_c[:, user_idx] if is_common else W_p[:, user_idx]
                H_i_hat = H_hat[user_idx]
                eps_i = epsilon[user_idx]

                R_interf = sigma2 * np.eye(self.Nr, dtype=complex)
                for j_id in active_sic[idx + 1:]:
                    j_user = j_id % self.K
                    w_j = W_c[:, j_user] if j_id < self.K else W_p[:, j_user]
                    H_j_hat = H_hat[j_user]
                    eps_j = epsilon[j_user]
                    R_interf += H_j_hat @ np.outer(w_j, w_j.conj()) @ H_j_hat.conj().T
                    R_interf += (eps_j ** 2 * la.norm(w_j) ** 2) * np.eye(self.Nr)

                R_total = R_interf + H_i_hat @ np.outer(w_i, w_i.conj()) @ H_i_hat.conj().T + (
                            eps_i ** 2 * la.norm(w_i) ** 2) * np.eye(self.Nr)
                try:
                    u_i = la.inv(R_total) @ H_i_hat @ w_i
                except la.LinAlgError:
                    u_i = np.zeros(self.Nr, dtype=complex)

                combiners[stream_id] = u_i

                signal_power = np.abs(u_i.conj().T @ H_i_hat @ w_i) ** 2
                interf_power = np.real(u_i.conj().T @ R_interf @ u_i) + eps_i ** 2 * la.norm(u_i) ** 2 * la.norm(
                    w_i) ** 2

                if interf_power > 0:
                    rates[stream_id] = np.log2(1 + signal_power / interf_power)

            R_users = [rates[k] + rates[self.K + k] for k in range(self.K)]
            return np.min(R_users), W_c, W_p, combiners

        # 优化目标：基站只能优化其能算得出的鲁棒下限
        def objective(powers):
            min_rate, _, _, _ = compute_robust_combiners_and_rate(powers)
            return -min_rate

        # 约束与边界
        constraints = []
        for k in range(self.K):
            def power_constraint(powers, idx=k):
                return self.P_max - (powers[idx] + powers[self.K + idx])

            constraints.append({'type': 'ineq', 'fun': power_constraint})

        bounds = [(0, self.P_max) for _ in range(2 * self.K)]
        p0 = np.ones(2 * self.K) * (self.P_max / 2)  # 从均分功率开始搜

        # 执行 SLSQP 求解 (模拟 BCD 迭代过程)
        res = minimize(objective, p0, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 50, 'ftol': 1e-3})
        opt_time = time.time() - start_time

        # 提取优化后的最差鲁棒速率、最终预编码矩阵、最终组合器
        robust_min_rate, W_c_opt, W_p_opt, combiners_opt = compute_robust_combiners_and_rate(res.x)

        # =======================================================
        # 核心修复：将优化出的变量代入真实的物理信道 H_true 中进行验证
        # =======================================================
        true_min_rate = robust_min_rate  # 默认保底
        if H_true is not None:
            true_rates = np.zeros(2 * self.K)
            for idx, stream_id in enumerate(active_sic):
                user_idx = stream_id % self.K
                is_common = stream_id < self.K

                # 基站优化好的预编码和组合器
                w_i = W_c_opt[:, user_idx] if is_common else W_p_opt[:, user_idx]
                u_i = combiners_opt[stream_id]

                # 真实的电磁波信道 (无误差)
                H_i_true = H_true[user_idx]

                R_interf_true = sigma2 * np.eye(self.Nr, dtype=complex)
                for j_id in active_sic[idx + 1:]:
                    j_user = j_id % self.K
                    w_j = W_c_opt[:, j_user] if j_id < self.K else W_p_opt[:, j_user]
                    H_j_true = H_true[j_user]
                    R_interf_true += H_j_true @ np.outer(w_j, w_j.conj()) @ H_j_true.conj().T

                signal_power_true = np.abs(u_i.conj().T @ H_i_true @ w_i) ** 2
                interf_power_true = np.real(u_i.conj().T @ R_interf_true @ u_i)

                if interf_power_true > 0:
                    true_rates[stream_id] = np.log2(1 + signal_power_true / interf_power_true)

            True_R_users = [true_rates[k] + true_rates[self.K + k] for k in range(self.K)]
            true_min_rate = np.min(True_R_users)  # 这才是真正的测试 Benchmark 速率！

        return robust_min_rate, true_min_rate, opt_time


class TraditionalSDMA:
    """
    传统 MU-MIMO (纯 SDMA) 基线
    不进行流拆分 (Pc=0)，接收端不使用 SIC，直接将其他用户视为噪声
    使用 RZF 预编码和数值优化分配私有流功率
    """

    def __init__(self, K, Nt, Nr, P_max):
        self.K = K
        self.Nt = Nt
        self.Nr = Nr
        self.P_max = P_max

    def optimize_and_evaluate(self, H_hat, epsilon, sigma2, H_true=None):
        start_time = time.time()

        # SDMA 只有私有流预编码
        W_p_dir = np.zeros((self.Nt, self.K), dtype=complex)
        H_cat = np.concatenate(H_hat, axis=1)
        RZF_mat = H_cat.conj().T @ la.inv(H_cat @ H_cat.conj().T + sigma2 * np.eye(self.Nr, dtype=complex))

        for k in range(self.K):
            w_p = RZF_mat[k * self.Nt: (k + 1) * self.Nt, k]
            norm_val = la.norm(w_p)
            if norm_val > 1e-12:
                W_p_dir[:, k] = w_p / norm_val

        # 辅助函数：给定私有功率，计算鲁棒组合器和速率
        def compute_robust_combiners_and_rate(powers):
            W_p = W_p_dir * np.sqrt(powers)[None, :]
            rates = np.zeros(self.K)
            combiners = {}

            for i in range(self.K):
                w_i = W_p[:, i]
                H_i_hat = H_hat[i]
                eps_i = epsilon[i]

                R_interf = sigma2 * np.eye(self.Nr, dtype=complex)
                # SDMA 不做 SIC，所有其他用户都是干扰
                for j in range(self.K):
                    if i != j:
                        w_j = W_p[:, j]
                        H_j_hat = H_hat[j]
                        eps_j = epsilon[j]
                        R_interf += H_j_hat @ np.outer(w_j, w_j.conj()) @ H_j_hat.conj().T
                        R_interf += (eps_j ** 2 * la.norm(w_j) ** 2) * np.eye(self.Nr)

                R_total = R_interf + H_i_hat @ np.outer(w_i, w_i.conj()) @ H_i_hat.conj().T + (
                        eps_i ** 2 * la.norm(w_i) ** 2) * np.eye(self.Nr)
                try:
                    u_i = la.inv(R_total) @ H_i_hat @ w_i
                except la.LinAlgError:
                    u_i = np.zeros(self.Nr, dtype=complex)

                combiners[i] = u_i
                signal_power = np.abs(u_i.conj().T @ H_i_hat @ w_i) ** 2
                interf_power = np.real(u_i.conj().T @ R_interf @ u_i) + eps_i ** 2 * la.norm(u_i) ** 2 * la.norm(
                    w_i) ** 2

                if interf_power > 0:
                    rates[i] = np.log2(1 + signal_power / interf_power)

            return np.min(rates), W_p, combiners

        # 优化目标 (仅优化 K 个私有流功率)
        def objective(powers):
            min_rate, _, _ = compute_robust_combiners_and_rate(powers)
            return -min_rate

        constraints = [{'type': 'ineq', 'fun': lambda p: self.P_max - np.sum(p)}]
        bounds = [(0, self.P_max) for _ in range(self.K)]
        p0 = np.ones(self.K) * (self.P_max / self.K)

        res = minimize(objective, p0, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 50, 'ftol': 1e-3})
        opt_time = time.time() - start_time

        robust_min_rate, W_p_opt, combiners_opt = compute_robust_combiners_and_rate(res.x)

        # 真实物理速率验证 (严格包含 H_true - H_hat 的残余误差)
        true_min_rate = robust_min_rate
        if H_true is not None:
            true_rates = np.zeros(self.K)
            for i in range(self.K):
                w_i = W_p_opt[:, i]
                u_i = combiners_opt[i]
                H_i_true = H_true[i]

                R_interf_true = sigma2 * np.eye(self.Nr, dtype=complex)
                for j in range(self.K):
                    if i != j:
                        w_j = W_p_opt[:, j]
                        H_j_true = H_true[j]
                        R_interf_true += H_j_true @ np.outer(w_j, w_j.conj()) @ H_j_true.conj().T

                # SDMA 这里没有 SIC，所以不存在解码后的流，但电磁波在真实信道中传播的本质不变
                signal_power_true = np.abs(u_i.conj().T @ H_i_true @ w_i) ** 2
                interf_power_true = np.real(u_i.conj().T @ R_interf_true @ u_i)

                if interf_power_true > 0:
                    true_rates[i] = np.log2(1 + signal_power_true / interf_power_true)
            true_min_rate = np.min(true_rates)

        return robust_min_rate, true_min_rate, opt_time


class EPARSMA:
    """
    均分功率 RSMA (EPA-RSMA) 基线
    不做迭代优化，直接将功率平均分配给所有公共流和私有流
    解码顺序固定为：先解所有公共流，再解私有流
    """

    def __init__(self, K, Nt, Nr, P_max):
        self.K = K
        self.Nt = Nt
        self.Nr = Nr
        self.P_max = P_max

    def optimize_and_evaluate(self, H_hat, epsilon, sigma2, H_true=None):
        start_time = time.time()

        W_c_dir = np.zeros((self.Nt, self.K), dtype=complex)
        W_p_dir = np.zeros((self.Nt, self.K), dtype=complex)

        H_cat = np.concatenate(H_hat, axis=1)
        RZF_mat = H_cat.conj().T @ la.inv(H_cat @ H_cat.conj().T + sigma2 * np.eye(self.Nr, dtype=complex))

        for k in range(self.K):
            U, S, Vh = la.svd(H_hat[k], full_matrices=False)
            W_c_dir[:, k] = Vh[0, :].conj()
            w_p = RZF_mat[k * self.Nt: (k + 1) * self.Nt, k]
            norm_val = la.norm(w_p)
            if norm_val > 1e-12:
                W_p_dir[:, k] = w_p / norm_val

        # 核心：功率均分，无需优化
        P_equal = self.P_max / (2 * self.K)
        powers = np.ones(2 * self.K) * P_equal

        P_c = powers[:self.K]
        P_p = powers[self.K:]

        W_c = W_c_dir * np.sqrt(P_c)[None, :]
        W_p = W_p_dir * np.sqrt(P_p)[None, :]

        active_sic = list(range(self.K)) + list(range(self.K, 2 * self.K))
        rates_robust = np.zeros(2 * self.K)
        combiners = {}

        # 计算鲁棒速率
        for idx, stream_id in enumerate(active_sic):
            user_idx = stream_id % self.K
            is_common = stream_id < self.K

            w_i = W_c[:, user_idx] if is_common else W_p[:, user_idx]
            H_i_hat = H_hat[user_idx]
            eps_i = epsilon[user_idx]

            R_interf = sigma2 * np.eye(self.Nr, dtype=complex)

            # 1. 未解码流带来的干扰 (估计信道干扰 + 最坏情况不确定性)
            for j_id in active_sic[idx + 1:]:
                j_user = j_id % self.K
                w_j = W_c[:, j_user] if j_id < self.K else W_p[:, j_user]
                H_j_hat = H_hat[j_user]
                eps_j = epsilon[j_user]
                R_interf += H_j_hat @ np.outer(w_j, w_j.conj()) @ H_j_hat.conj().T
                R_interf += (eps_j ** 2 * la.norm(w_j) ** 2) * np.eye(self.Nr)

            # 2. 已解码流带来的鲁棒残余干扰界限 (补上缺失的这一环！)
            for j_id in active_sic[:idx]:
                j_user = j_id % self.K
                w_j = W_c[:, j_user] if j_id < self.K else W_p[:, j_user]
                eps_j = epsilon[j_user]
                R_interf += (eps_j ** 2 * la.norm(w_j) ** 2) * np.eye(self.Nr)

            R_total = R_interf + H_i_hat @ np.outer(w_i, w_i.conj()) @ H_i_hat.conj().T + (
                        eps_i ** 2 * la.norm(w_i) ** 2) * np.eye(self.Nr)
            try:
                u_i = la.inv(R_total) @ H_i_hat @ w_i
            except la.LinAlgError:
                u_i = np.zeros(self.Nr, dtype=complex)

            combiners[stream_id] = u_i
            signal_power = np.abs(u_i.conj().T @ H_i_hat @ w_i) ** 2
            interf_power = np.real(u_i.conj().T @ R_interf @ u_i) + eps_i ** 2 * la.norm(u_i) ** 2 * la.norm(w_i) ** 2

            if interf_power > 0:
                rates_robust[stream_id] = np.log2(1 + signal_power / interf_power)

        robust_R_users = [rates_robust[k] + rates_robust[self.K + k] for k in range(self.K)]
        robust_min_rate = np.min(robust_R_users)
        opt_time = time.time() - start_time

        # 计算真实物理速率 (严格包含残余误差)
        true_min_rate = robust_min_rate
        if H_true is not None:
            true_rates = np.zeros(2 * self.K)
            for idx, stream_id in enumerate(active_sic):
                user_idx = stream_id % self.K
                is_common = stream_id < self.K

                w_i = W_c[:, user_idx] if is_common else W_p[:, user_idx]
                u_i = combiners[stream_id]
                H_i_true = H_true[user_idx]

                R_interf_true = sigma2 * np.eye(self.Nr, dtype=complex)
                # 1. 未解码流
                for j_id in active_sic[idx + 1:]:
                    j_user = j_id % self.K
                    w_j = W_c[:, j_user] if j_id < self.K else W_p[:, j_user]
                    H_j_true = H_true[j_user]
                    R_interf_true += H_j_true @ np.outer(w_j, w_j.conj()) @ H_j_true.conj().T

                # 2. 已解码流留下的真实残余误差
                for j_id in active_sic[:idx]:
                    j_user = j_id % self.K
                    w_j = W_c[:, j_user] if j_id < self.K else W_p[:, j_user]
                    H_err = H_true[j_user] - H_hat[j_user]
                    R_interf_true += H_err @ np.outer(w_j, w_j.conj()) @ H_err.conj().T

                signal_power_true = np.abs(u_i.conj().T @ H_i_true @ w_i) ** 2
                interf_power_true = np.real(u_i.conj().T @ R_interf_true @ u_i)

                if interf_power_true > 0:
                    true_rates[stream_id] = np.log2(1 + signal_power_true / interf_power_true)

            True_R_users = [true_rates[k] + true_rates[self.K + k] for k in range(self.K)]
            true_min_rate = np.min(True_R_users)

        return robust_min_rate, true_min_rate, opt_time