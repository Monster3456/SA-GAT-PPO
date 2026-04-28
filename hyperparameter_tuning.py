import numpy as np
import torch
import matplotlib.pyplot as plt
from rsma_env import UplinkRSMAEnv
from ppo_agent import PPOAgent


# ==========================================
# 辅助函数：平滑学习曲线 (顶刊绘图必备)
# ==========================================
def smooth_curve(data, window_size=50):
    padded = np.pad(data, (window_size // 2, window_size - 1 - window_size // 2), mode='edge')
    return np.convolve(padded, np.ones(window_size) / window_size, mode='valid')


# ==========================================
# 迷你训练循环 (用于快速参数扫描)
# ==========================================
def run_tuning_session(env_params, agent_params, max_episodes=2500, steps_per_ep=100):
    env = UplinkRSMAEnv(**env_params)

    # 提取并删除不属于 PPOAgent 初始化的 kwargs
    init_ent = agent_params.pop('ent_coef_init', 0.05)
    # 提取 batch_size，默认为 256
    current_batch_size = agent_params.pop('batch_size', 256)

    agent = PPOAgent(K=env.K, Nt=env.Nt, Nr=env.Nr, action_dim=env.action_space.shape[0], **agent_params)

    update_timestep = 2000  # 扫描时为了提速，缩短轨迹收集长度
    time_step = 0
    rollouts = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'values': []}
    history_norm_rewards = []

    for ep in range(max_episodes):
        state, _ = env.reset(options={'snr_db': 20.0, 'epsilon': 0.15})
        ep_norm_reward_sum = 0
        current_eta = 0.0

        # 动态熵系数
        current_ent_coef = max(0.001, init_ent * (0.995 ** ep))

        for t in range(steps_per_ep):
            action, log_prob, value = agent.get_action(state)
            next_state, reward, done, _, info = env.step(action, eta=current_eta)

            rollouts['states'].append(state)
            rollouts['actions'].append(action)
            rollouts['log_probs'].append(log_prob)
            rollouts['rewards'].append(reward)
            rollouts['values'].append(value)

            ep_norm_reward_sum += reward
            state = next_state
            time_step += 1

            if time_step % update_timestep == 0:
                _, _, next_value = agent.get_action(state)
                values_plus = rollouts['values'] + [next_value]
                gae = 0
                returns, advantages = [], []
                for step in reversed(range(len(rollouts['rewards']))):
                    delta = rollouts['rewards'][step] + 0.99 * values_plus[step + 1] - values_plus[step]
                    gae = delta + 0.99 * 0.95 * gae
                    advantages.insert(0, gae)
                    returns.insert(0, gae + values_plus[step])

                rollouts['returns'] = returns
                rollouts['advantages'] = advantages

                # ==========================================
                # 核心执行：将 batch_size 传入更新函数
                # ==========================================
                agent.update(rollouts, ent_coef=current_ent_coef, batch_size=current_batch_size)
                rollouts = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'values': []}

        history_norm_rewards.append(ep_norm_reward_sum / steps_per_ep)
        if (ep + 1) % 200 == 0:
            print(f"  -> Episode {ep + 1}/{max_episodes} | Avg Reward: {np.mean(history_norm_rewards[-50:]):.3f}")

    return history_norm_rewards


if __name__ == "__main__":
    print("🚀 开始 SA-GAT-PPO 4D 超参数敏感性分析 (包含 Batch Size)...")

    env_params = {'K': 4, 'Nt': 2, 'Nr': 4}
    test_episodes = 2500

    # 1. 扫描 Learning Rate
    lr_candidates = [3e-6,1e-5, 1e-4, 1e-3]
    lr_results = {}
    print("\n--- [1/4] Sweeping Actor Learning Rate ---")
    for lr in lr_candidates:
        print(f"\nTesting LR = {lr}")
        agent_params = {'model_type': 'transformer', 'lr': lr, 'clip_ratio': 0.2, 'ent_coef_init': 0.01, 'batch_size': 512}
        lr_results[lr] = run_tuning_session(env_params, agent_params, max_episodes=test_episodes)

    # 2. 扫描 Clip Ratio
    clip_candidates = [0.1, 0.15, 0.2, 0.3]
    clip_results = {}
    print("\n--- [2/4] Sweeping Clip Ratio ---")
    for clip in clip_candidates:
        print(f"\nTesting Clip Ratio = {clip}")
        agent_params = {'model_type': 'transformer', 'lr': 1e-5, 'clip_ratio': clip, 'ent_coef_init': 0.01, 'batch_size': 512}
        clip_results[clip] = run_tuning_session(env_params, agent_params, max_episodes=test_episodes)

    # 3. 扫描 Entropy Coefficient
    ent_candidates = [0.005,0.01, 0.05, 0.1]
    ent_results = {}
    print("\n--- [3/4] Sweeping Initial Entropy ---")
    for ent in ent_candidates:
        print(f"\nTesting Init Entropy = {ent}")
        agent_params = {'model_type': 'transformer', 'lr': 1e-5, 'clip_ratio': 0.2, 'ent_coef_init': ent, 'batch_size': 512}
        ent_results[ent] = run_tuning_session(env_params, agent_params, max_episodes=test_episodes)

    # 4. 扫描 Mini-batch Size
    batch_candidates = [64, 128, 256, 512]
    batch_results = {}
    print("\n--- [4/4] Sweeping Mini-batch Size ---")
    for bz in batch_candidates:
        print(f"\nTesting Batch Size = {bz}")
        agent_params = {'model_type': 'transformer', 'lr': 1e-5, 'clip_ratio': 0.2, 'ent_coef_init': 0.01, 'batch_size': bz}
        batch_results[bz] = run_tuning_session(env_params, agent_params, max_episodes=test_episodes)

    # ==========================================
    # 绘制 2x2 绝美排版对比图
    # ==========================================
    # 1. 学习率敏感性
    plt.figure(figsize=(8, 6))
    for lr, data in lr_results.items():
        smoothed = smooth_curve(data)
        plt.plot(smoothed, label=f'$lr_\\pi = {lr}$')
    plt.ylabel('Average Normalized Reward', fontsize=12)
    plt.xlabel('Training Episodes', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('hyperparameter_lr.pdf')  # 保存为 PDF（矢量，支持透明度）
    plt.show()

    # 2. PPO Clip Ratio 敏感性
    plt.figure(figsize=(8, 6))
    for clip, data in clip_results.items():
        smoothed = smooth_curve(data)
        plt.plot(smoothed, label=f'$\\epsilon_{{clip}} = {clip}$')
    plt.ylabel('Average Normalized Reward', fontsize=12)
    plt.xlabel('Training Episodes', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('hyperparameter_clip.pdf')
    plt.show()

    # 3. 初始熵系数敏感性
    plt.figure(figsize=(8, 6))
    for ent, data in ent_results.items():
        smoothed = smooth_curve(data)
        plt.plot(smoothed, label=f'$c_{{ent}} = {ent}$')
    plt.ylabel('Average Normalized Reward', fontsize=12)
    plt.xlabel('Training Episodes', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('hyperparameter_entropy.pdf')
    plt.show()

    # 4. Mini-batch Size 敏感性
    plt.figure(figsize=(8, 6))
    for bz, data in batch_results.items():
        smoothed = smooth_curve(data)
        plt.plot(smoothed, label=f'Batch Size = {bz}')
    plt.ylabel('Average Normalized Reward', fontsize=12)
    plt.xlabel('Training Episodes', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('hyperparameter_batchsize.pdf')
    plt.show()