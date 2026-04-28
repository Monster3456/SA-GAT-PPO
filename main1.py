import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from rsma_env import UplinkRSMAEnv
from ppo_agent import PPOAgent
from maddpg_baseline import LiteratureMADDPGAgent  # 导入纯正的文献复现基线


def compute_gae(rewards, values, next_value, gamma=0.99, lam=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    advantages = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] - values[step]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[step])
    return returns, advantages


def train(env, agent, max_episodes=2000, steps_per_ep=100, weight_path="weights.pth"):
    print(f"🚀 开始训练模型 [{agent.model_type.upper()}] | 具备方差感知早停机制...")

    update_timestep = 1000
    time_step = 0

    rollouts = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'values': []}
    history_raw_rewards = []
    history_norm_rewards = []

    window_size = 50
    std_tol = 0.5
    min_episodes = 200

    for ep in range(max_episodes):
        state, _ = env.reset()
        ep_raw_reward_sum = 0
        ep_norm_reward_sum = 0
        current_eta = max(0.05 * (1 - ep / max_episodes), 0.005)

        for t in range(steps_per_ep):
            # ==========================================
            # 1. 动作获取 (兼容 PPO 和 DDPG)
            # ==========================================
            if 'ppo' in agent.model_type or agent.model_type in ['transformer', 'mlp']:
                action, log_prob, value = agent.get_action(state)
            else:  # MADDPG
                # DDPG 添加高斯探索噪声
                action, log_prob, value = agent.get_action(state, explore_noise=0.1)

            # ==========================================
            # 2. 环境交互 (所有算法共享完全一样的 BCD 奖励！)
            # ==========================================
            next_state, reward, done, _, info = env.step(action, eta=current_eta)
            raw_reward = info['min_rate_true']

            # ==========================================
            # 3. 经验存储与网络更新 (架构分流)
            # ==========================================
            if 'ppo' in agent.model_type or agent.model_type in ['transformer', 'mlp']:
                # --- PPO (On-Policy) 逻辑 ---
                rollouts['states'].append(state)
                rollouts['actions'].append(action)
                rollouts['log_probs'].append(log_prob)
                rollouts['rewards'].append(reward)
                rollouts['values'].append(value)

                if (time_step + 1) % update_timestep == 0:
                    _, _, next_value = agent.get_action(next_state)
                    returns, advantages = compute_gae(rollouts['rewards'], rollouts['values'], next_value)
                    rollouts['returns'] = returns
                    rollouts['advantages'] = advantages
                    agent.update(rollouts)
                    rollouts = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'values': []}
            else:
                # --- MADDPG (Off-Policy) 逻辑 ---
                # 每步存入回放池，每步都进行小批量更新
                agent.replay_buffer.add(state, action, reward, next_state)
                agent.update(batch_size=128)

            ep_raw_reward_sum += raw_reward
            ep_norm_reward_sum += reward
            state = next_state
            time_step += 1

        history_raw_rewards.append(ep_raw_reward_sum)
        history_norm_rewards.append(ep_norm_reward_sum)

        if (ep + 1) % 10 == 0:
            recent_raw = history_raw_rewards[-10:]
            recent_norm = history_norm_rewards[-10:]
            print(f"Episode {ep + 1:4d} | SNR: {env.current_snr_db:4.1f}dB | "
                  f"Raw Avg: {np.mean(recent_raw):6.2f} || "
                  f"Norm Avg: {np.mean(recent_norm):6.2f} | Norm Std: {np.std(recent_norm):5.2f}")

        # 早停检测
        if ep >= min_episodes and len(history_norm_rewards) >= window_size:
            current_norm_std = np.std(history_norm_rewards[-window_size:])
            if current_norm_std < std_tol:
                print(f"\n🛑 触发方差早停！最终 Norm Std: {current_norm_std:.4f}")
                break

    # ==========================================
    # 4. 模型保存 (兼容不同架构的提取)
    # ==========================================
    if 'ppo' in agent.model_type or agent.model_type in ['transformer', 'mlp']:
        torch.save(agent.policy.state_dict(), weight_path)
    else:
        # MADDPG 需要分别保存 Actor 和 Critic，但为了简便，测试时我们只加载 Actor
        torch.save(agent.actor.state_dict(), weight_path)

    print(f"✅ 训练完成，权重已保存至: {weight_path}")


def test_snr_curve(env, agent, test_steps_per_snr=200, weight_path="weights.pth"):
    print(f"🔬 开始 SNR 性能验证测试 [{agent.model_type.upper()}]...")

    # 加载权重 (兼容 PPO 和 DDPG)
    if 'ppo' in agent.model_type or agent.model_type in ['transformer', 'mlp']:
        agent.policy.load_state_dict(torch.load(weight_path, weights_only=True))
        agent.policy.eval()
    else:
        agent.actor.load_state_dict(torch.load(weight_path, weights_only=True))
        agent.actor.eval()

    snr_grid = [0, 5, 10, 15, 20, 25, 30]
    avg_min_rate_results = []
    true_min_rates = []

    for snr in snr_grid:
        state, _ = env.reset(options={'snr_db': snr})
        min_rates_at_snr = []
        true_min_rates_at_snr = []

        for _ in range(test_steps_per_snr):
            state_ts = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                # 获取确定性动作 (兼容接口)
                if 'ppo' in agent.model_type or agent.model_type in ['transformer', 'mlp']:
                    dist, _ = agent.policy(state_ts)
                    action = dist.mean.squeeze(0).numpy()
                else:
                    # 对于 MADDPG，直接获取 Actor 输出，不加探索噪声
                    # MADDPG 的 Actor 需要的输入形状是 [batch, K, obs_per_agent]
                    state_ts_local = state_ts.view(1, env.K, agent.obs_dim_per_agent)
                    action = agent.actor(state_ts_local).squeeze(0).numpy()

            state, _, _, _, info = env.step(action, eta=0.0)
            min_rates_at_snr.append(info['min_rate_robust'])
            true_min_rates_at_snr.append(info['min_rate_true'])

        mean_val = np.mean(min_rates_at_snr)
        true_min_rates.append(np.mean(true_min_rates_at_snr))
        avg_min_rate_results.append(mean_val)
        print(f"验证完成 -> SNR: {snr} dB | Average True Rate: {true_min_rates[-1]:.4f} bps/Hz")

    # 绘图逻辑保持不变...
    plt.figure(figsize=(10, 6))
    plt.plot(true_min_rates, label=f'Actual Rate ({agent.model_type.upper()})', color='g', linewidth=2, alpha=0.9)
    plt.plot(avg_min_rate_results, label='Robust Rate (Estimated Lower Bound)', color='b', linewidth=2, linestyle='--')
    plt.fill_between(range(len(snr_grid)), avg_min_rate_results, true_min_rates, color='green', alpha=0.1,
                     label='Safety Margin')
    plt.xlabel('Transmit SNR (dB)', fontsize=12)
    plt.ylabel('Minimum User Rate (bps/Hz)', fontsize=12)
    plt.title(f'Robustness Verification: {agent.model_type.upper()} under Imperfect CSI', fontsize=14)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    # 终极兵器库：3 种模型随意切换
    parser.add_argument('--model', type=str, default='maddpg', choices=['transformer', 'mlp', 'maddpg'])
    args = parser.parse_args()

    env = UplinkRSMAEnv(K=4, Nt=2, Nr=4)

    # 根据命令行实例化不同的 Agent
    if args.model in ['transformer', 'mlp']:
        agent = PPOAgent(K=4, Nt=2, Nr=4, action_dim=env.action_space.shape[0], model_type=args.model)
    elif args.model == 'maddpg':
        agent = LiteratureMADDPGAgent(K=4, Nt=2, Nr=4, total_action_dim=env.action_space.shape[0])

    weight_file = f"rsma_{args.model}_weights.pth"

    if args.mode == 'train':
        train(env, agent, max_episodes=20000, weight_path=weight_file)
    else:
        test_snr_curve(env, agent, weight_path=weight_file)