
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from rsma_env import UplinkRSMAEnv
from ppo_agent import PPOAgent


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


# 核心修改：传入 weight_path 确保不同模型权重隔离
def train(env, agent, max_episodes=2000, steps_per_ep=100, weight_path="rsma_ppo_unified_weights.pth"):
    print(f"🚀 开始训练模型 [{agent.model_type.upper()}] | 具备方差感知早停机制...")

    update_timestep = 1000  # 核心修改：每 1000 步 (约 10 个不同 SNR 的回合) 更新一次网络
    time_step = 0

    rollouts = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'values': []}

    history_raw_rewards = []
    history_norm_rewards = []

    window_size = 50
    std_tol = 0.5  # 由于奖励现在是 [0, 1] 左右的相对容量，标准差阈值需降到很小
    min_episodes = 200

    for ep in range(max_episodes):
        state, _ = env.reset()

        ep_raw_reward_sum = 0
        ep_norm_reward_sum = 0
        current_eta = max(0.05 * (1 - ep / max_episodes), 0.005)

        for t in range(steps_per_ep):
            action, log_prob, value = agent.get_action(state)

            # 这里的 reward 已经是 rsma_env 里算好的香农归一化 reward
            next_state, reward, done, _, info = env.step(action, eta=current_eta)
            raw_reward = info['min_rate_true']  # 提取物理奖励用于打印日志

            rollouts['states'].append(state)
            rollouts['actions'].append(action)
            rollouts['log_probs'].append(log_prob)
            rollouts['rewards'].append(reward)  # 直接使用环境发出的无滞后归一化奖励
            rollouts['values'].append(value)

            ep_raw_reward_sum += raw_reward
            ep_norm_reward_sum += reward
            state = next_state
            time_step += 1

            # 核心机制：达到 Update Timestep 时才进行 PPO 更新
            if time_step % update_timestep == 0:
                _, _, next_value = agent.get_action(state)
                returns, advantages = compute_gae(rollouts['rewards'], rollouts['values'], next_value)
                rollouts['returns'] = returns
                rollouts['advantages'] = advantages

                agent.update(rollouts)

                # 更新完毕后清空经验池
                rollouts = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'values': []}

        history_raw_rewards.append(ep_raw_reward_sum)
        history_norm_rewards.append(ep_norm_reward_sum)

        if (ep + 1) % 10 == 0:
            recent_raw = history_raw_rewards[-10:]
            recent_norm = history_norm_rewards[-10:]
            # 现在的 Norm Avg 应该在稳定在十几到几十左右 (100步累积的效率)
            print(f"Episode {ep + 1:4d} | SNR: {env.current_snr_db:4.1f}dB | "
                  f"Raw Avg: {np.mean(recent_raw):6.2f} || "
                  f"Norm Avg: {np.mean(recent_norm):6.2f} | Norm Std: {np.std(recent_norm):5.2f}")

        # 早停检测
        if ep >= min_episodes and len(history_norm_rewards) >= window_size:
            current_norm_std = np.std(history_norm_rewards[-window_size:])
            if current_norm_std < std_tol:
                print(f"\n🛑 触发方差早停！最终 Norm Std: {current_norm_std:.4f}")
                break

    #保存权重时使用传入的安全路径
    torch.save(agent.policy.state_dict(), weight_path)
    print(f"✅ 训练完成，权重已保存至: {weight_path}")


# 核心修改：传入 weight_path 确保加载正确的模型
def test_snr_curve(env, agent, test_steps_per_snr=200, weight_path="rsma_ppo_unified_weights.pth"):
    print(f"🔬 开始 SNR 性能验证测试 [{agent.model_type.upper()}]...")

    # 加载对应的权重
    agent.policy.load_state_dict(torch.load(weight_path, weights_only=True))
    agent.policy.eval()

    # 典型的验证信噪比网格
    snr_grid = [0, 5, 10, 15, 20, 25, 30]
    avg_min_rate_results = []
    true_min_rates=[]

    for snr in snr_grid:
        # 强制环境初始化在这个特定的 SNR 下
        state, _ = env.reset(options={'snr_db': snr})

        min_rates_at_snr = []
        true_min_rates_at_snr=[]
        for _ in range(test_steps_per_snr):
            state_ts = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                # 直接调用 forward，兼容所有网络架构
                dist, _ = agent.policy(state_ts)
                # 取出高斯分布的均值作为测试时的确定性动作
                action = dist.mean.squeeze(0).numpy()

            state, _, _, _, info = env.step(action, eta=0.0)
            min_rates_at_snr.append(info['min_rate_robust'])
            true_min_rates_at_snr.append(info['min_rate_true'])

        # 统计该 SNR 下所有 200 个快变信道时隙的平均最低速率
        mean_val = np.mean(min_rates_at_snr)
        true_min_rates.append(np.mean(true_min_rates_at_snr))
        avg_min_rate_results.append(mean_val)
        print(f"验证完成 -> SNR: {snr} dB | Average Minimum Rate: {mean_val:.4f} bps/Hz")

    # ==========================================
    # 可视化 1：鲁棒预估速率 vs 真实物理速率
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.plot(true_min_rates, label='Actual Rate (Real Channel)', color='g', linewidth=2, alpha=0.9)
    plt.plot(avg_min_rate_results, label='Robust Rate (Estimated Lower Bound)', color='b', linewidth=2, linestyle='--')

    # 填充两条线之间的区域，代表我们设计的“安全裕度(Safety Margin)”
    plt.fill_between(range(len(snr_grid)), avg_min_rate_results, true_min_rates, color='green', alpha=0.1, label='Safety Margin')

    plt.xlabel('Time Slots (1ms)', fontsize=12)
    plt.ylabel('Minimum User Rate (bps/Hz)', fontsize=12)
    plt.title('Robustness Verification under 240Hz Doppler Shift (SNR = 20dB)', fontsize=14)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    # 新增模型切换参数，默认设为 mlp 以便你立刻开始消融实验
    parser.add_argument('--model', type=str, default='transformer', choices=['transformer', 'mlp'])
    args = parser.parse_args()

    env = UplinkRSMAEnv(K=4, Nt=2, Nr=4)
    # 将命令行参数传入 Agent
    agent = PPOAgent(K=4, Nt=2, Nr=4, action_dim=env.action_space.shape[0], model_type=args.model)

    # 自动分配安全的文件名，防止互相覆盖
    weight_file = f"rsma_ppo_{args.model}_weights.pth"

    if args.mode == 'train':
        # 纯 MLP 网络因为没有 Attention 提取拓扑，可能需要更多的轮数才能把高维空间试错完
        train(env, agent, max_episodes=20000, weight_path=weight_file)
    else:
        test_snr_curve(env, agent, weight_path=weight_file)