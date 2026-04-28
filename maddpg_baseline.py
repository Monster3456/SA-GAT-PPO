import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy


class LiteratureMADDPGActor(nn.Module):
    """
    严格复现文献的 Decentralized Actor (去中心化执行)
    文献说明: 采用 4 层全连接层 (Fully Connected Layers)
    输入: 该用户(智能体)的局部状态 (Local Observation)
    输出: 该用户的 5 个动作
    """

    def __init__(self, obs_dim_per_agent, action_dim_per_agent):
        super(LiteratureMADDPGActor, self).__init__()
        # 严格的 4 层 MLP 架构
        self.net = nn.Sequential(
            nn.Linear(obs_dim_per_agent, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim_per_agent),
            nn.Tanh()
        )

    def forward(self, local_obs):
        # local_obs 形状: [batch_size, K, obs_dim_per_agent]
        batch_size, K, _ = local_obs.shape

        # 共享参数处理每个智能体的局部观测
        actions = self.net(local_obs)  # 形状: [batch_size, K, 5]

        # 张量转置以对齐 RSMA 环境的解包逻辑: a_tilde, rho, alpha, z_scores
        aligned_actions = actions.transpose(1, 2).contiguous()
        return aligned_actions.view(batch_size, -1)


class LiteratureMADDPGCritic(nn.Module):
    """
    严格复现文献的 Centralized Critic (集中式训练)
    文献说明: 采用 4 层全连接层
    输入: 全局状态 (Global State) + 所有智能体的联合动作 (Joint Action)
    输出: 全局 Q 值
    """

    def __init__(self, global_obs_dim, total_action_dim):
        super(LiteratureMADDPGCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim + total_action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, global_obs, joint_action):
        # 将所有状态和动作拼接
        xu = torch.cat([global_obs, joint_action], dim=1)
        return self.net(xu)


class ReplayBuffer:
    def __init__(self, max_size=50000):
        self.buffer = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, state, action, reward, next_state):
        if len(self.buffer) < self.max_size:
            self.buffer.append((state, action, reward, next_state))
        else:
            self.buffer[self.ptr] = (state, action, reward, next_state)
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.buffer), size=batch_size)
        states, actions, rewards, next_states = [], [], [], []
        for i in ind:
            s, a, r, n_s = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(n_s)
        return (torch.FloatTensor(np.array(states)),
                torch.FloatTensor(np.array(actions)),
                torch.FloatTensor(np.array(rewards)).reshape(-1, 1),
                torch.FloatTensor(np.array(next_states)))


class LiteratureMADDPGAgent:
    """
    文献基线: MA-DDPG 算法封装
    """

    def __init__(self, K, Nt, Nr, total_action_dim, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.01):
        self.K = K
        self.obs_dim_per_agent = 2 * Nr * Nt + 2
        self.global_obs_dim = self.obs_dim_per_agent * K
        self.total_action_dim = total_action_dim
        self.action_dim_per_agent = total_action_dim // K

        self.gamma = gamma
        self.tau = tau
        self.model_type = 'maddpg_literature'

        self.actor = LiteratureMADDPGActor(self.obs_dim_per_agent, self.action_dim_per_agent)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = LiteratureMADDPGCritic(self.global_obs_dim, self.total_action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.replay_buffer = ReplayBuffer()

    def get_action(self, state, explore_noise=0.1):
        # 将一维全局 state 变形为 [batch, K, obs_per_agent] 供去中心化 Actor 使用
        state_ts = torch.FloatTensor(state).view(1, self.K, self.obs_dim_per_agent)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_ts).squeeze(0).numpy()
        self.actor.train()

        # DDPG 的探索机制
        noise = np.random.normal(0, explore_noise, size=self.total_action_dim)
        action = np.clip(action + noise, -1.0, 1.0)

        return action, 0.0, 0.0  # 返回格式兼容环境的解包

    def update(self, batch_size=128):
        if len(self.replay_buffer.buffer) < batch_size:
            return

        states, actions, rewards, next_states = self.replay_buffer.sample(batch_size)

        # 将全局状态 reshape 为 local_obs 供 target_actor 使用
        next_states_local = next_states.view(batch_size, self.K, self.obs_dim_per_agent)
        states_local = states.view(batch_size, self.K, self.obs_dim_per_agent)

        # ---------------- 更新 Critic ----------------
        with torch.no_grad():
            next_actions = self.actor_target(next_states_local)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + self.gamma * target_Q

        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ---------------- 更新 Actor ----------------
        actor_loss = -self.critic(states, self.actor(states_local)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # ---------------- 软更新 ----------------
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)